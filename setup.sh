#!/usr/bin/env bash
set -Eeuo pipefail

ROLE="preproc"
DEFAULT_SWAP_GB="128"
DATA_MOUNT="${DATA_MOUNT:-/data}"
DATA_ROOT="${AINA_DATA_ROOT:-$DATA_MOUNT/aina-code}"
SWAP_GB="${SWAP_GB:-$DEFAULT_SWAP_GB}"
SWAPPINESS="${SWAPPINESS:-10}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-3}"
TARGET_USER="${SUDO_USER:-${USER:-$(id -un)}}"
TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || id -gn)"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
TARGET_HOME="${TARGET_HOME:-/home/$TARGET_USER}"
PROJECT_DIR="${AINA_PREPROC_DIR:-$TARGET_HOME/preproc-pipeline}"

log() {
  printf '[%s] %s\n' "$ROLE" "$*"
}

warn() {
  printf '[%s] WARNING: %s\n' "$ROLE" "$*" >&2
}

fail() {
  printf '[%s] ERROR: %s\n' "$ROLE" "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

confirm_format() {
  printf '\n'
  warn "This will format the listed data disk(s). Existing data on them will be lost."
  printf 'Disks selected for %s:\n' "$DATA_MOUNT"
  printf '  %s\n' "$@"

  if [ "${AINA_AUTO_CONFIRM:-0}" = "1" ]; then
    log "AINA_AUTO_CONFIRM=1, continuing without interactive prompt."
    return 0
  fi

  if [ ! -t 0 ]; then
    fail "non-interactive shell; set AINA_AUTO_CONFIRM=1 only after verifying the selected disks"
  fi

  printf 'Type FORMAT_DATA to continue: '
  read -r answer
  [ "$answer" = "FORMAT_DATA" ] || fail "format cancelled"
}

resolve_project_dir() {
  if [ -d "$PROJECT_DIR" ]; then
    printf '%s\n' "$PROJECT_DIR"
    return
  fi

  fail "missing preproc repo: $PROJECT_DIR. Clone it first or set AINA_PREPROC_DIR=/path/to/preproc-pipeline"
}

install_base_packages() {
  need_cmd sudo
  log "Installing base packages."
  run_sudo apt-get update
  run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl unzip build-essential python3 python3-venv python3-pip awscli mdadm
}

is_blank_unmounted_disk() {
  local disk="$1"

  [ -b "$disk" ] || return 1

  local mounted
  mounted="$(lsblk -nr -o MOUNTPOINT "$disk" | awk 'NF { print }' || true)"
  [ -z "$mounted" ] || return 1

  local line_count
  line_count="$(lsblk -nr -o NAME "$disk" | wc -l | tr -d ' ')"
  [ "$line_count" = "1" ] || return 1

  local signatures
  signatures="$(run_sudo wipefs -n "$disk" 2>/dev/null | awk 'NR > 1 { print }' || true)"
  [ -z "$signatures" ] || return 1

  return 0
}

find_blank_data_disks() {
  while read -r disk type removable readonly; do
    [ "$type" = "disk" ] || continue
    [ "$removable" = "0" ] || continue
    [ "$readonly" = "0" ] || continue
    if is_blank_unmounted_disk "$disk"; then
      printf '%s\n' "$disk"
    fi
  done < <(lsblk -dpno NAME,TYPE,RM,RO)
}

append_fstab_once() {
  local uuid="$1"
  local mount_point="$2"
  local fs_type="$3"

  if ! grep -q "UUID=$uuid[[:space:]]" /etc/fstab; then
    printf 'UUID=%s %s %s defaults,nofail 0 2\n' "$uuid" "$mount_point" "$fs_type" | run_sudo tee -a /etc/fstab >/dev/null
    reload_systemd_daemon
  fi
}

reload_systemd_daemon() {
  if command -v systemctl >/dev/null 2>&1; then
    run_sudo systemctl daemon-reload || warn "systemctl daemon-reload failed"
  fi
}

mount_single_disk() {
  local disk="$1"

  confirm_format "$disk"
  log "Formatting $disk as ext4."
  run_sudo mkfs.ext4 -F "$disk"

  local uuid
  uuid="$(run_sudo blkid -s UUID -o value "$disk")"
  run_sudo mkdir -p "$DATA_MOUNT"
  append_fstab_once "$uuid" "$DATA_MOUNT" "ext4"
  run_sudo mount "$DATA_MOUNT"
}

mount_raid0() {
  local disk_a="$1"
  local disk_b="$2"
  local md_device="/dev/md0"

  confirm_format "$disk_a" "$disk_b"
  log "Creating RAID0 $md_device from $disk_a and $disk_b."
  run_sudo mdadm --create "$md_device" --level=0 --raid-devices=2 "$disk_a" "$disk_b" --metadata=1.2 --force

  log "Saving mdadm config."
  run_sudo mkdir -p /etc/mdadm
  run_sudo sh -c "mdadm --detail --scan >> /etc/mdadm/mdadm.conf"
  run_sudo update-initramfs -u || warn "update-initramfs failed; RAID may still work after boot depending on distro"

  log "Formatting $md_device as ext4."
  run_sudo mkfs.ext4 -F "$md_device"

  local uuid
  uuid="$(run_sudo blkid -s UUID -o value "$md_device")"
  run_sudo mkdir -p "$DATA_MOUNT"
  append_fstab_once "$uuid" "$DATA_MOUNT" "ext4"
  run_sudo mount "$DATA_MOUNT"
}

ensure_data_mount() {
  if ! mountpoint -q "$DATA_MOUNT" && grep -q "[[:space:]]$DATA_MOUNT[[:space:]]" /etc/fstab 2>/dev/null; then
    log "$DATA_MOUNT exists in /etc/fstab; trying to mount it."
    run_sudo mount "$DATA_MOUNT" || warn "Failed to mount $DATA_MOUNT from /etc/fstab."
  fi

  if mountpoint -q "$DATA_MOUNT"; then
    log "$DATA_MOUNT is already mounted; skipping disk setup."
    return
  fi

  mapfile -t disks < <(find_blank_data_disks)

  if [ "${#disks[@]}" -eq 0 ]; then
    if [ "${AINA_ALLOW_ROOT_DATA:-0}" = "1" ]; then
      warn "No blank unmounted data disk found. Creating $DATA_MOUNT on the root filesystem because AINA_ALLOW_ROOT_DATA=1."
      run_sudo mkdir -p "$DATA_MOUNT"
      return
    fi
    fail "No blank unmounted data disk found and $DATA_MOUNT is not mounted. Mount /data manually or set AINA_ALLOW_ROOT_DATA=1."
  fi

  if [ "${#disks[@]}" -eq 1 ]; then
    mount_single_disk "${disks[0]}"
    return
  fi

  mount_raid0 "${disks[0]}" "${disks[1]}"

  if [ "${#disks[@]}" -gt 2 ]; then
    warn "More than 2 blank disks detected; only the first 2 were used for RAID0: ${disks[0]} ${disks[1]}"
  fi
}

ensure_data_dirs() {
  run_sudo mkdir -p \
    "$DATA_ROOT" \
    "$DATA_ROOT/datasets" \
    "$DATA_ROOT/training" \
    "$DATA_ROOT/tokenizers/gpt2-8k-chat" \
    "$DATA_ROOT/work" \
    "$DATA_ROOT/hf-cache" \
    "$DATA_ROOT/hf-cache/datasets"
  run_sudo chown -R "$TARGET_USER:$TARGET_GROUP" \
    "$DATA_ROOT"
  chmod -R u+rwX \
    "$DATA_ROOT"
}

ensure_swap() {
  local swap_gb
  swap_gb="$(normalize_swap_gb "$SWAP_GB")"

  if [ "$swap_gb" = "0" ]; then
    log "SWAP_GB=0, skipping swap setup."
    return
  fi

  local swap_file="$DATA_MOUNT/swapfile"
  if swapon --show=NAME --noheadings | awk '{ print $1 }' | grep -qx "$swap_file"; then
    log "Swap already active at $swap_file."
  else
    if [ ! -f "$swap_file" ]; then
      create_swapfile "$swap_file" "$swap_gb"
    elif ! swapfile_size_matches "$swap_file" "$swap_gb"; then
      warn "Existing $swap_file size does not match SWAP_GB=$swap_gb and is not active; recreating it."
      run_sudo rm -f "$swap_file"
      create_swapfile "$swap_file" "$swap_gb"
    else
      run_sudo chmod 600 "$swap_file"
      run_sudo mkswap "$swap_file"
    fi
    run_sudo chmod 600 "$swap_file"
    run_sudo swapon "$swap_file"
  fi

  if ! grep -q "^$swap_file[[:space:]]" /etc/fstab; then
    printf '%s none swap sw 0 0\n' "$swap_file" | run_sudo tee -a /etc/fstab >/dev/null
    reload_systemd_daemon
  fi

  printf 'vm.swappiness=%s\n' "$SWAPPINESS" | run_sudo tee /etc/sysctl.d/99-aina-swap.conf >/dev/null
  run_sudo sysctl -w "vm.swappiness=$SWAPPINESS" >/dev/null
}

normalize_swap_gb() {
  local value="${1:-0}"
  value="${value%G}"
  value="${value%g}"
  value="${value%GB}"
  value="${value%gb}"
  if ! printf '%s' "$value" | grep -Eq '^[0-9]+$'; then
    fail "SWAP_GB must be a whole number of GiB, e.g. SWAP_GB=8 or SWAP_GB=256"
  fi
  printf '%s\n' "$value"
}

create_swapfile() {
  local swap_file="$1"
  local swap_gb="$2"
  log "Creating ${swap_gb}G swapfile at $swap_file."
  run_sudo fallocate -l "${swap_gb}G" "$swap_file" || run_sudo dd if=/dev/zero of="$swap_file" bs=1G count="$swap_gb" status=progress
  run_sudo chmod 600 "$swap_file"
  run_sudo mkswap "$swap_file"
}

swapfile_size_matches() {
  local swap_file="$1"
  local swap_gb="$2"
  local expected actual tolerance
  expected=$((swap_gb * 1024 * 1024 * 1024))
  actual="$(stat -c '%s' "$swap_file")"
  tolerance=$((16 * 1024 * 1024))
  [ "$actual" -ge $((expected - tolerance)) ] && [ "$actual" -le $((expected + tolerance)) ]
}

ensure_shell_env() {
  local bashrc="$HOME/.bashrc"
  local marker_start="# >>> aina-code vm env >>>"
  local marker_end="# <<< aina-code vm env <<<"

  if grep -qF "$marker_start" "$bashrc" 2>/dev/null; then
    log "Refreshing AINA environment block in $bashrc."
    local tmp
    tmp="$(mktemp)"
    awk -v start="$marker_start" -v end="$marker_end" '
      $0 == start { skip = 1; next }
      $0 == end { skip = 0; next }
      !skip { print }
    ' "$bashrc" > "$tmp"
    cat "$tmp" > "$bashrc"
    rm -f "$tmp"
  else
    log "Appending AINA environment block to $bashrc."
  fi

  cat >> "$bashrc" <<EOF_BASHRC

$marker_start
export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
export HF_HOME=$DATA_ROOT/hf-cache
export HF_DATASETS_CACHE=$DATA_ROOT/hf-cache/datasets
$marker_end
EOF_BASHRC
}

setup_preproc_venv() {
  local project_dir="$1"
  [ -d "$project_dir" ] || fail "missing directory: $project_dir"

  log "Setting up preprocessing venv in $project_dir/.venv."
  cd "$project_dir"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e .
}

ensure_aina_tokenizer() {
  local project_dir="$1"
  local tokenizer_dir="$DATA_ROOT/tokenizers/gpt2-8k-chat"

  if [ -f "$tokenizer_dir/tokenizer.json" ]; then
    log "Tokenizer already exists at $tokenizer_dir."
    return
  fi

  log "Creating pinned GPT-2 tokenizer at $tokenizer_dir."
  cd "$project_dir"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  AINA_TOKENIZER_DIR="$tokenizer_dir" python - <<'PY'
import os
from pathlib import Path

from transformers import AutoTokenizer

chat_template = """{% for message in messages %}<|{{ message['role'] }}|>
{{ message['content'] }}
{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""

output = Path(os.environ["AINA_TOKENIZER_DIR"])
output.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.model_max_length = 8192
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = chat_template
tokenizer.save_pretrained(output)
PY
  run_sudo chown -R "$TARGET_USER:$TARGET_GROUP" "$tokenizer_dir"
}

print_next_steps() {
  local project_dir="$1"
  cat <<EOF_NEXT

Preprocessing VM is ready.

Set HF_TOKEN manually before running dataset builds:

  export HF_TOKEN="hf_xxx"
  export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
  export HF_HOME=$DATA_ROOT/hf-cache
  export HF_DATASETS_CACHE=$DATA_ROOT/hf-cache/datasets
  aws sts get-caller-identity

Run preprocessing:

EOF_NEXT
  printf '  cd %s\n' "$project_dir"
  cat <<'EOF_NEXT'
  source .venv/bin/activate

  python scripts/build_dataset.py --config configs/aina_code_3m_1k_pretrain.yaml --resume
  python scripts/build_dataset.py --config configs/aina_code_3m_1k_sft.yaml --resume

  python scripts/build_dataset.py --config configs/aina_code_50m_2k_pretrain.yaml --resume
  python scripts/build_dataset.py --config configs/aina_code_50m_2k_sft.yaml --resume

  python scripts/build_dataset.py --config configs/aina_code_500m_8k_pretrain.yaml --resume
  python scripts/build_dataset.py --config configs/aina_code_500m_8k_sft.yaml --resume
EOF_NEXT
}

main() {
  local project_dir
  project_dir="$(resolve_project_dir)"

  log "Preproc repo: $project_dir"
  install_base_packages
  ensure_data_mount
  ensure_data_dirs
  ensure_swap
  ensure_shell_env
  setup_preproc_venv "$project_dir"
  ensure_aina_tokenizer "$project_dir"
  print_next_steps "$project_dir"
}

main "$@"
