import json
import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aina_preproc.config import PipelineConfig, SourceConfig
from aina_preproc.pipeline import run_pipeline
from aina_preproc.tokenize import TokenizerBundle


class FakeTokenizer:
    eos_token_id = 1
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return [(ord(char) % 200) + 2 for char in text]

    def save_pretrained(self, save_directory):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_tokenizer.json").write_text("{}")


class SftPipelineSmokeTests(unittest.TestCase):
    def test_pipeline_writes_sft_jsonl_shards(self):
        rows = [
            {
                "instruction": "Write a Python function that adds two numbers and returns the result.",
                "response": "def add(a, b):\n    return a + b\n",
            },
            {
                "instruction": "Explain what a list comprehension does in Python with a code example.",
                "response": "A list comprehension builds a list from an iterable, for example: squares = [x*x for x in range(5)].",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = PipelineConfig(
                project_name="unit-sft",
                output_mode="sft_jsonl",
                artifact_format="jsonl_messages",
                target_tokens=64,
                sequence_length=8,
                val_ratio=0.0,
                seed=7,
                tokenizer_path=str(root / "missing-tokenizer"),
                fallback_tokenizer="fake",
                output_dir=str(root / "sft"),
                work_dir=str(root / "work"),
                progress_path=str(root / "work" / "progress.json"),
                report_path=str(root / "reports" / "dataset_report.json"),
                s3_output=None,
                checkpoint_interval_tokens=16,
                sft_samples_per_shard=1,
                sources=(
                    SourceConfig(
                        name="unit_instruct",
                        type="instruct",
                        mix_role="instruct_only",
                        hf_id="local",
                        target_tokens=64,
                    ),
                ),
            )

            with patch(
                "aina_preproc.pipeline.load_tokenizer",
                return_value=TokenizerBundle(FakeTokenizer(), "fake"),
            ):
                result = run_pipeline(
                    config,
                    resume=False,
                    skip_upload=True,
                    streams={"unit_instruct": rows},
                )

            shards = sorted((root / "sft").glob("train-*.jsonl"))
            self.assertTrue(shards)
            self.assertFalse(any((root / "sft").glob("train-*.bin")))
            self.assertTrue((root / "sft" / "manifest.json").exists())
            self.assertTrue((root / "sft" / "metadata.json").exists())
            row = json.loads(shards[0].read_text().splitlines()[0])
            self.assertIn("messages", row)
            self.assertEqual(result["metadata"]["output_mode"], "sft_jsonl")

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "requires fork multiprocessing")
    def test_pipeline_writes_sft_jsonl_shards_with_workers(self):
        rows = [
            {
                "instruction": f"Write a Python function number {index} that returns the index.",
                "response": f"def value():\n    return {index}\n",
            }
            for index in range(8)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = PipelineConfig(
                project_name="unit-sft-parallel",
                output_mode="sft_jsonl",
                artifact_format="jsonl_messages",
                target_tokens=96,
                sequence_length=8,
                val_ratio=0.0,
                seed=7,
                tokenizer_path=str(root / "missing-tokenizer"),
                fallback_tokenizer="fake",
                output_dir=str(root / "sft"),
                work_dir=str(root / "work"),
                progress_path=str(root / "work" / "progress.json"),
                report_path=str(root / "reports" / "dataset_report.json"),
                s3_output=None,
                checkpoint_interval_tokens=32,
                sft_samples_per_shard=2,
                num_workers=2,
                worker_batch_size=2,
                worker_start_method="fork",
                sources=(
                    SourceConfig(
                        name="unit_instruct",
                        type="instruct",
                        mix_role="instruct_only",
                        hf_id="local",
                        target_tokens=96,
                    ),
                ),
            )

            with patch(
                "aina_preproc.pipeline.load_tokenizer",
                return_value=TokenizerBundle(FakeTokenizer(), "fake"),
            ):
                result = run_pipeline(
                    config,
                    resume=False,
                    skip_upload=True,
                    streams={"unit_instruct": rows},
                )

            shards = sorted((root / "sft").glob("train-*.jsonl"))
            self.assertGreaterEqual(len(shards), 1)
            total_rows = sum(len(path.read_text().splitlines()) for path in shards)
            self.assertGreater(total_rows, 0)
            self.assertGreater(result["metadata"]["actual_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
