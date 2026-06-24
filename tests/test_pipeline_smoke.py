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


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_writes_outputs_from_local_stream(self):
        rows = [
            {
                "content": (
                    "def add_numbers(left, right):\n"
                    "    total = left + right\n"
                    "    return total\n\n"
                    "print(add_numbers(1, 2))\n"
                ),
                "path": "math.py",
            },
            {
                "content": (
                    "function greet(name) {\n"
                    "  const message = `hello ${name}`;\n"
                    "  return message.toUpperCase();\n"
                    "}\n"
                ),
                "path": "greet.js",
                "language": "javascript",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = PipelineConfig(
                project_name="unit",
                target_tokens=64,
                sequence_length=8,
                val_ratio=0.0,
                seed=7,
                tokenizer_path=str(root / "missing-tokenizer"),
                fallback_tokenizer="fake",
                output_dir=str(root / "packed"),
                work_dir=str(root / "work"),
                progress_path=str(root / "work" / "progress.json"),
                report_path=str(root / "reports" / "dataset_report.json"),
                s3_output=None,
                checkpoint_interval_tokens=16,
                shard_sequences=2,
                sources=(
                    SourceConfig(
                        name="unit_python",
                        type="base",
                        hf_id="local",
                        target_tokens=64,
                        language="python",
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
                    streams={"unit_python": rows},
                )

            self.assertTrue(any((root / "packed").glob("train-*.bin")))
            self.assertTrue(any((root / "packed").glob("val-*.bin")))
            self.assertTrue((root / "packed" / "manifest.json").exists())
            self.assertTrue((root / "packed" / "metadata.json").exists())
            self.assertTrue((root / "reports" / "dataset_report.json").exists())
            self.assertGreaterEqual(result["metadata"]["total_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
