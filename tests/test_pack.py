import tempfile
import unittest
from pathlib import Path

from aina_preproc.pack import PackedDatasetWriter, validate_bin


class PackTests(unittest.TestCase):
    def test_writes_fixed_length_bins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with PackedDatasetWriter(
                tmpdir,
                sequence_length=4,
                dtype="uint16",
                val_ratio=0.0,
                seed=42,
                shard_sequences=1,
            ) as writer:
                writer.add_tokens([1, 2, 3, 4, 5, 6, 7, 8, 9])
                writer.flush_remainder()

            train_tokens_0, train_ok_0 = validate_bin(Path(tmpdir) / "train-00000.bin", "uint16", 4)
            train_tokens_1, train_ok_1 = validate_bin(Path(tmpdir) / "train-00001.bin", "uint16", 4)

            self.assertEqual(train_tokens_0 + train_tokens_1, 8)
            self.assertTrue(train_ok_0)
            self.assertTrue(train_ok_1)
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
