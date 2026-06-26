import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aina_preproc.upload_s3 import upload_directory


class UploadS3Tests(unittest.TestCase):
    def test_upload_directory_skips_unchanged_selected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train-00000.bin").write_bytes(b"abc")
            (root / "metadata.partial.json").write_text("{}")

            client = FakeS3Client(existing_sizes={"prefix/train-00000.bin": 3})
            boto3_module = types.SimpleNamespace(client=lambda service: client)
            botocore_module = types.ModuleType("botocore")
            exceptions_module = types.ModuleType("botocore.exceptions")
            exceptions_module.ClientError = FakeClientError

            with patch.dict(
                sys.modules,
                {
                    "boto3": boto3_module,
                    "botocore": botocore_module,
                    "botocore.exceptions": exceptions_module,
                },
            ):
                uploaded = upload_directory(
                    root,
                    "s3://bucket/prefix/",
                    include={"train-00000.bin", "metadata.partial.json"},
                    skip_unchanged={"train-00000.bin"},
                )

        self.assertEqual(uploaded, ["s3://bucket/prefix/metadata.partial.json"])
        self.assertEqual(client.uploaded, [("metadata.partial.json", "bucket", "prefix/metadata.partial.json")])


class FakeS3Client:
    def __init__(self, existing_sizes):
        self.existing_sizes = existing_sizes
        self.uploaded = []

    def head_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.existing_sizes:
            raise FakeClientError("404")
        return {"ContentLength": self.existing_sizes[Key]}

    def upload_file(self, local_file, bucket, key):
        self.uploaded.append((Path(local_file).name, bucket, key))


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


if __name__ == "__main__":
    unittest.main()
