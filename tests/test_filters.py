import unittest

from aina_preproc.filters import should_keep


class FilterTests(unittest.TestCase):
    def test_rejects_secret_pattern(self):
        record = {
            "type": "base",
            "text": "def handler():\n    password='super-secret-value'\n    return password\n" * 3,
            "source": "unit",
            "language": "python",
        }

        result = should_keep(record)

        self.assertFalse(result.keep)
        self.assertEqual(result.reason, "secret")

    def test_accepts_normal_code(self):
        record = {
            "type": "base",
            "text": (
                "def add_numbers(left, right):\n"
                "    total = left + right\n"
                "    return total\n\n"
                "print(add_numbers(20, 22))\n"
            ),
            "source": "unit",
            "language": "python",
        }

        result = should_keep(record)

        self.assertTrue(result.keep)


if __name__ == "__main__":
    unittest.main()

