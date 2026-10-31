import io
import unittest

import pandas as pd

from core import MAX_BYTES, csv_bytes, predict_reviews, read_reviews


class StubModel:
    # Проверяет интеграционный контракт; не является обученной ML-моделью.
    def predict(self, texts):
        return ["negative" if "bad" in text else "positive" for text in texts]


class CoreTests(unittest.TestCase):
    def test_valid_order_and_labels(self):
        result = predict_reviews(StubModel(), b"text\ngood\nbad\ngood again\n")
        self.assertEqual(result.text.tolist(), ["good", "bad", "good again"])
        self.assertEqual(result.sentiment.tolist(), ["positive", "negative", "positive"])

    def test_content_alias(self):
        self.assertEqual(len(read_reviews(b"content\nhello\n")[0]), 1)

    def test_filter_and_export(self):
        result = predict_reviews(StubModel(), b"text\ngood\nbad\n")
        neg = result[result.sentiment == "negative"]
        exported = pd.read_csv(io.BytesIO(csv_bytes(neg)))
        self.assertEqual(exported.text.tolist(), ["bad"])

    def test_missing_column(self):
        with self.assertRaises(ValueError): read_reviews(b"other\nhello\n")

    def test_empty_file(self):
        with self.assertRaises(ValueError): read_reviews(b"")

    def test_empty_review(self):
        with self.assertRaises(ValueError): read_reviews(b'text\n"   "\n')

    def test_invalid_encoding(self):
        with self.assertRaises(ValueError): read_reviews(b"text\n\xff\n")

    def test_broken_csv(self):
        with self.assertRaises(ValueError): read_reviews(b'text\n"unfinished')

    def test_1000_rows(self):
        self.assertEqual(len(read_reviews(b"text\n" + b"good\n" * 1000)[0]), 1000)

    def test_1001_rows(self):
        with self.assertRaises(ValueError): read_reviews(b"text\n" + b"good\n" * 1001)

    def test_5000_characters(self):
        self.assertEqual(len(read_reviews(b"text\n" + b"a" * 5000)[0]), 1)

    def test_5001_characters(self):
        with self.assertRaises(ValueError): read_reviews(b"text\n" + b"a" * 5001)

    def test_5mb_boundary(self):
        # Валидная граница: длинный заголовок вспомогательного поля.
        raw = b"text," + b"x" * (MAX_BYTES - 12) + b"\ngood,\n"
        self.assertEqual(len(raw), MAX_BYTES)
        self.assertEqual(len(read_reviews(raw)[0]), 1)

    def test_over_5mb(self):
        with self.assertRaises(ValueError): read_reviews(b"x" * (MAX_BYTES + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
