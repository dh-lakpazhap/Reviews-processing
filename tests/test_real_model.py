import unittest
from pathlib import Path
import joblib
from core import predict_reviews, read_reviews

ROOT = Path(__file__).resolve().parents[1]

class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / "sentiment_model.joblib")

    def test_TC01_20_rows(self):
        result = predict_reviews(self.model, (ROOT / "test_data/valid.csv").read_bytes())
        self.assertEqual(len(result), 20)
        self.assertEqual(result.columns.tolist(), ["row_id", "text", "sentiment"])
        self.assertEqual(result.row_id.tolist(), list(range(1,21)))
        self.assertTrue(set(result.sentiment).issubset({"negative","neutral","positive"}))

    def test_TC02_filter(self):
        result = predict_reviews(self.model, (ROOT / "test_data/valid.csv").read_bytes())
        negative = result[result.sentiment == "negative"]
        self.assertTrue(all(negative.sentiment == "negative"))
        self.assertEqual(len(result), 20)

    def test_TC03_missing_text(self):
        with self.assertRaisesRegex(ValueError, "столбец text"):
            predict_reviews(self.model,(ROOT / "test_data/no_text.csv").read_bytes())

    def test_TC04_empty_review(self):
        with self.assertRaisesRegex(ValueError, "пустые отзывы"):
            predict_reviews(self.model,(ROOT / "test_data/empty_row.csv").read_bytes())

    def test_TC05_1000_rows(self):
        result = predict_reviews(self.model,(ROOT / "test_data/1000_rows.csv").read_bytes())
        self.assertEqual(len(result), 1000)

    def test_TC06_1001_rows(self):
        with self.assertRaisesRegex(ValueError, "1000"):
            predict_reviews(self.model,(ROOT / "test_data/1001_rows.csv").read_bytes())
