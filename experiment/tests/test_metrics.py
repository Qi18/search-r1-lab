import unittest

from search_r1_lab.metrics import compute_metrics, contains_answer, exact_match, token_f1


class MetricsTest(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(exact_match("The Pelion Island.", "Pelion Island"), 1.0)
        self.assertEqual(token_f1("Mira Voss", "Mira Voss"), 1.0)
        self.assertEqual(contains_answer("Engineer Rhea Calder", "Rhea Calder"), 1.0)

    def test_summary(self):
        rows = [
            {
                "mode": "search",
                "prediction": "Mira Voss",
                "answer": "Mira Voss",
                "trajectory": "<answer>Mira Voss</answer>",
                "search_count": 1,
                "searches": [{"results": [{"id": "orilon"}]}],
                "evidence_id": "orilon",
                "latency_seconds": 1.0,
            }
        ]
        summary = compute_metrics(rows)["search"]
        self.assertEqual(summary["exact_match"], 1.0)
        self.assertEqual(summary["answer_contains"], 1.0)
        self.assertEqual(summary["retrieval_hit_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
