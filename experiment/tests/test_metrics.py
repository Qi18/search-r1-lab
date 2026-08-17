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
                "generated_search_count": 1,
                "retriever_request_count": 1,
                "search_events": [
                    {
                        "retriever_requested": True,
                        "results": [{"id": "orilon"}],
                    }
                ],
                "evidence_id": "orilon",
                "latency_seconds": 1.0,
            },
            {
                "mode": "no-search",
                "prediction": "unknown",
                "answer": "Mira Voss",
                "trajectory": "<search>Orilon</search><answer>unknown</answer>",
                "generated_search_count": 1,
                "retriever_request_count": 0,
                "search_events": [
                    {
                        "retriever_requested": False,
                        "results": [],
                    }
                ],
                "evidence_id": "orilon",
                "latency_seconds": 1.0,
            }
        ]
        search = compute_metrics(rows)["search"]
        no_search = compute_metrics(rows)["no-search"]
        self.assertEqual(search["exact_match"], 1.0)
        self.assertEqual(search["generated_search_tag_rate"], 1.0)
        self.assertEqual(search["retriever_request_rate"], 1.0)
        self.assertEqual(search["retrieval_hit_rate"], 1.0)
        self.assertEqual(no_search["generated_search_tag_rate"], 1.0)
        self.assertEqual(no_search["retriever_request_rate"], 0.0)
        self.assertEqual(no_search["retrieval_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
