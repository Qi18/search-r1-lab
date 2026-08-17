import unittest

from search_r1_lab.protocol import extract_answer, extract_search_query, trim_to_first_action


class ProtocolTest(unittest.TestCase):
    def test_extracts_last_search(self):
        text = "<search>first</search><search>second query</search>"
        self.assertEqual(extract_search_query(text), "second query")

    def test_extracts_answer(self):
        self.assertEqual(extract_answer("<answer> Mira Voss </answer>"), "Mira Voss")

    def test_trims_after_first_action(self):
        text = "<think>x</think><search>Orilon</search>ignored"
        self.assertEqual(trim_to_first_action(text), "<think>x</think><search>Orilon</search>")


if __name__ == "__main__":
    unittest.main()
