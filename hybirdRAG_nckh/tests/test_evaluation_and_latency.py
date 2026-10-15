import unittest
import time
from retrieval.metrics.latency_tracker import LatencyTracker
from evaluation.dataset import BENCHMARK_DATASET
from evaluation.ragas_evaluator import RagasEvaluator


class TestLatencyAndEvaluation(unittest.TestCase):

    def test_latency_tracker_basic(self):
        tracker = LatencyTracker()

        with tracker.track("query_rewrite"):
            time.sleep(0.01)

        with tracker.track("vector_search"):
            time.sleep(0.01)

        tracker.stop_total()
        durations = tracker.to_dict(unit="seconds")

        self.assertIn("query_rewrite", durations)
        self.assertIn("vector_search", durations)
        self.assertIn("total", durations)
        self.assertGreaterEqual(durations["query_rewrite"], 0.009)
        self.assertGreaterEqual(durations["total"], 0.02)

        summary_text = tracker.summary()
        self.assertIn("GIAI ĐOẠN", summary_text)
        self.assertIn("Query Rewrite", summary_text)
        self.assertIn("Vector Search", summary_text)

    def test_latency_tracker_ms_conversion(self):
        tracker = LatencyTracker()
        tracker.record("graph_search", 0.05)
        tracker.record("reranking", 0.03)

        dur_ms = tracker.to_dict(unit="ms")
        self.assertAlmostEqual(dur_ms["graph_search"], 50.0, places=1)
        self.assertAlmostEqual(dur_ms["reranking"], 30.0, places=1)

    def test_benchmark_dataset(self):
        self.assertGreater(len(BENCHMARK_DATASET), 0)
        first_item = BENCHMARK_DATASET[0]
        self.assertIn("user_input", first_item)
        self.assertIn("reference", first_item)
        self.assertIn("reference_contexts", first_item)
        self.assertIsInstance(first_item["reference_contexts"], list)

    def test_evaluator_fallback_metrics(self):
        evaluator = RagasEvaluator()
        sample_results = [
            {
                "question": "Cây mai bị cháy lá do bệnh gì?",
                "answer": "Bệnh cháy lá trên mai vàng do nấm Pestalotia funerea gây ra, dùng thuốc Coc 85 để xịt.",
                "contexts": [
                    "Bệnh cháy lá do nấm Pestalotia funerea gây ra. Có thể dùng thuốc Coc 85 hoặc Anvil."
                ],
                "ground_truth": "Bệnh do nấm Pestalotia funerea gây ra, điều trị bằng Coc 85."
            }
        ]

        metrics = evaluator.evaluate_fallback(sample_results)
        self.assertIn("faithfulness", metrics)
        self.assertIn("answer_relevancy", metrics)
        self.assertIn("context_precision", metrics)
        self.assertIn("context_recall", metrics)

        self.assertGreater(metrics["faithfulness"], 0.5)
        self.assertGreater(metrics["answer_relevancy"], 0.3)


if __name__ == "__main__":
    unittest.main()
