"""
Unit tests for evaluation metrics.
Tests perplexity, BLEU, ROUGE, and other metrics.
"""

import unittest
from transformers import AutoTokenizer
from utils.metrics import MetricsCalculator, LanguageModelingMetrics


class TestLanguageModelingMetrics(unittest.TestCase):
    """Test language modeling metrics."""
    
    def test_perplexity_calculation(self):
        """Test perplexity calculation."""
        loss = 2.0
        metrics = LanguageModelingMetrics(loss=loss)
        
        perplexity = metrics.perplexity
        expected_perplexity = 2.718281828459045 ** 2.0  # e^2
        
        self.assertAlmostEqual(perplexity, expected_perplexity, places=5)
        
    def test_zero_loss(self):
        """Test perplexity with zero loss."""
        metrics = LanguageModelingMetrics(loss=0.0)
        self.assertEqual(metrics.perplexity, 1.0)


class TestMetricsCalculator(unittest.TestCase):
    """Test metrics calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.calculator = MetricsCalculator(self.tokenizer)
        
    def test_bleu_score_identical(self):
        """Test BLEU score with identical texts."""
        predictions = ["The quick brown fox"]
        references = ["The quick brown fox"]
        
        metrics = self.calculator.calculate_metrics(predictions, references, loss=1.0)
        
        # Identical texts should have high BLEU
        self.assertGreater(metrics.bleu, 0.9)
        
    def test_bleu_score_different(self):
        """Test BLEU score with different texts."""
        predictions = ["The quick brown fox"]
        references = ["A slow red dog"]
        
        metrics = self.calculator.calculate_metrics(predictions, references, loss=1.0)
        
        # Different texts should have lower BLEU
        self.assertLess(metrics.bleu, 0.5)
        
    def test_rouge_scores(self):
        """Test ROUGE scores."""
        predictions = ["The quick brown fox jumps over the lazy dog"]
        references = ["The fast brown fox leaps over the sleepy dog"]
        
        metrics = self.calculator.calculate_metrics(predictions, references, loss=1.0)
        
        # Check ROUGE scores exist
        self.assertIsNotNone(metrics.rouge1)
        self.assertIsNotNone(metrics.rouge2)
        self.assertIsNotNone(metrics.rougeL)
        
        # Check ROUGE-1 structure
        self.assertIn('precision', metrics.rouge1)
        self.assertIn('recall', metrics.rouge1)
        self.assertIn('f1', metrics.rouge1)
        
    def test_multiple_predictions(self):
        """Test metrics with multiple predictions."""
        predictions = [
            "Hello world",
            "Goodbye world",
            "Nice to meet you"
        ]
        references = [
            "Hello there",
            "Goodbye friend",
            "Pleased to meet you"
        ]
        
        metrics = self.calculator.calculate_metrics(predictions, references, loss=1.5)
        
        self.assertIsNotNone(metrics.bleu)
        self.assertIsNotNone(metrics.perplexity)


if __name__ == '__main__':
    unittest.main()
