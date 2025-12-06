"""
Unit tests for text generation and inference.
Tests generation strategies and configurations.
"""

import unittest
import torch
from transformers import AutoTokenizer
from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig


class TestGenerationConfig(unittest.TestCase):
    """Test generation configuration."""
    
    def test_default_config(self):
        """Test default generation config."""
        config = GenerationConfig()
        self.assertEqual(config.max_length, 100)
        self.assertEqual(config.temperature, 1.0)
        self.assertFalse(config.do_sample)
        
    def test_custom_config(self):
        """Test custom generation config."""
        config = GenerationConfig(
            max_length=200,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        self.assertEqual(config.max_length, 200)
        self.assertEqual(config.temperature, 0.7)
        self.assertTrue(config.do_sample)


class TestTextGenerator(unittest.TestCase):
    """Test text generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = create_gpt_model('small')
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.generator = TextGenerator(self.model, self.tokenizer)
        
    def test_greedy_generation(self):
        """Test greedy decoding."""
        config = GenerationConfig(
            max_length=20,
            do_sample=False
        )
        
        prompt = "Hello"
        output = self.generator.generate(prompt, config)
        
        self.assertIsInstance(output, str)
        self.assertTrue(output.startswith(prompt) or len(output) > 0)
        
    def test_sampling_generation(self):
        """Test sampling-based generation."""
        config = GenerationConfig(
            max_length=20,
            do_sample=True,
            temperature=0.7
        )
        
        prompt = "The future of AI"
        output = self.generator.generate(prompt, config)
        
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), len(prompt))
        
    def test_top_k_generation(self):
        """Test top-k sampling."""
        config = GenerationConfig(
            max_length=20,
            do_sample=True,
            top_k=50
        )
        
        prompt = "Once upon a time"
        output = self.generator.generate(prompt, config)
        
        self.assertIsInstance(output, str)
        
    def test_top_p_generation(self):
        """Test nucleus (top-p) sampling."""
        config = GenerationConfig(
            max_length=20,
            do_sample=True,
            top_p=0.95
        )
        
        prompt = "In the beginning"
        output = self.generator.generate(prompt, config)
        
        self.assertIsInstance(output, str)
        
    def test_batch_generation(self):
        """Test batch text generation."""
        config = GenerationConfig(max_length=20)
        prompts = ["Hello", "World", "Test"]
        
        outputs = self.generator.generate_batch(prompts, config)
        
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertIsInstance(output, str)


if __name__ == '__main__':
    unittest.main()
