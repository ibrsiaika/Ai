"""
Unit tests for data loading and preprocessing.
Tests data pipeline components and preprocessing.
"""

import unittest
import torch
from transformers import AutoTokenizer
from data.data_loader import DataProcessor, DataCollatorForLanguageModeling


class TestDataProcessor(unittest.TestCase):
    """Test data processor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer_name = 'gpt2'
        self.processor = DataProcessor(
            tokenizer_name=self.tokenizer_name,
            max_seq_length=512
        )
        
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.tokenizer)
        self.assertEqual(self.processor.max_seq_length, 512)
        
    def test_tokenize_function(self):
        """Test tokenization function."""
        examples = {
            'text': ['Hello world', 'This is a test']
        }
        
        result = self.processor._tokenize_function(examples)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertEqual(len(result['input_ids']), 2)
        
    def test_load_from_text_list(self):
        """Test loading from text list."""
        texts = ['Hello world', 'This is a test', 'Another example']
        
        dataset = self.processor.load_from_text_list(texts)
        
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)


class TestDataCollator(unittest.TestCase):
    """Test data collator for language modeling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )
        
    def test_collate_batch(self):
        """Test batch collation."""
        # Create sample batch
        batch = [
            {'input_ids': [1, 2, 3, 4, 5]},
            {'input_ids': [6, 7, 8]},
            {'input_ids': [9, 10, 11, 12]}
        ]
        
        result = self.collator(batch)
        
        self.assertIn('input_ids', result)
        self.assertIn('labels', result)
        self.assertIsInstance(result['input_ids'], torch.Tensor)
        
        # Check that all sequences have same length (padded)
        batch_size = len(batch)
        max_len = max(len(item['input_ids']) for item in batch)
        self.assertEqual(result['input_ids'].shape, (batch_size, max_len))


if __name__ == '__main__':
    unittest.main()
