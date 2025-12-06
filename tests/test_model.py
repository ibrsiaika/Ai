"""
Unit tests for GPT model architecture.
Tests model components, forward pass, and generation capabilities.
"""

import unittest
import torch
from models.gpt_model import (
    GPTConfig, MultiHeadAttention, FeedForward, 
    TransformerBlock, GPTModel, GPTForCausalLM, create_gpt_model
)


class TestGPTConfig(unittest.TestCase):
    """Test GPT configuration."""
    
    def test_small_config(self):
        """Test small model configuration."""
        config = GPTConfig(d_model=384, num_layers=6, num_heads=6, d_ff=1536)
        self.assertEqual(config.d_model, 384)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_heads, 6)
        
    def test_base_config(self):
        """Test base model configuration."""
        config = GPTConfig(d_model=768, num_layers=12, num_heads=12, d_ff=3072)
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.num_heads, 12)
        
    def test_large_config(self):
        """Test large model configuration."""
        config = GPTConfig(d_model=1024, num_layers=24, num_heads=16, d_ff=4096)
        self.assertEqual(config.d_model, 1024)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.num_heads, 16)


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention mechanism."""
    
    def setUp(self):
        self.config = GPTConfig(model_size='small')
        self.attention = MultiHeadAttention(self.config)
        
    def test_forward_pass(self):
        """Test attention forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        
        output, cache = self.attention(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))
        self.assertFalse(torch.isnan(output).any())
        
    def test_with_mask(self):
        """Test attention with causal mask."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        
        # Create attention mask
        mask = torch.ones(batch_size, seq_len)
        
        output, cache = self.attention(x, attention_mask=mask)
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))


class TestFeedForward(unittest.TestCase):
    """Test feed-forward network."""
    
    def setUp(self):
        self.config = GPTConfig(model_size='small')
        self.ff = FeedForward(self.config)
        
    def test_forward_pass(self):
        """Test feed-forward forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        
        output = self.ff(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))
        self.assertFalse(torch.isnan(output).any())


class TestTransformerBlock(unittest.TestCase):
    """Test complete transformer block."""
    
    def setUp(self):
        self.config = GPTConfig(model_size='small')
        self.block = TransformerBlock(self.config)
        
    def test_forward_pass(self):
        """Test transformer block forward pass."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        
        output, cache = self.block(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))
        self.assertFalse(torch.isnan(output).any())


class TestGPTModel(unittest.TestCase):
    """Test GPT model."""
    
    def setUp(self):
        self.config = GPTConfig(model_size='small')
        self.model = GPTModel(self.config)
        
    def test_forward_pass(self):
        """Test model forward pass."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = self.model(input_ids)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))
        self.assertFalse(torch.isnan(output).any())
        
    def test_position_embeddings(self):
        """Test position embeddings are applied."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Different positions should produce different outputs
        output1 = self.model(input_ids)
        
        # Shift input
        shifted_ids = torch.cat([
            torch.randint(0, self.config.vocab_size, (batch_size, 1)),
            input_ids[:, :-1]
        ], dim=1)
        output2 = self.model(shifted_ids)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output1, output2))


class TestGPTForCausalLM(unittest.TestCase):
    """Test GPT for causal language modeling."""
    
    def setUp(self):
        self.config = GPTConfig(model_size='small')
        self.model = GPTForCausalLM(self.config)
        
    def test_forward_pass(self):
        """Test causal LM forward pass."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = self.model(input_ids)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertFalse(torch.isnan(output).any())
        
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        loss = self.model.compute_loss(input_ids, labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        
    def test_generate(self):
        """Test text generation."""
        input_ids = torch.tensor([[1, 2, 3]])  # Small input
        
        output = self.model.generate(input_ids, max_new_tokens=5)
        
        self.assertEqual(output.shape[0], 1)  # Same batch size
        self.assertEqual(output.shape[1], 3 + 5)  # Original + new tokens
        

class TestModelCreation(unittest.TestCase):
    """Test model creation utilities."""
    
    def test_create_small_model(self):
        """Test creating small model."""
        model = create_gpt_model('small')
        self.assertIsInstance(model, GPTForCausalLM)
        
    def test_create_base_model(self):
        """Test creating base model."""
        model = create_gpt_model('base')
        self.assertIsInstance(model, GPTForCausalLM)
        
    def test_parameter_count(self):
        """Test parameter counting."""
        model = create_gpt_model('small')
        total_params = sum(p.numel() for p in model.parameters())
        # Small model should have roughly 100M parameters
        self.assertGreater(total_params, 50_000_000)
        self.assertLess(total_params, 150_000_000)


if __name__ == '__main__':
    unittest.main()
