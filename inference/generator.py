"""
Inference and text generation for India's GPT model.
Supports various decoding strategies and optimizations.
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 128
    min_length: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    do_sample: bool = False
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 50256
    bos_token_id: int = 50256


class TextGenerator:
    """Text generation using various decoding strategies."""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 device: str = 'cuda'):
        """
        Initialize text generator.
        
        Args:
            model: GPT model
            tokenizer: Tokenizer for encoding/decoding
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate(self,
                 prompt: str,
                 config: GenerationConfig = None,
                 return_prompt: bool = False) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            return_prompt: Whether to include prompt in output
        
        Returns:
            Generated text
        """
        if config is None:
            config = GenerationConfig()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        if config.do_sample:
            output_ids = self._sample_generate(input_ids, config)
        elif config.num_beams > 1:
            output_ids = self._beam_search(input_ids, config)
        else:
            output_ids = self._greedy_generate(input_ids, config)
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if not return_prompt:
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def _greedy_generate(self,
                        input_ids: torch.Tensor,
                        config: GenerationConfig) -> torch.Tensor:
        """
        Greedy decoding.
        
        Args:
            input_ids: Input token IDs
            config: Generation config
        
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        max_length = config.max_length
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs['logits']
                
                # Get last token logits
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(input_ids[i].tolist()):
                            next_token_logits[i, token_id] /= config.repetition_penalty
                
                # Greedy selection
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if (next_tokens == config.eos_token_id).all():
                    break
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        return input_ids
    
    def _sample_generate(self,
                        input_ids: torch.Tensor,
                        config: GenerationConfig) -> torch.Tensor:
        """
        Sampling-based generation with top-k and top-p.
        
        Args:
            input_ids: Input token IDs
            config: Generation config
        
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        max_length = config.max_length
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs['logits']
                
                # Get last token logits
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(input_ids[i].tolist()):
                            next_token_logits[i, token_id] /= config.repetition_penalty
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Top-k filtering
                if config.top_k is not None and config.top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(
                        probs, config.top_k, dim=-1
                    )
                    probs = torch.zeros_like(probs)
                    probs.scatter_(-1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Top-p (nucleus) filtering
                if config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumsum_probs > config.top_p
                    sorted_indices_to_remove[..., 0] = 0  # Keep at least one token
                    
                    # Create mask
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    indices_to_remove.scatter_(
                        -1, sorted_indices,
                        sorted_indices_to_remove
                    )
                    
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from distribution
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                if (next_tokens == config.eos_token_id).all():
                    break
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        return input_ids
    
    def _beam_search(self,
                    input_ids: torch.Tensor,
                    config: GenerationConfig) -> torch.Tensor:
        """
        Beam search generation.
        
        Args:
            input_ids: Input token IDs
            config: Generation config
        
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        max_length = config.max_length
        
        # Initialize beam state
        beam_scores = torch.zeros(
            (batch_size, num_beams),
            dtype=torch.float,
            device=self.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        # Expand input for beams
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        
        # Keep track of generated sequences
        generated = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs['logits']
                
                # Get last token logits
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature
                
                # Softmax
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                
                # Log probabilities
                next_token_log_probs = torch.log(next_token_logits + 1e-10)
                
                # Reshape for beam search
                vocab_size = next_token_log_probs.shape[-1]
                next_token_log_probs = next_token_log_probs.view(
                    batch_size, num_beams, vocab_size
                )
                
                # Add beam scores
                next_token_log_probs = next_token_log_probs + beam_scores.view(
                    batch_size, num_beams, 1
                )
                
                # Flatten and get top-k
                next_token_log_probs = next_token_log_probs.view(batch_size, -1)
                top_log_probs, top_indices = torch.topk(
                    next_token_log_probs, num_beams, dim=-1
                )
                
                # Get beam and token indices
                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size
                
                # Update beam scores
                beam_scores = top_log_probs.view(-1)
                
                # Update input_ids and generated
                input_ids = input_ids[beam_indices]
                generated = generated[beam_indices]
                
                # Append new tokens
                input_ids = torch.cat(
                    [input_ids, token_indices.view(-1, 1)],
                    dim=-1
                )
                generated = torch.cat(
                    [generated, token_indices.view(-1, 1)],
                    dim=-1
                )
        
        # Return best sequence
        return generated[:batch_size]
    
    def batch_generate(self,
                      prompts: List[str],
                      config: GenerationConfig = None) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
        
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config)
            results.append(result)
        return results


class TextClassifier:
    """Text classification using GPT embeddings."""
    
    def __init__(self, model, tokenizer, device: str = 'cuda'):
        """
        Initialize classifier.
        
        Args:
            model: GPT model
            tokenizer: Tokenizer
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get text embeddings.
        
        Args:
            texts: List of texts
        
        Returns:
            Embedding matrix of shape (num_texts, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                input_ids = self.tokenizer.encode(
                    text, return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(input_ids=input_ids)
                hidden_states = outputs['hidden_states']
                
                # Use mean pooling
                embedding = hidden_states.mean(dim=1).cpu()
                embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score
        """
        embeddings = self.get_embeddings([text1, text2])
        
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        # Cosine similarity
        similarity = (embeddings[0] * embeddings[1]).sum().item()
        
        return similarity


def create_generator(model, tokenizer, device: str = 'cuda') -> TextGenerator:
    """
    Create text generator.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer
        device: Device to use
    
    Returns:
        TextGenerator instance
    """
    return TextGenerator(model, tokenizer, device)
