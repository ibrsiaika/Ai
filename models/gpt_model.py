"""
GPT model architecture for India's AI model.
Implements a transformer-based GPT model with support for various configurations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 50257
    context_length: int = 2048
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = 'gelu'
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    num_labels: int = 1
    problem_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_layers > 0, "num_layers must be > 0"
        assert self.vocab_size > 0, "vocab_size must be > 0"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.num_heads
        
        assert config.d_model % config.num_heads == 0, \
            f"d_model must be divisible by num_heads"
        
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len)
            past_key_value: Cached key and value for generation
            use_cache: Whether to return cached values
        
        Returns:
            output: (batch_size, seq_len, d_model)
            cache: Tuple of (key, value) if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        Q = self.query(hidden_states)  # (batch_size, seq_len, d_model)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use past key-value for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            K = torch.cat([past_key, K], dim=-2)
            V = torch.cat([past_value, V], dim=-2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch_size, num_heads, seq_len, key_len)
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            
            scores = scores + (1.0 - attention_mask) * -10000.0
        
        # Causal mask (for autoregressive)
        if past_key_value is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * -10000.0,
                diagonal=1
            )
            scores = scores + causal_mask[None, None, :, :]
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.output(context)
        
        # Prepare cache
        present = (K, V) if use_cache else None
        
        return output, present


class FeedForward(nn.Module):
    """Feed-forward layer."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation == 'gelu':
            self.activation = F.gelu
        elif config.activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, config: GPTConfig, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        
        self.attention = MultiHeadAttention(config)
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.ffn = FeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: Attention mask
            past_key_value: Cached attention values
            use_cache: Whether to cache
        
        Returns:
            output: (batch_size, seq_len, d_model)
            present: Cache tuple or None
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_output, present = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + self.dropout(ffn_output)
        
        return hidden_states, present


class GPTModel(nn.Module):
    """GPT model implementation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_id=i) 
            for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[list] = None,
                use_cache: bool = False,
                return_dict: bool = True) -> Union[Tuple, dict]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            past_key_values: Cache from previous generation
            use_cache: Whether to cache for generation
            return_dict: Whether to return dict or tuple
        
        Returns:
            output: Model output (logits or full output dict)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = hidden_states + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Expand attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.float()
        
        # Process through transformer layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'past_key_values': present_key_values if use_cache else None,
                'hidden_states': hidden_states
            }
        else:
            return logits, present_key_values


class GPTForCausalLM(nn.Module):
    """GPT model with language modeling head for training."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.model = GPTModel(config)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[list] = None,
                use_cache: bool = False) -> dict:
        """
        Forward pass for training.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) for language modeling
            past_key_values: Cache for generation
            use_cache: Whether to cache
        
        Returns:
            Dictionary with loss and other outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True
        )
        
        logits = outputs['logits']
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tensors
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits, shift_labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs['past_key_values'],
            'hidden_states': outputs['hidden_states']
        }


def create_gpt_model(model_size: str = 'small',
                    vocab_size: int = 50257,
                    context_length: int = 2048) -> GPTForCausalLM:
    """
    Create a GPT model with predefined configurations.
    
    Args:
        model_size: 'small', 'base', 'large', 'xlarge'
        vocab_size: Vocabulary size
        context_length: Context length
    
    Returns:
        GPTForCausalLM model
    """
    size_configs = {
        'small': {
            'd_model': 384,
            'num_layers': 6,
            'num_heads': 6,
            'd_ff': 1536,
        },
        'base': {
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'd_ff': 3072,
        },
        'large': {
            'd_model': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'd_ff': 4096,
        },
        'xlarge': {
            'd_model': 1536,
            'num_layers': 32,
            'num_heads': 24,
            'd_ff': 6144,
        }
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(size_configs.keys())}")
    
    config_dict = size_configs[model_size]
    config = GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        **config_dict
    )
    
    model = GPTForCausalLM(config)
    
    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created {model_size} GPT model with {num_params:,} parameters")
    
    return model
