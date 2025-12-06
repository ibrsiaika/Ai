# Configuration Guide for India's GPT Model

## Overview

This guide explains how to configure India's GPT model for complex tasks, realistic conversations, code generation, and competitive performance.

## Table of Contents

1. [Configuration Files](#configuration-files)
2. [Model Configuration](#model-configuration)
3. [Training Configuration](#training-configuration)
4. [Generation Presets](#generation-presets)
5. [Task-Specific Settings](#task-specific-settings)
6. [Optimization Settings](#optimization-settings)

---

## Configuration Files

### Default Configuration
**File**: `config/default_config.json`

Basic configuration for standard use:
- Base model size (125M parameters)
- Standard training settings
- Simple generation config
- Suitable for learning and experimentation

### Advanced Configuration
**File**: `config/advanced_config.json`

Advanced configuration for production use:
- Large model size (355M parameters)
- Optimized training settings
- Multiple generation presets
- Task-specific configurations
- Competitive features enabled

---

## Model Configuration

### Model Sizes

#### Small (100M parameters)
```json
{
  "d_model": 384,
  "num_layers": 6,
  "num_heads": 6,
  "d_ff": 1536
}
```
**Use for**: Quick experiments, limited resources

#### Base (125M parameters)
```json
{
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072
}
```
**Use for**: General purpose, balanced performance

#### Large (355M parameters)
```json
{
  "d_model": 1024,
  "num_layers": 24,
  "num_heads": 16,
  "d_ff": 4096
}
```
**Use for**: Complex tasks, high quality generation

#### XLarge (1B parameters)
```json
{
  "d_model": 1536,
  "num_layers": 32,
  "num_heads": 24,
  "d_ff": 6144
}
```
**Use for**: State-of-the-art performance, research

### Context Length

```json
{
  "context_length": 2048  // Standard
  "context_length": 4096  // Extended for long documents
  "context_length": 8192  // Maximum for complex tasks
}
```

**Recommendations**:
- **Code**: 2048-4096 tokens
- **Conversation**: 1024-2048 tokens
- **Documents**: 4096-8192 tokens

---

## Training Configuration

### For Complex Tasks

```json
{
  "training": {
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "warmup_steps": 2000,
    "gradient_accumulation_steps": 4,
    "fp16": true
  }
}
```

**Key Settings**:
- **Higher epochs** (10+): Better learning of complex patterns
- **Smaller learning rate** (1e-4): More stable training
- **Larger warmup** (2000+): Gradual learning rate increase
- **Gradient accumulation**: Simulate larger batch sizes

### For Realistic Conversations

```json
{
  "training": {
    "num_epochs": 5,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "max_seq_length": 2048,
    "use_language_mixing": true
  }
}
```

**Key Settings**:
- **Moderate epochs** (5-7): Avoid overfitting
- **Larger batch size**: Diverse conversation patterns
- **Language mixing**: Multilingual conversations

### For Code Generation

```json
{
  "training": {
    "num_epochs": 15,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 4096,
    "gradient_checkpointing": true
  }
}
```

**Key Settings**:
- **More epochs** (15+): Learn code syntax thoroughly
- **Longer sequences**: Complete code blocks
- **Gradient checkpointing**: Handle memory constraints

---

## Generation Presets

### Code Generation Preset

```json
{
  "code_generation": {
    "max_length": 512,
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": true
  }
}
```

**Characteristics**:
- **Low temperature** (0.2): More deterministic, accurate code
- **Moderate top-k** (40): Focused vocabulary
- **Repetition penalty**: Avoid code duplication

**Best for**:
- Function completion
- Class implementation
- Algorithm generation
- Bug fixing

### Conversation Preset

```json
{
  "conversation": {
    "max_length": 256,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "do_sample": true
  }
}
```

**Characteristics**:
- **Moderate temperature** (0.7): Natural variation
- **Higher repetition penalty**: Diverse responses
- **Balanced sampling**: Natural yet coherent

**Best for**:
- Q&A systems
- Chatbots
- Virtual assistants
- Customer support

### Creative Writing Preset

```json
{
  "creative_writing": {
    "max_length": 1024,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "do_sample": true
  }
}
```

**Characteristics**:
- **High temperature** (0.8): Creative variation
- **High top-p** (0.95): Diverse word choices
- **Strong repetition penalty**: Avoid repetition

**Best for**:
- Story writing
- Content creation
- Brainstorming
- Marketing copy

### Accurate Completion Preset

```json
{
  "accurate_completion": {
    "max_length": 256,
    "temperature": 0.1,
    "top_k": 10,
    "num_beams": 4,
    "do_sample": false,
    "early_stopping": true
  }
}
```

**Characteristics**:
- **Very low temperature** (0.1): Deterministic
- **Beam search**: Multiple hypotheses
- **No sampling**: Most likely output

**Best for**:
- Factual completion
- Technical documentation
- Formal writing
- Consistent output

---

## Task-Specific Settings

### Code Generation

```json
{
  "tasks": {
    "code_generation": {
      "enabled": true,
      "languages": ["python", "javascript", "java", "c++", "go"],
      "max_context": 2048,
      "instruction_format": "### Instruction:\n{instruction}\n\n### Code:\n"
    }
  }
}
```

**Usage**:
```python
prompt = "### Instruction:\nWrite a function to calculate factorial\n\n### Code:\n"
```

### Conversation

```json
{
  "tasks": {
    "conversation": {
      "enabled": true,
      "system_prompt": "You are a helpful, respectful assistant.",
      "max_turns": 10,
      "format": "### Human:\n{user}\n\n### Assistant:\n"
    }
  }
}
```

**Usage**:
```python
prompt = "### Human:\nWhat is machine learning?\n\n### Assistant:\n"
```

### Question Answering

```json
{
  "tasks": {
    "question_answering": {
      "enabled": true,
      "max_context": 2048,
      "format": "### Question:\n{question}\n\n### Answer:\n"
    }
  }
}
```

**Usage**:
```python
prompt = "### Question:\nHow does GPT work?\n\n### Answer:\n"
```

---

## Optimization Settings

### For Speed

```json
{
  "optimization": {
    "fp16": true,
    "gradient_accumulation_steps": 4,
    "num_workers": 8
  },
  "deployment": {
    "quantization": {
      "enabled": true,
      "method": "int8"
    }
  }
}
```

**Benefits**:
- **2-3x faster** training with FP16
- **Smaller memory** footprint
- **Faster inference** with quantization

### For Quality

```json
{
  "model": {
    "model_size": "large",
    "context_length": 4096
  },
  "training": {
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "warmup_steps": 5000
  }
}
```

**Benefits**:
- **Better understanding** with larger model
- **Longer context** for complex tasks
- **More stable** learning

### For Memory Efficiency

```json
{
  "optimization": {
    "gradient_checkpointing": true,
    "fp16": true
  },
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 8
  }
}
```

**Benefits**:
- **Reduced memory** usage
- **Effective batch size** of 64
- **Can use larger** models

---

## Competitive Features

### Instruction Following

```json
{
  "competitive_features": {
    "instruction_following": true,
    "few_shot_learning": true,
    "chain_of_thought": true
  }
}
```

**Enables**:
- Task-specific instructions
- Learning from examples
- Step-by-step reasoning

### Advanced Capabilities

```json
{
  "competitive_features": {
    "code_execution": false,
    "retrieval_augmentation": false,
    "multimodal": false
  }
}
```

**Future capabilities**:
- Code execution and validation
- External knowledge retrieval
- Image understanding

---

## Example Configurations

### Configuration for Complex Coding Tasks

```json
{
  "model": {
    "model_size": "large",
    "context_length": 4096
  },
  "training": {
    "num_epochs": 20,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "fp16": true,
    "gradient_checkpointing": true
  },
  "generation": {
    "presets": {
      "code": {
        "max_length": 1024,
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 0.9
      }
    }
  }
}
```

### Configuration for Realistic Conversations

```json
{
  "model": {
    "model_size": "base",
    "context_length": 2048
  },
  "training": {
    "num_epochs": 5,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "use_language_mixing": true
  },
  "generation": {
    "presets": {
      "chat": {
        "max_length": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15
      }
    }
  },
  "tasks": {
    "conversation": {
      "system_prompt": "You are a helpful AI assistant.",
      "max_turns": 10
    }
  }
}
```

### Configuration for Competitive Performance

```json
{
  "model": {
    "model_size": "xlarge",
    "context_length": 8192
  },
  "training": {
    "num_epochs": 15,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "warmup_steps": 5000,
    "gradient_accumulation_steps": 16,
    "fp16": true,
    "gradient_checkpointing": true
  },
  "competitive_features": {
    "instruction_following": true,
    "few_shot_learning": true,
    "chain_of_thought": true
  },
  "evaluation": {
    "metrics": ["perplexity", "bleu", "rouge", "accuracy"],
    "code_eval": {"enabled": true},
    "conversation_eval": {"enabled": true}
  }
}
```

---

## Usage Examples

### Loading Configuration

```python
import json

# Load configuration
with open('config/advanced_config.json', 'r') as f:
    config = json.load(f)

# Use specific preset
generation_config = config['generation']['presets']['code_generation']
```

### Custom Configuration

```python
from inference.generator import GenerationConfig

# Create custom config
custom_config = GenerationConfig(
    max_length=512,
    temperature=0.5,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)
```

### Switching Presets

```python
# For code
config = GenerationConfig(**presets['code_generation'])

# For conversation
config = GenerationConfig(**presets['conversation'])

# For creative writing
config = GenerationConfig(**presets['creative_writing'])
```

---

## Best Practices

1. **Start with defaults**, then adjust
2. **Use presets** for common tasks
3. **Test configurations** before training
4. **Monitor metrics** during training
5. **Compare results** across configurations
6. **Document changes** for reproducibility

---

## Troubleshooting

### Generation Too Random
- Decrease `temperature` (0.5 → 0.2)
- Decrease `top_p` (0.95 → 0.8)
- Increase `repetition_penalty`

### Generation Too Repetitive
- Increase `temperature` (0.5 → 0.8)
- Increase `top_p` (0.8 → 0.95)
- Enable `do_sample`

### Out of Memory
- Enable `gradient_checkpointing`
- Enable `fp16`
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`

### Poor Quality
- Increase model size
- Increase training epochs
- Use larger context length
- Adjust learning rate

---

**Last Updated**: December 2024
**Version**: 1.0.0
