# Examples for India's GPT Model

This directory contains example scripts demonstrating how to use India's GPT model for various tasks.

## Available Examples

### 01_basic_generation.py
Basic text generation with different configurations:
- Simple text completion
- Creative writing
- Factual completion

**Run:**
```bash
python examples/01_basic_generation.py
```

### 02_code_generation.py
Code generation examples:
- Python function completion
- Class implementation
- Algorithm generation
- JavaScript code

**Run:**
```bash
python examples/02_code_generation.py
```

### 03_conversation.py
Conversational AI examples:
- Question answering
- Multi-turn conversations
- Context-aware responses
- Detailed explanations

**Run:**
```bash
python examples/03_conversation.py
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run any example:
```bash
python examples/01_basic_generation.py
```

## Configuration Presets

Each example uses optimized configurations for specific tasks:

### Code Generation
- Temperature: 0.2 (low for accuracy)
- Top-k: 40
- Top-p: 0.9
- Repetition penalty: 1.1

### Conversation
- Temperature: 0.7 (moderate for natural variation)
- Top-k: 50
- Top-p: 0.9
- Repetition penalty: 1.15

### Creative Writing
- Temperature: 0.8 (high for creativity)
- Top-k: 50
- Top-p: 0.95
- Repetition penalty: 1.2

## Custom Usage

You can create your own examples by:

1. Importing the necessary modules:
```python
from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer
```

2. Creating a custom configuration:
```python
config = GenerationConfig(
    max_length=256,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)
```

3. Generating text:
```python
model = create_gpt_model('base')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
generator = TextGenerator(model, tokenizer)

result = generator.generate("Your prompt here", config)
```

## Tips

- Use **small temperature** (0.1-0.3) for factual/code generation
- Use **medium temperature** (0.5-0.7) for conversation
- Use **high temperature** (0.7-0.9) for creative writing
- Adjust **repetition_penalty** (1.1-1.3) to reduce repetition
- Use **top_k** and **top_p** together for best quality

## More Information

- See `TESTING_GUIDE.md` for validation and testing
- See `CONFIGURATION_GUIDE.md` for detailed configuration options
- See `README.md` for complete documentation
