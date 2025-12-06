# India's First AI Model - GPT Implementation

A comprehensive implementation of India's first large language model built from scratch using PyTorch and HuggingFace Transformers.

## Overview

This project implements a full-featured GPT (Generative Pre-trained Transformer) model with support for:

- **Multi-language Support**: Hindi, English, Tamil, Telugu, Kannada, Malayalam
- **Distributed Training**: Multi-GPU/TPU support with Accelerate
- **Mixed Precision**: FP16 training for efficiency
- **Advanced Generation**: Beam search, temperature sampling, top-k/top-p filtering
- **Comprehensive Evaluation**: BLEU, ROUGE, perplexity metrics
- **Production Ready**: Checkpointing, monitoring, deployment utilities

## Project Structure

```
india_gpt/
├── data/
│   └── data_loader.py           # Data loading and preprocessing
├── models/
│   └── gpt_model.py             # GPT model architecture
├── training/
│   └── trainer.py               # Training pipeline
├── inference/
│   └── generator.py             # Text generation and inference
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   └── helpers.py               # Utility functions
├── config/
│   └── default_config.json      # Configuration template
├── notebooks/
│   └── training_demo.ipynb      # Jupyter notebook tutorial
├── train.py                     # Main training script
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 50GB+ free disk space (for model and data)

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd india_gpt
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
python train.py \
    --model_size base \
    --train_data wikitext \
    --num_epochs 3 \
    --batch_size 32 \
    --output_dir ./outputs
```

### Advanced Training with Custom Config

```bash
python train.py \
    --train_data /path/to/data \
    --model_size large \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --warmup_steps 2000 \
    --fp16 \
    --use_wandb \
    --generate_samples
```

## Model Sizes

- **small**: 384D, 6 layers, 6 heads → ~100M parameters
- **base**: 768D, 12 layers, 12 heads → ~125M parameters
- **large**: 1024D, 24 layers, 16 heads → ~355M parameters
- **xlarge**: 1536D, 32 layers, 24 heads → ~1B parameters

## Training Features

### Data Pipeline
- Automatic data loading from local files or HuggingFace Hub
- Support for JSON, JSONL, and text formats
- Language-aware preprocessing
- Token-level batching with dynamic padding

### Optimization
- AdamW optimizer with weight decay
- Cosine annealing with warm-up
- Gradient accumulation
- Gradient clipping
- Mixed precision training (FP16)

### Checkpointing
- Automatic checkpoint saving
- Best model tracking
- Resume from checkpoint
- Configuration persistence

### Monitoring
- Weights & Biases integration
- Training/validation loss tracking
- Perplexity monitoring
- Learning rate scheduling visualization

## Text Generation

### Python API

```python
from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer

# Load model and tokenizer
model = create_gpt_model('base')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Create generator
generator = TextGenerator(model, tokenizer)

# Generate text
config = GenerationConfig(
    max_length=128,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

prompt = "India is a country"
generated = generator.generate(prompt, config)
print(generated)
```

### Decoding Strategies

1. **Greedy**: Select highest probability token
2. **Sampling**: Random sampling from distribution
3. **Top-k**: Sample from top k most likely tokens
4. **Top-p (Nucleus)**: Sample from tokens with cumulative probability > p
5. **Beam Search**: Multi-hypothesis search with pruning

## Evaluation Metrics

### Supported Metrics

- **Perplexity**: PPL = exp(loss)
- **BLEU**: N-gram overlap with references
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence

### Example

```python
from utils.metrics import MetricsCalculator

calculator = MetricsCalculator(tokenizer)

predictions = ["The quick brown fox"]
references = ["A fast brown fox"]

metrics = calculator.calculate_metrics(predictions, references, loss=2.5)
print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"BLEU: {metrics.bleu:.4f}")
print(f"ROUGE-1 F1: {metrics.rouge1['f1']:.4f}")
```

## Advanced Usage

### Custom Data Loading

```python
from data.data_loader import DataProcessor

processor = DataProcessor(
    tokenizer_name='gpt2',
    max_seq_length=2048,
    languages=['en', 'hi', 'ta']
)

# Load custom data
dataset = processor.load_nworld_data('/path/to/data.jsonl')
prepared = processor.prepare_dataset('/path/to/data', split='train')
```

### Language Mixing

```python
from data.data_loader import LanguageMixingDataset

# Create language-specific datasets
en_dataset = load_dataset('wikitext', name='wikitext-2', split='train')
hi_dataset = load_dataset('hindi_corpus', split='train')

# Mix languages
mixed_dataset = LanguageMixingDataset(
    {
        'english': en_dataset,
        'hindi': hi_dataset
    },
    sampling_strategy='temperature'
)
```

### Custom Training Loop

```python
from training.trainer import TrainingConfig, Trainer

config = TrainingConfig(
    output_dir='./outputs',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    fp16=True,
    use_wandb=True
)

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
    config=config
)

results = trainer.train()
```

## Performance Optimization

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use lower precision
model.half()

# Reduce batch size
batch_size = 8
gradient_accumulation_steps = 8  # Effective batch = 64
```

### Speed Optimization

```python
# Use mixed precision
fp16 = True

# Enable gradient accumulation
gradient_accumulation_steps = 4

# Use multiple GPUs
# Automatically handled by Accelerate
```

## Deployment

### Save Model for Inference

```python
trainer.save_model('./outputs/final_model')
```

### Load Trained Model

```python
import torch
from models.gpt_model import GPTForCausalLM, GPTConfig

# Load config
config = GPTConfig.from_json_file('./outputs/final_model/config.json')

# Load model
model = GPTForCausalLM(config)
model.load_state_dict(
    torch.load('./outputs/final_model/model.pt')
)
```

## Multilingual Support

The model supports multiple Indian languages:

- **Hindi** (देवनागरी)
- **Tamil** (தமிழ்)
- **Telugu** (తెలుగు)
- **Kannada** (ಕನ್ನಡ)
- **Malayalam** (മലയാളം)
- **English**

### Multilingual Training

```bash
python train.py \
    --train_data /path/to/multilingual/data \
    --max_seq_length 2048 \
    --use_language_mixing
```

## Configuration

Edit `config/default_config.json` to customize:
- Model architecture (layers, heads, dimensions)
- Training hyperparameters (learning rate, batch size)
- Data paths and preprocessing
- Generation parameters
- Monitoring and logging

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 8

# Enable gradient accumulation
gradient_accumulation_steps = 8

# Use mixed precision
fp16 = True

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Slow Training
```python
# Use more workers for data loading
num_workers = 8

# Enable mixed precision
fp16 = True

# Use multiple GPUs
# Handled automatically by PyTorch Distributed
```

### Poor Generation Quality
```python
# Increase training data
# Lower learning rate
learning_rate = 1e-5

# Increase model size
model_size = 'large'

# Train for more epochs
num_epochs = 10
```

## Contributing

Contributions are welcome! Please follow the coding standards and add tests for new features.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@software{india_gpt_2024,
    title={India's First AI Model - GPT Implementation},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/india_gpt}
}
```

## Acknowledgments

- Based on the Transformer architecture from "Attention is All You Need"
- Inspired by OpenAI's GPT series
- Built with PyTorch and HuggingFace Transformers
- Data from NWorld corpus and public datasets

## References

1. Vaswani et al. (2017) - Attention is All You Need
2. Radford et al. (2019) - Language Models are Unsupervised Multitask Learners
3. HuggingFace Transformers Documentation
4. PyTorch Documentation

---

**Status**: Under Development
**Last Updated**: December 2024
**Version**: 1.0.0
