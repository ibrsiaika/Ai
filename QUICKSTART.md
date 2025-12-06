"""
Quick start guide for India's GPT model.
Get started training and using the model.
"""

# QUICK START GUIDE

## 1. Installation (5 minutes)

```bash
# Clone and setup
cd india_gpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Training (Variable)

### Basic Training
```bash
python train.py \
    --model_size base \
    --train_data wikitext \
    --num_epochs 3
```

### Advanced Training
```bash
python train.py \
    --model_size large \
    --train_data /path/to/nworld/data \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --fp16 \
    --use_wandb \
    --num_epochs 10
```

## 3. Inference

### Python API
```python
from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer

# Load model
model = create_gpt_model('base')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Generate text
generator = TextGenerator(model, tokenizer)
config = GenerationConfig(max_length=128, temperature=0.7)
text = generator.generate("India is", config)
print(text)
```

### REST API
```bash
# Start API server
python -m inference.api_server

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "India is",
    "max_length": 128,
    "temperature": 0.7,
    "do_sample": true
  }'
```

## 4. Evaluation

```python
from utils.metrics import MetricsCalculator

calculator = MetricsCalculator(tokenizer)
metrics = calculator.calculate_metrics(predictions, references, loss)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"BLEU: {metrics.bleu:.4f}")
print(f"ROUGE-1 F1: {metrics.rouge1['f1']:.4f}")
```

## 5. Model Optimization

```python
from deployment import ModelOptimizer

# Quantize model
quantized_model = ModelOptimizer.quantize_dynamic(model)

# Get model info
info = ModelOptimizer.get_model_size(model)
print(f"Model size (FP32): {info['size_fp32_gb']:.2f}GB")
print(f"Model size (INT8): {info['size_int8_gb']:.2f}GB")
```

## Project Structure

```
india_gpt/
├── data/                    # Data loading
├── models/                  # Model architecture
├── training/               # Training pipeline
├── inference/              # Inference & API
├── utils/                  # Utilities & metrics
├── config/                 # Configuration
├── notebooks/              # Tutorials
├── deploy/                 # Docker files
├── train.py               # Main script
└── requirements.txt       # Dependencies
```

## Key Components

### 1. Data Pipeline (data/data_loader.py)
- Load from files or HuggingFace Hub
- Multi-language support
- Automatic preprocessing
- Batching and padding

### 2. GPT Model (models/gpt_model.py)
- Full transformer architecture
- Multi-head attention
- Position embeddings
- Configurable sizes (small/base/large/xlarge)

### 3. Training (training/trainer.py)
- Mixed precision (FP16)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Weights & Biases integration

### 4. Inference (inference/generator.py)
- Multiple decoding strategies
- Beam search
- Temperature sampling
- Top-k/Top-p filtering

### 5. Metrics (utils/metrics.py)
- Perplexity
- BLEU score
- ROUGE scores
- Token accuracy

## Configuration

Edit `config/default_config.json`:
```json
{
  "model": {
    "model_size": "base",
    "vocab_size": 50257,
    "context_length": 2048
  },
  "training": {
    "num_epochs": 3,
    "batch_size": 32,
    "learning_rate": 5e-4
  }
}
```

## Training Tips

1. **Start small**: Train on smaller model first
2. **Use fp16**: Speeds up training 2-3x
3. **Gradient accumulation**: Simulate larger batches
4. **Learning rate**: 1e-4 to 5e-4 typically works
5. **Warmup steps**: 1000-5000 for large models
6. **Evaluation**: Evaluate every 500-1000 steps

## Deployment

### Docker
```bash
# Build
docker build -t india-gpt .

# Run
docker run --gpus all -p 8000:8000 india-gpt
```

### Docker Compose
```bash
docker-compose up -d
```

## Performance

### Model Sizes
- Small: ~100M params, ~400MB (FP32)
- Base: ~125M params, ~500MB (FP32)
- Large: ~355M params, ~1.4GB (FP32)
- XLarge: ~1B params, ~4GB (FP32)

### Inference Speed (V100, batch=1)
- Small: ~50ms per token
- Base: ~60ms per token
- Large: ~150ms per token

## Troubleshooting

### Out of Memory
- Reduce batch_size
- Enable fp16
- Reduce max_seq_length
- Enable gradient_accumulation

### Slow Training
- Use more workers (num_workers=8)
- Enable fp16
- Check GPU utilization

### Poor Results
- Use more training data
- Train for more epochs
- Adjust learning rate
- Increase model size

## Resources

- Transformer paper: https://arxiv.org/abs/1706.03762
- HuggingFace: https://huggingface.co/
- PyTorch: https://pytorch.org/
- Weights & Biases: https://wandb.ai/

## Next Steps

1. **Prepare NWorld Data**: Download and process dataset
2. **Fine-tune**: Train on specific domain data
3. **Evaluate**: Run comprehensive benchmarks
4. **Deploy**: Set up production API
5. **Monitor**: Track performance in production

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check documentation
- Review example notebooks
- Contact team

---

**Status**: Ready for training
**Last Updated**: December 2024
