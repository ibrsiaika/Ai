# Testing Guide for India's GPT Model

## Overview

This guide provides comprehensive information on testing India's GPT model, including unit tests, integration tests, validation, and benchmarking.

## Table of Contents

1. [Unit Tests](#unit-tests)
2. [Validation](#validation)
3. [Interactive Testing](#interactive-testing)
4. [Benchmarking](#benchmarking)
5. [Configuration](#configuration)

---

## Unit Tests

### Running Tests

Run all tests:
```bash
python -m unittest discover tests -v
```

Run specific test file:
```bash
python -m unittest tests.test_model -v
```

Run specific test class:
```bash
python -m unittest tests.test_model.TestGPTConfig -v
```

### Test Coverage

#### Model Tests (`tests/test_model.py`)
- Configuration validation
- Multi-head attention mechanism
- Feed-forward networks
- Transformer blocks
- Complete model architecture
- Text generation
- Parameter counting

#### Data Loading Tests (`tests/test_data_loader.py`)
- Data processor initialization
- Tokenization functions
- Dataset loading
- Batch collation
- Padding and masking

#### Generation Tests (`tests/test_generation.py`)
- Generation configuration
- Greedy decoding
- Sampling strategies
- Top-k filtering
- Top-p (nucleus) sampling
- Batch generation

#### Metrics Tests (`tests/test_metrics.py`)
- Perplexity calculation
- BLEU scores
- ROUGE metrics
- Accuracy metrics

---

## Validation

### Comprehensive Model Validation

The `validate_model.py` script provides end-to-end validation of model capabilities.

```bash
python validate_model.py --model_size base --output validation_results.json
```

### Options

- `--model_size`: Model size (small/base/large/xlarge)
- `--model_path`: Path to trained checkpoint (optional)
- `--config`: Path to configuration file
- `--output`: Path to save validation results
- `--tests`: Which tests to run

### Test Categories

#### 1. Code Generation
Tests the model's ability to:
- Complete Python functions
- Implement classes
- Generate JavaScript/other languages
- Write algorithms

Example:
```bash
python validate_model.py --tests code
```

#### 2. Conversation
Tests conversational capabilities:
- Question answering
- Natural dialogue
- Context understanding
- Topic coherence

Example:
```bash
python validate_model.py --tests conversation
```

#### 3. Complex Reasoning
Tests logical and analytical thinking:
- Logical deduction
- Mathematical problem-solving
- Critical analysis
- Step-by-step reasoning
- Concept explanation

Example:
```bash
python validate_model.py --tests reasoning
```

#### 4. Multilingual
Tests language capabilities:
- Translation
- Code-switching
- Multilingual completion
- Language understanding

Example:
```bash
python validate_model.py --tests multilingual
```

#### 5. General Completion
Tests general text completion:
- Topic continuation
- Coherent generation
- Factual accuracy
- Fluency

Example:
```bash
python validate_model.py --tests completion
```

---

## Interactive Testing

### Starting Interactive Tester

```bash
python interactive_test.py --model_size base
```

### Features

#### 1. Text Generation
Generate text with various presets:
```
Command: generate
Prompt: The future of artificial intelligence
Preset: conversation
```

#### 2. Code Generation
Specialized mode for code:
```
Command: code
Prompt: def quicksort(arr):
```

#### 3. Chat Mode
Conversational interface with context:
```
Command: chat
You: What is machine learning?
Assistant: [response]
You: How does it work?
Assistant: [contextual response]
```

#### 4. Batch Generation
Generate multiple outputs at once:
```
Command: batch
Prompt 1: Hello world
Prompt 2: Good morning
Prompt 3: How are you
```

### Available Presets

#### Code Generation
- Temperature: 0.2
- Top-k: 40
- Top-p: 0.9
- Max length: 512

#### Creative Writing
- Temperature: 0.8
- Top-k: 50
- Top-p: 0.95
- Max length: 1024

#### Conversation
- Temperature: 0.7
- Top-k: 50
- Top-p: 0.9
- Max length: 256

#### Accurate Completion
- Temperature: 0.1
- Top-k: 10
- Beam search: 4
- Max length: 256

---

## Benchmarking

### Running Benchmarks

Full benchmark:
```bash
python benchmark.py --model_size base --output benchmark_results.json
```

Specific benchmarks:
```bash
python benchmark.py --benchmarks speed quality tasks
```

### Benchmark Categories

#### 1. Model Information
- Parameter count
- Architecture details
- Memory footprint
- Device information

#### 2. Inference Speed
- Average latency per sample
- Throughput (samples/second)
- Batch processing speed
- GPU utilization

#### 3. Generation Quality
- BLEU scores
- ROUGE metrics
- Coherence evaluation
- Factual accuracy

#### 4. Task-Specific Performance
- Code completion speed/quality
- Question answering accuracy
- Creative writing diversity
- Translation quality

### Benchmark Options

```bash
python benchmark.py \
    --model_size base \
    --num_samples 100 \
    --benchmarks all \
    --output results.json
```

- `--model_size`: Model size to benchmark
- `--model_path`: Path to checkpoint (optional)
- `--num_samples`: Number of samples for speed test
- `--benchmarks`: Which benchmarks to run
- `--output`: Where to save results

### Interpreting Results

#### Speed Metrics
- **Latency < 100ms**: Excellent for real-time use
- **Latency 100-500ms**: Good for interactive applications
- **Latency > 500ms**: Suitable for batch processing

#### Quality Metrics
- **BLEU > 0.3**: Good quality generation
- **ROUGE-L > 0.4**: High overlap with reference
- **Perplexity < 50**: Well-calibrated model

---

## Configuration

### Default Configuration

Located in `config/default_config.json`:
- Basic model settings
- Standard training parameters
- Simple generation config

### Advanced Configuration

Located in `config/advanced_config.json`:
- Large model settings
- Multiple generation presets
- Task-specific configurations
- Competitive features

### Configuration Structure

```json
{
  "model": {
    "model_size": "base",
    "vocab_size": 50257,
    "context_length": 2048,
    ...
  },
  "training": {
    "num_epochs": 3,
    "batch_size": 32,
    ...
  },
  "generation": {
    "presets": {
      "code_generation": {...},
      "creative_writing": {...},
      "conversation": {...}
    }
  }
}
```

### Using Custom Configurations

With training:
```bash
python train.py --config config/advanced_config.json
```

With validation:
```bash
python validate_model.py --config config/advanced_config.json
```

With interactive testing:
```bash
python interactive_test.py --config config/advanced_config.json
```

---

## Best Practices

### For Testing

1. **Run unit tests frequently** during development
2. **Validate on diverse inputs** to catch edge cases
3. **Use multiple presets** to test different scenarios
4. **Compare with baselines** to track improvements

### For Validation

1. **Start with small model** for quick iteration
2. **Use validation script** before deploying
3. **Test all task categories** to ensure coverage
4. **Save validation results** for comparison

### For Benchmarking

1. **Benchmark on target hardware** for realistic metrics
2. **Use consistent test data** for fair comparisons
3. **Run multiple times** to account for variance
4. **Compare with other models** to gauge competitiveness

---

## Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Use smaller model
python validate_model.py --model_size small

# Reduce batch size in interactive testing
# Reduce max_length in generation config
```

#### Slow Generation
```bash
# Use greedy decoding (faster)
preset: accurate

# Reduce max_length
max_length: 128

# Disable sampling
do_sample: false
```

#### Poor Quality
```bash
# Use larger model
--model_size large

# Adjust temperature
temperature: 0.7

# Use beam search
num_beams: 4
```

---

## Advanced Usage

### Custom Test Cases

Add your own test cases to `validate_model.py`:

```python
custom_prompts = [
    {
        'prompt': 'Your custom prompt',
        'task': 'Your task description'
    }
]
```

### Custom Presets

Add presets to `config/advanced_config.json`:

```json
"your_preset": {
  "max_length": 256,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.95,
  "do_sample": true
}
```

### Automated Testing

Create a test script:

```bash
#!/bin/bash
# Run all tests
python -m unittest discover tests
python validate_model.py --tests all
python benchmark.py --benchmarks all
```

---

## Performance Targets

### Model Quality Targets

| Task | Target Metric | Value |
|------|--------------|-------|
| Code Completion | BLEU | > 0.4 |
| Conversation | Coherence | > 0.7 |
| Translation | BLEU | > 0.3 |
| General | Perplexity | < 50 |

### Speed Targets

| Model Size | Latency | Throughput |
|------------|---------|------------|
| Small | < 50ms | > 20 samples/s |
| Base | < 100ms | > 10 samples/s |
| Large | < 200ms | > 5 samples/s |

---

## Continuous Testing

### Pre-commit Testing
```bash
# Run before committing
python -m unittest discover tests -v
```

### Pre-deployment Testing
```bash
# Full validation before deployment
python validate_model.py --tests all
python benchmark.py --benchmarks all
```

---

## Support

For issues or questions about testing:
1. Check this documentation
2. Review test examples in `tests/` directory
3. Run interactive tester for experimentation
4. Check validation results for insights

---

**Last Updated**: December 2024
**Version**: 1.0.0
