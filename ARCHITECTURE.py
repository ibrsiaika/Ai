"""
Architecture visualization and pipeline flow for India's GPT model.
"""

PIPELINE_FLOW = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   INDIA'S GPT MODEL - COMPLETE PIPELINE                    ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA PREPARATION                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Data (NWorld)                                                           │
│      ↓                                                                       │
│  DataProcessor (data/data_loader.py)                                        │
│      ├─ Load from file/HuggingFace                                          │
│      ├─ Language detection                                                  │
│      ├─ Text cleaning                                                       │
│      └─ Tokenization (GPT-2 tokenizer)                                      │
│      ↓                                                                       │
│  LanguageMixingDataset                                                      │
│      ├─ Uniform sampling                                                    │
│      ├─ Proportional sampling                                               │
│      └─ Temperature-based sampling                                          │
│      ↓                                                                       │
│  DataCollator                                                               │
│      ├─ Batching                                                            │
│      ├─ Padding                                                             │
│      └─ Attention mask creation                                             │
│      ↓                                                                       │
│  DataLoader (train, validation, test splits)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: MODEL ARCHITECTURE                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GPTConfig (Configuration)                                                  │
│      ├─ vocab_size: 50257                                                   │
│      ├─ context_length: 2048                                                │
│      ├─ d_model: 768 (or 384/1024/1536)                                     │
│      ├─ num_layers: 12                                                      │
│      ├─ num_heads: 12                                                       │
│      └─ d_ff: 3072                                                          │
│      ↓                                                                       │
│  GPTModel (Core Architecture)                                               │
│      ├─ Token Embedding (50257 → 768)                                       │
│      ├─ Position Embedding (2048 → 768)                                     │
│      └─ Transformer Layers [12x]                                            │
│          ├─ Layer Normalization                                             │
│          ├─ MultiHeadAttention (12 heads, 64 dim/head)                      │
│          ├─ Residual Connection                                             │
│          ├─ Layer Normalization                                             │
│          ├─ FeedForward (768 → 3072 → 768)                                  │
│          └─ Residual Connection                                             │
│      ├─ Output Layer Normalization                                          │
│      └─ Language Modeling Head (768 → 50257)                                │
│      ↓                                                                       │
│  GPTForCausalLM (Training Wrapper)                                          │
│      └─ Cross-Entropy Loss (with label shifting)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: TRAINING                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Training Configuration                                                     │
│      ├─ Optimizer: AdamW (lr=5e-4, weight_decay=0.01)                       │
│      ├─ Scheduler: Cosine + Warmup (1000 steps)                             │
│      ├─ Precision: Mixed (FP16)                                             │
│      ├─ Gradient Accumulation: 1-4 steps                                    │
│      ├─ Batch Size: 32 (configurable)                                       │
│      └─ Epochs: 3-10 (configurable)                                         │
│      ↓                                                                       │
│  Training Loop (trainer.py)                                                 │
│      └─ For each epoch:                                                     │
│          ├─ Forward Pass                                                    │
│          ├─ Loss Calculation                                                │
│          ├─ Backward Pass                                                   │
│          ├─ Gradient Clipping (max_grad_norm=1.0)                           │
│          ├─ Optimizer Step                                                  │
│          ├─ Learning Rate Step                                              │
│          ├─ Logging (loss, lr, perplexity)                                  │
│          ├─ Validation (every 500 steps)                                    │
│          └─ Checkpointing (every 1000 steps)                                │
│      ↓                                                                       │
│  Monitoring                                                                 │
│      ├─ Weights & Biases (optional)                                         │
│      ├─ TensorBoard                                                         │
│      └─ Local logging                                                       │
│      ↓                                                                       │
│  Checkpoints Saved                                                          │
│      ├─ Model weights (model.pt)                                            │
│      ├─ Optimizer state (optimizer.pt)                                      │
│      ├─ Scheduler state (scheduler.pt)                                      │
│      └─ Configuration (config.json)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: INFERENCE & GENERATION                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Prompt Input                                                               │
│      ↓                                                                       │
│  TextGenerator (inference/generator.py)                                     │
│      ├─ Tokenize prompt                                                     │
│      └─ Select decoding strategy:                                           │
│          ├─ Greedy: argmax(logits)                                          │
│          ├─ Sampling: sample from distribution                              │
│          │   ├─ Temperature scaling                                         │
│          │   ├─ Top-k filtering                                             │
│          │   └─ Top-p (nucleus) sampling                                    │
│          └─ Beam Search: multi-hypothesis search                            │
│      ↓                                                                       │
│  Generation Loop                                                            │
│      └─ While seq_len < max_length:                                         │
│          ├─ Forward pass                                                    │
│          ├─ Get next token logits                                           │
│          ├─ Apply decoding strategy                                         │
│          ├─ Append token                                                    │
│          └─ Check EOS token                                                 │
│      ↓                                                                       │
│  Post-processing                                                            │
│      ├─ Decode tokens to text                                               │
│      ├─ Remove special tokens                                               │
│      └─ Optional: remove prompt                                             │
│      ↓                                                                       │
│  Generated Text Output                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: EVALUATION                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Predictions vs References                                                  │
│      ↓                                                                       │
│  MetricsCalculator (utils/metrics.py)                                       │
│      ├─ Loss → Perplexity                                                   │
│      │   └─ PPL = exp(loss)                                                 │
│      ├─ Token Accuracy                                                      │
│      │   └─ Correct predictions / Total tokens                              │
│      ├─ BLEU Score                                                          │
│      │   ├─ 1-gram precision                                                │
│      │   ├─ 2-gram precision                                                │
│      │   ├─ 3-gram precision                                                │
│      │   ├─ 4-gram precision                                                │
│      │   └─ Brevity penalty                                                 │
│      ├─ ROUGE-1 (Unigram overlap)                                           │
│      │   └─ Recall, Precision, F1                                           │
│      ├─ ROUGE-2 (Bigram overlap)                                            │
│      │   └─ Recall, Precision, F1                                           │
│      └─ ROUGE-L (LCS-based)                                                 │
│          └─ Recall, Precision, F1                                           │
│      ↓                                                                       │
│  Results                                                                    │
│      ├─ Perplexity: Lower is better (< 20 is good)                         │
│      ├─ BLEU: 0-1 range (> 0.3 is good)                                     │
│      ├─ ROUGE-1 F1: 0-1 range (> 0.4 is good)                               │
│      └─ ROUGE-L F1: 0-1 range (> 0.4 is good)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: DEPLOYMENT                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Model Optimization (deployment.py)                                         │
│      ├─ Quantization: FP32 → FP16 → INT8                                    │
│      │   └─ Model size: 4GB → 2GB → 1GB                                     │
│      ├─ ONNX Export: PyTorch → ONNX                                          │
│      │   └─ Hardware acceleration                                           │
│      ├─ Pruning: Remove redundant weights                                   │
│      │   └─ Compression: 20-50%                                             │
│      └─ Benchmarking: Speed, memory, accuracy                               │
│      ↓                                                                       │
│  API Server (inference/api_server.py)                                       │
│      ├─ FastAPI endpoints                                                   │
│      ├─ Request validation                                                  │
│      ├─ Error handling                                                      │
│      └─ Response formatting                                                 │
│      ↓                                                                       │
│  Containerization (Docker)                                                  │
│      ├─ Dockerfile: Build image                                             │
│      ├─ docker-compose.yml: Orchestration                                   │
│      └─ Deployment to cloud/on-premise                                      │
│      ↓                                                                       │
│  Production Monitoring                                                      │
│      ├─ API metrics (latency, throughput)                                   │
│      ├─ Model performance (accuracy)                                        │
│      ├─ System metrics (CPU, GPU, memory)                                   │
│      └─ Alerts and logging                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                            SYSTEM ARCHITECTURE                             ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐                ║
║  │   Training  │────▶│  Inference   │────▶│  Deployment  │                ║
║  │   Pipeline  │     │   Pipeline   │     │   Pipeline   │                ║
║  └─────────────┘     └──────────────┘     └──────────────┘                ║
║        │                    │                     │                        ║
║        ├─ Data Loader       ├─ Generator         ├─ API Server            ║
║        ├─ Trainer           ├─ Classifier        ├─ Docker                ║
║        ├─ Optimizer         └─ Embeddings        ├─ Monitoring            ║
║        └─ Metrics                               └─ Cloud Deploy          ║
║                                                                            ║
║  ┌───────────────────────────────────────────────────────────────┐        ║
║  │  Utilities & Helpers                                          │        ║
║  │  ├─ ModelManager (save/load)                                  │        ║
║  │  ├─ ConfigManager (configuration)                             │        ║
║  │  ├─ DataUtils (processing)                                    │        ║
║  │  └─ PerformanceMonitor (tracking)                             │        ║
║  └───────────────────────────────────────────────────────────────┘        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════╗
║                          DATA FLOW DIAGRAM                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  NWorld Dataset                                                            ║
║        ↓                                                                   ║
║  Data Loader                                                               ║
║        ↓                                                                   ║
║  [Train | Validation | Test]                                               ║
║        ↓                                                                   ║
║  GPT Model                                                                 ║
║        ↓                                                                   ║
║  Loss Calculation                                                          ║
║        ↓                                                                   ║
║  Backpropagation                                                           ║
║        ↓                                                                   ║
║  Parameter Update                                                          ║
║        ↓                                                                   ║
║  Checkpoint Saved                                                          ║
║        ↓                                                                   ║
║  [Evaluation | Fine-tuning | Deployment]                                   ║
║        ↓                                                                   ║
║  [Generate | Classify | Embed]                                             ║
║        ↓                                                                   ║
║  REST API / Output                                                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

COMPONENT_DIAGRAM = """
┌──────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INTERACTION                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     ┌─────────────────────┐                                                 │
│     │  Data Loading       │                                                 │
│     │  (data_loader.py)   │                                                 │
│     └────────────┬────────┘                                                 │
│                  │                                                          │
│                  ▼                                                          │
│     ┌─────────────────────┐                                                 │
│     │  GPT Model          │                                                 │
│     │  (gpt_model.py)     │──────────────┐                                  │
│     └────────────┬────────┘              │                                  │
│                  │                       │                                  │
│        ┌─────────┴──────────┐            │                                  │
│        │                    │            │                                  │
│        ▼                    ▼            │                                  │
│   ┌────────┐           ┌─────────────┐  │                                  │
│   │Training│           │ Inference   │  │                                  │
│   │        │           │             │  │                                  │
│   │┌──────┐│           │┌──────────┐ │  │                                  │
│   ││Trainer││           ││Generator │ │  │                                  │
│   │├──────┤│           │├──────────┤ │  │                                  │
│   ││Loss  ││           ││Embedding │ │  │                                  │
│   │├──────┤│           │├──────────┤ │  │                                  │
│   ││Optim ││           ││Classifier│ │  │                                  │
│   │└──────┘│           │└──────────┘ │  │                                  │
│   └────────┘           └─────────────┘  │                                  │
│        │                    │            │                                  │
│        ▼                    │            │                                  │
│   ┌────────────┐            │            │                                  │
│   │ Metrics    │            │            │                                  │
│   │ (metrics.  │            │            │                                  │
│   │  py)       │            │            │                                  │
│   └────────────┘            │            │                                  │
│        │                    │            │                                  │
│        └────────┬───────────┤            │                                  │
│                 │           │            │                                  │
│                 ▼           ▼            │                                  │
│        ┌────────────────────┐            │                                  │
│        │  Checkpoints &     │            │                                  │
│        │  Results           │            │                                  │
│        └────────────────────┘            │                                  │
│                                          │                                  │
│                                          ▼                                  │
│                                   ┌─────────────────┐                       │
│                                   │ API Server      │                       │
│                                   │ (api_server.py) │                       │
│                                   │                 │                       │
│                                   │ /generate       │                       │
│                                   │ /embeddings     │                       │
│                                   │ /similarity     │                       │
│                                   │ /health         │                       │
│                                   └────────┬────────┘                       │
│                                            │                                │
│                                            ▼                                │
│                                   ┌─────────────────┐                       │
│                                   │ Deployment      │                       │
│                                   │ (deployment.py) │                       │
│                                   │                 │                       │
│                                   │ Quantization    │                       │
│                                   │ ONNX Export     │                       │
│                                   │ Docker          │                       │
│                                   └─────────────────┘                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    print(PIPELINE_FLOW)
    print("\n\n")
    print(COMPONENT_DIAGRAM)
