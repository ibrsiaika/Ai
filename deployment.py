"""
Deployment and optimization guide for India's GPT model.
Includes ONNX export, quantization, and deployment strategies.
"""

import torch
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize model for deployment."""
    
    @staticmethod
    def convert_to_onnx(model,
                       tokenizer,
                       output_path: str = './model.onnx',
                       opset_version: int = 14):
        """
        Convert model to ONNX format.
        
        Args:
            model: GPT model
            tokenizer: Tokenizer
            output_path: Output path
            opset_version: ONNX opset version
        """
        try:
            import torch.onnx
            
            dummy_input = torch.randint(0, 50257, (1, 512))
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input_ids'],
                output_names=['logits'],
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
        
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    @staticmethod
    def quantize_dynamic(model,
                        output_path: str = './model_quantized.pt'):
        """
        Apply dynamic quantization to model.
        
        Args:
            model: GPT model
            output_path: Output path
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            torch.save(quantized_model.state_dict(), output_path)
            logger.info(f"Model quantized and saved: {output_path}")
            
            return quantized_model
        
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise
    
    @staticmethod
    def prune_model(model, pruning_amount: float = 0.3):
        """
        Apply structured pruning to model.
        
        Args:
            model: GPT model
            pruning_amount: Amount to prune (0-1)
        """
        try:
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=pruning_amount
                    )
            
            logger.info(f"Model pruned by {pruning_amount*100}%")
            return model
        
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise
    
    @staticmethod
    def get_model_size(model) -> Dict[str, float]:
        """
        Get model size information.
        
        Args:
            model: GPT model
        
        Returns:
            Dictionary with size information
        """
        # Full precision
        num_params = sum(p.numel() for p in model.parameters())
        size_fp32 = num_params * 4 / (1024**3)  # GB
        size_fp16 = num_params * 2 / (1024**3)  # GB
        size_int8 = num_params * 1 / (1024**3)  # GB
        
        return {
            'num_parameters': num_params,
            'size_fp32_gb': size_fp32,
            'size_fp16_gb': size_fp16,
            'size_int8_gb': size_int8
        }


class DeploymentConfig:
    """Configuration for deployment."""
    
    # Docker deployment
    DOCKERFILE = """
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy model files
COPY . .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "inference/api_server.py"]
"""
    
    # Docker compose
    DOCKER_COMPOSE = """
version: '3.8'

services:
  india-gpt:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
    restart: unless-stopped
"""
    
    @staticmethod
    def create_docker_files(output_dir: str = './deploy'):
        """Create Docker deployment files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save Dockerfile
        with open(f'{output_dir}/Dockerfile', 'w') as f:
            f.write(DeploymentConfig.DOCKERFILE)
        
        # Save docker-compose.yml
        with open(f'{output_dir}/docker-compose.yml', 'w') as f:
            f.write(DeploymentConfig.DOCKER_COMPOSE)
        
        logger.info(f"Docker files created in {output_dir}")


class PerformanceBenchmark:
    """Benchmark model performance."""
    
    @staticmethod
    def benchmark_inference(model,
                           tokenizer,
                           batch_sizes: list = None,
                           seq_lengths: list = None,
                           num_runs: int = 10):
        """
        Benchmark inference performance.
        
        Args:
            model: GPT model
            tokenizer: Tokenizer
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            num_runs: Number of runs per config
        
        Returns:
            Benchmark results
        """
        import time
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
        if seq_lengths is None:
            seq_lengths = [128, 256, 512, 1024]
        
        device = next(model.parameters()).device
        results = {}
        
        model.eval()
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    # Create dummy input
                    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
                    
                    # Warmup
                    for _ in range(2):
                        _ = model(input_ids=input_ids)
                    
                    # Benchmark
                    torch.cuda.synchronize()
                    start = time.time()
                    
                    for _ in range(num_runs):
                        _ = model(input_ids=input_ids)
                    
                    torch.cuda.synchronize()
                    elapsed = time.time() - start
                    
                    throughput = (batch_size * seq_len * num_runs) / elapsed
                    latency = (elapsed / num_runs) * 1000  # ms
                    
                    key = f"bs={batch_size}, seq_len={seq_len}"
                    results[key] = {
                        'throughput_tokens_per_sec': throughput,
                        'latency_ms': latency,
                        'throughput_samples_per_sec': (batch_size * num_runs) / elapsed
                    }
        
        return results


# Deployment checklist
DEPLOYMENT_CHECKLIST = """
## India's GPT Model - Deployment Checklist

### Pre-Deployment
- [ ] Model training completed
- [ ] Best model saved and verified
- [ ] Evaluation metrics calculated
- [ ] Model performance benchmarked
- [ ] Hyperparameters documented

### Optimization
- [ ] Model quantized (INT8/FP16)
- [ ] ONNX export tested
- [ ] Pruning applied if needed
- [ ] Model size reduced
- [ ] Inference speed tested

### Deployment Setup
- [ ] Docker image built
- [ ] Environment variables configured
- [ ] API endpoints tested
- [ ] Load balancing configured
- [ ] Monitoring setup (Prometheus/Grafana)

### API Server
- [ ] FastAPI server running
- [ ] Health check endpoint working
- [ ] Generation endpoint tested
- [ ] Embeddings endpoint tested
- [ ] Error handling implemented

### Monitoring & Logging
- [ ] Request/response logging enabled
- [ ] Error tracking configured
- [ ] Performance metrics collected
- [ ] Alerts configured
- [ ] Dashboard created

### Documentation
- [ ] API documentation completed
- [ ] Usage examples provided
- [ ] Troubleshooting guide created
- [ ] Performance benchmarks documented
- [ ] Deployment guide written

### Security
- [ ] API authentication configured
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] Output sanitization applied
- [ ] Security headers added

### Production
- [ ] Staging deployment tested
- [ ] Rollback procedure documented
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan created
- [ ] Production monitoring active
"""


def create_deployment_guide(output_path: str = './DEPLOYMENT.md'):
    """Create deployment guide."""
    with open(output_path, 'w') as f:
        f.write(DEPLOYMENT_CHECKLIST)
    logger.info(f"Deployment guide created: {output_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Create deployment files
    DeploymentConfig.create_docker_files('./deploy')
    create_deployment_guide('./DEPLOYMENT.md')
    
    print("Deployment setup completed!")
