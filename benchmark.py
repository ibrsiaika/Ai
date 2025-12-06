"""
Benchmark script to compare India's GPT model with other AI models.
Evaluates performance on various tasks.
"""

import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from utils.metrics import MetricsCalculator


class ModelBenchmark:
    """Benchmark model performance."""
    
    def __init__(self, model, tokenizer):
        """Initialize benchmark.
        
        Args:
            model: GPT model to benchmark
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerator(model, tokenizer)
        self.metrics_calculator = MetricsCalculator(tokenizer)
        
        self.results = {
            'model_info': {},
            'performance': {},
            'quality': {},
            'tasks': {}
        }
        
    def benchmark_model_info(self):
        """Gather model information."""
        print("\n" + "="*80)
        print("MODEL INFORMATION")
        print("="*80)
        
        config = self.model.config
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'vocab_size': config.vocab_size,
            'context_length': config.context_length,
            'd_model': config.d_model,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'd_ff': config.d_ff,
            'total_parameters': num_params,
            'trainable_parameters': trainable_params,
            'device': str(next(self.model.parameters()).device)
        }
        
        print(f"\nTotal parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Context length: {config.context_length}")
        print(f"Model dimension: {config.d_model}")
        print(f"Number of layers: {config.num_layers}")
        
        self.results['model_info'] = info
        return info
        
    def benchmark_inference_speed(self, num_samples: int = 100):
        """Benchmark inference speed.
        
        Args:
            num_samples: Number of samples to test
        """
        print("\n" + "="*80)
        print("INFERENCE SPEED BENCHMARK")
        print("="*80)
        
        prompts = [
            "The future of AI",
            "In machine learning",
            "Python programming",
            "Data science is",
            "Neural networks work by"
        ] * (num_samples // 5)
        
        config = GenerationConfig(
            max_length=50,
            do_sample=False
        )
        
        # Warmup
        print("\nWarmup...")
        self.generator.generate(prompts[0], config)
        
        # Benchmark
        print(f"Generating {num_samples} samples...")
        
        start_time = time.time()
        for prompt in prompts[:num_samples]:
            _ = self.generator.generate(prompt, config)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_samples
        throughput = num_samples / total_time
        
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Average time per sample: {avg_time*1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} samples/s")
        
        self.results['performance'] = {
            'total_time_seconds': total_time,
            'avg_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'num_samples': num_samples
        }
        
        return self.results['performance']
        
    def benchmark_generation_quality(self):
        """Benchmark generation quality."""
        print("\n" + "="*80)
        print("GENERATION QUALITY BENCHMARK")
        print("="*80)
        
        test_cases = [
            {
                'prompt': 'The capital of France is',
                'reference': 'The capital of France is Paris',
                'category': 'factual'
            },
            {
                'prompt': 'To sort a list in Python, you can use',
                'reference': 'To sort a list in Python, you can use the sort() method or sorted() function',
                'category': 'technical'
            },
            {
                'prompt': 'Machine learning is a subset of',
                'reference': 'Machine learning is a subset of artificial intelligence',
                'category': 'conceptual'
            }
        ]
        
        config = GenerationConfig(
            max_length=100,
            temperature=0.1,
            do_sample=False
        )
        
        predictions = []
        references = []
        
        for case in test_cases:
            generated = self.generator.generate(case['prompt'], config)
            predictions.append(generated)
            references.append(case['reference'])
            
            print(f"\nCategory: {case['category']}")
            print(f"Prompt: {case['prompt']}")
            print(f"Generated: {generated}")
            print(f"Reference: {case['reference']}")
            
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            predictions, references, loss=1.0
        )
        
        quality = {
            'bleu': metrics.bleu,
            'rouge1_f1': metrics.rouge1['f1'],
            'rouge2_f1': metrics.rouge2['f1'],
            'rougeL_f1': metrics.rougeL['f1']
        }
        
        print("\n" + "-"*80)
        print("Quality Metrics:")
        print(f"BLEU: {metrics.bleu:.4f}")
        print(f"ROUGE-1 F1: {metrics.rouge1['f1']:.4f}")
        print(f"ROUGE-2 F1: {metrics.rouge2['f1']:.4f}")
        print(f"ROUGE-L F1: {metrics.rougeL['f1']:.4f}")
        
        self.results['quality'] = quality
        return quality
        
    def benchmark_task_performance(self):
        """Benchmark performance on specific tasks."""
        print("\n" + "="*80)
        print("TASK-SPECIFIC PERFORMANCE")
        print("="*80)
        
        tasks = {
            'code_completion': {
                'prompts': [
                    'def factorial(n):\n    if n == 0:',
                    'class Node:\n    def __init__(self, value):',
                    'def is_prime(n):\n    if n < 2:'
                ],
                'config': GenerationConfig(max_length=100, temperature=0.2, do_sample=True)
            },
            'question_answering': {
                'prompts': [
                    'Question: What is Python?\nAnswer:',
                    'Question: How does a neural network learn?\nAnswer:',
                    'Question: What is recursion?\nAnswer:'
                ],
                'config': GenerationConfig(max_length=150, temperature=0.5, do_sample=True)
            },
            'creative_writing': {
                'prompts': [
                    'Once upon a time in a distant land,',
                    'The scientist discovered something incredible:',
                    'In the year 2050,'
                ],
                'config': GenerationConfig(max_length=200, temperature=0.8, do_sample=True)
            }
        }
        
        task_results = {}
        
        for task_name, task_data in tasks.items():
            print(f"\n{'-'*80}")
            print(f"Task: {task_name.replace('_', ' ').title()}")
            print(f"{'-'*80}")
            
            outputs = []
            start_time = time.time()
            
            for prompt in task_data['prompts']:
                output = self.generator.generate(prompt, task_data['config'])
                outputs.append(output)
                print(f"\nPrompt: {prompt[:50]}...")
                print(f"Output: {output[:100]}...")
                
            task_time = time.time() - start_time
            
            task_results[task_name] = {
                'num_prompts': len(task_data['prompts']),
                'avg_time': task_time / len(task_data['prompts']),
                'total_time': task_time,
                'samples': outputs[:3]  # Store first 3 samples
            }
            
        self.results['tasks'] = task_results
        return task_results
        
    def run_full_benchmark(self):
        """Run all benchmarks."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL BENCHMARK")
        print("="*80)
        
        # Run all benchmarks
        self.benchmark_model_info()
        self.benchmark_inference_speed()
        self.benchmark_generation_quality()
        self.benchmark_task_performance()
        
        # Print summary
        self.print_summary()
        
        return self.results
        
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        if 'model_info' in self.results:
            print(f"\nModel Parameters: {self.results['model_info']['total_parameters']:,}")
            
        if 'performance' in self.results:
            print(f"\nInference Speed:")
            print(f"  Average: {self.results['performance']['avg_time_ms']:.2f}ms/sample")
            print(f"  Throughput: {self.results['performance']['throughput_samples_per_sec']:.2f} samples/s")
            
        if 'quality' in self.results:
            print(f"\nGeneration Quality:")
            print(f"  BLEU: {self.results['quality']['bleu']:.4f}")
            print(f"  ROUGE-L F1: {self.results['quality']['rougeL_f1']:.4f}")
            
        if 'tasks' in self.results:
            print(f"\nTask Performance:")
            for task, data in self.results['tasks'].items():
                print(f"  {task.replace('_', ' ').title()}: {data['avg_time']*1000:.2f}ms/prompt")
                
    def save_results(self, output_path: str):
        """Save benchmark results."""
        # Convert to serializable format
        results_to_save = {
            'model_info': self.results.get('model_info', {}),
            'performance': self.results.get('performance', {}),
            'quality': self.results.get('quality', {}),
            'tasks': {}
        }
        
        # Filter task results to remove non-serializable data
        for task, data in self.results.get('tasks', {}).items():
            results_to_save['tasks'][task] = {
                'num_prompts': data['num_prompts'],
                'avg_time': data['avg_time'],
                'total_time': data['total_time']
            }
            
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
            
        print(f"\nBenchmark results saved to: {output_path}")


def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark India's GPT model performance"
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large', 'xlarge'],
        help='Model size to benchmark'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Path to save benchmark results'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples for speed benchmark'
    )
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=['all'],
        choices=['all', 'info', 'speed', 'quality', 'tasks'],
        help='Which benchmarks to run'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("INDIA'S GPT MODEL - PERFORMANCE BENCHMARK")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {args.model_size}")
    if args.model_path:
        model = torch.load(args.model_path, map_location='cpu')
    else:
        model = create_gpt_model(args.model_size)
        
    model.eval()  # Set to evaluation mode
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Create benchmark
    benchmark = ModelBenchmark(model, tokenizer)
    
    # Run benchmarks
    if 'all' in args.benchmarks:
        benchmark.run_full_benchmark()
    else:
        if 'info' in args.benchmarks:
            benchmark.benchmark_model_info()
        if 'speed' in args.benchmarks:
            benchmark.benchmark_inference_speed(args.num_samples)
        if 'quality' in args.benchmarks:
            benchmark.benchmark_generation_quality()
        if 'tasks' in args.benchmarks:
            benchmark.benchmark_task_performance()
            
        benchmark.print_summary()
        
    # Save results
    benchmark.save_results(args.output)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
