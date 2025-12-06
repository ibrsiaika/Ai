"""
Comprehensive testing and validation script for India's GPT model.
Tests model capabilities including code generation, conversation, and complex reasoning.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.gpt_model import create_gpt_model, GPTForCausalLM
from inference.generator import TextGenerator, GenerationConfig
from utils.metrics import MetricsCalculator


class ModelValidator:
    """Comprehensive model validation and testing."""
    
    def __init__(self, model: GPTForCausalLM, tokenizer: PreTrainedTokenizer, config_path: str = None):
        """Initialize validator.
        
        Args:
            model: GPT model to validate
            tokenizer: Tokenizer for text processing
            config_path: Path to configuration file
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerator(model, tokenizer)
        self.metrics_calculator = MetricsCalculator(tokenizer)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
            
        self.results = {
            'code_generation': {},
            'conversation': {},
            'reasoning': {},
            'general_completion': {},
            'multilingual': {}
        }
        
    def test_code_generation(self) -> Dict:
        """Test code generation capabilities."""
        print("\n" + "="*80)
        print("TESTING CODE GENERATION")
        print("="*80)
        
        code_prompts = [
            {
                'prompt': 'def fibonacci(n):\n    """Calculate nth Fibonacci number"""',
                'task': 'Complete Fibonacci function'
            },
            {
                'prompt': 'class BinaryTree:\n    def __init__(self, value):',
                'task': 'Complete Binary Tree class'
            },
            {
                'prompt': '# Function to reverse a string\ndef reverse_string(s):',
                'task': 'String reversal function'
            },
            {
                'prompt': 'import numpy as np\n\n# Calculate mean and standard deviation',
                'task': 'Numpy statistics'
            },
            {
                'prompt': '// JavaScript: Create a promise that resolves after delay\nfunction delay(ms) {',
                'task': 'JavaScript async function'
            }
        ]
        
        config = GenerationConfig(
            max_length=256,
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        results = []
        for item in code_prompts:
            print(f"\n{'─'*80}")
            print(f"Task: {item['task']}")
            print(f"Prompt:\n{item['prompt']}")
            print(f"\nGenerated:")
            
            generated = self.generator.generate(item['prompt'], config)
            print(generated)
            
            results.append({
                'task': item['task'],
                'prompt': item['prompt'],
                'generated': generated,
                'length': len(generated)
            })
            
        self.results['code_generation'] = {
            'total_tests': len(code_prompts),
            'results': results
        }
        
        return self.results['code_generation']
        
    def test_conversation(self) -> Dict:
        """Test conversational capabilities."""
        print("\n" + "="*80)
        print("TESTING CONVERSATION")
        print("="*80)
        
        conversations = [
            {
                'prompt': 'Human: What is artificial intelligence?\nAssistant:',
                'topic': 'AI definition'
            },
            {
                'prompt': 'Human: Can you explain how neural networks work?\nAssistant:',
                'topic': 'Neural networks'
            },
            {
                'prompt': 'Human: What are the benefits of learning Python?\nAssistant:',
                'topic': 'Programming languages'
            },
            {
                'prompt': 'Human: Tell me about India\'s contribution to mathematics.\nAssistant:',
                'topic': 'History of mathematics'
            },
            {
                'prompt': 'Human: How can I improve my coding skills?\nAssistant:',
                'topic': 'Learning advice'
            }
        ]
        
        config = GenerationConfig(
            max_length=256,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15
        )
        
        results = []
        for item in conversations:
            print(f"\n{'─'*80}")
            print(f"Topic: {item['topic']}")
            print(f"Prompt:\n{item['prompt']}")
            print(f"\nResponse:")
            
            generated = self.generator.generate(item['prompt'], config)
            print(generated)
            
            results.append({
                'topic': item['topic'],
                'prompt': item['prompt'],
                'response': generated,
                'length': len(generated)
            })
            
        self.results['conversation'] = {
            'total_tests': len(conversations),
            'results': results
        }
        
        return self.results['conversation']
        
    def test_complex_reasoning(self) -> Dict:
        """Test complex reasoning and problem-solving."""
        print("\n" + "="*80)
        print("TESTING COMPLEX REASONING")
        print("="*80)
        
        reasoning_prompts = [
            {
                'prompt': 'Question: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?\nAnswer:',
                'task': 'Logical reasoning'
            },
            {
                'prompt': 'Problem: A train travels 120 km in 2 hours. At this rate, how far will it travel in 5 hours?\nSolution:',
                'task': 'Math problem'
            },
            {
                'prompt': 'Analyze: What are the advantages and disadvantages of renewable energy?\nAnalysis:',
                'task': 'Critical analysis'
            },
            {
                'prompt': 'Chain of thought: To make a sandwich, I need to\n1.',
                'task': 'Step-by-step reasoning'
            },
            {
                'prompt': 'Explain the concept: Machine learning is different from traditional programming because',
                'task': 'Concept explanation'
            }
        ]
        
        config = GenerationConfig(
            max_length=300,
            temperature=0.5,
            top_k=40,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        results = []
        for item in reasoning_prompts:
            print(f"\n{'─'*80}")
            print(f"Task: {item['task']}")
            print(f"Prompt:\n{item['prompt']}")
            print(f"\nGenerated:")
            
            generated = self.generator.generate(item['prompt'], config)
            print(generated)
            
            results.append({
                'task': item['task'],
                'prompt': item['prompt'],
                'generated': generated,
                'length': len(generated)
            })
            
        self.results['reasoning'] = {
            'total_tests': len(reasoning_prompts),
            'results': results
        }
        
        return self.results['reasoning']
        
    def test_multilingual(self) -> Dict:
        """Test multilingual capabilities."""
        print("\n" + "="*80)
        print("TESTING MULTILINGUAL CAPABILITIES")
        print("="*80)
        
        multilingual_prompts = [
            {
                'prompt': 'Translate to Hindi: Hello, how are you?\nHindi:',
                'language': 'Hindi translation'
            },
            {
                'prompt': 'भारत एक महान देश है क्योंकि',
                'language': 'Hindi completion'
            },
            {
                'prompt': 'Tamil: வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?\nEnglish:',
                'language': 'Tamil to English'
            },
            {
                'prompt': 'Mix languages: India is known as भारत in Hindi and',
                'language': 'Code-switching'
            }
        ]
        
        config = GenerationConfig(
            max_length=200,
            temperature=0.6,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        results = []
        for item in multilingual_prompts:
            print(f"\n{'─'*80}")
            print(f"Language: {item['language']}")
            print(f"Prompt:\n{item['prompt']}")
            print(f"\nGenerated:")
            
            generated = self.generator.generate(item['prompt'], config)
            print(generated)
            
            results.append({
                'language': item['language'],
                'prompt': item['prompt'],
                'generated': generated
            })
            
        self.results['multilingual'] = {
            'total_tests': len(multilingual_prompts),
            'results': results
        }
        
        return self.results['multilingual']
        
    def test_general_completion(self) -> Dict:
        """Test general text completion."""
        print("\n" + "="*80)
        print("TESTING GENERAL COMPLETION")
        print("="*80)
        
        prompts = [
            'The future of technology is',
            'Artificial intelligence will revolutionize',
            'In the field of medicine,',
            'Climate change is affecting',
            'The importance of education lies in'
        ]
        
        config = GenerationConfig(
            max_length=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
        results = []
        for prompt in prompts:
            print(f"\n{'─'*80}")
            print(f"Prompt: {prompt}")
            print(f"Generated:")
            
            generated = self.generator.generate(prompt, config)
            print(generated)
            
            results.append({
                'prompt': prompt,
                'generated': generated
            })
            
        self.results['general_completion'] = {
            'total_tests': len(prompts),
            'results': results
        }
        
        return self.results['general_completion']
        
    def run_all_tests(self) -> Dict:
        """Run all validation tests."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE MODEL VALIDATION")
        print("="*80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {next(self.model.parameters()).device}")
        
        # Run all test categories
        self.test_code_generation()
        self.test_conversation()
        self.test_complex_reasoning()
        self.test_multilingual()
        self.test_general_completion()
        
        # Print summary
        self.print_summary()
        
        return self.results
        
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        total_tests = sum(
            category.get('total_tests', 0) 
            for category in self.results.values() 
            if isinstance(category, dict)
        )
        
        print(f"\nTotal tests run: {total_tests}")
        print("\nResults by category:")
        for category, data in self.results.items():
            if isinstance(data, dict) and 'total_tests' in data:
                print(f"  - {category.replace('_', ' ').title()}: {data['total_tests']} tests")
                
    def save_results(self, output_path: str):
        """Save validation results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate India's GPT model capabilities"
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large', 'xlarge'],
        help='Model size to test'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/advanced_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results.json',
        help='Path to save validation results'
    )
    parser.add_argument(
        '--tests',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'code', 'conversation', 'reasoning', 'multilingual', 'completion'],
        help='Which tests to run'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("INDIA'S GPT MODEL - COMPREHENSIVE VALIDATION")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {args.model_size}")
    if args.model_path:
        print(f"From checkpoint: {args.model_path}")
        # Load from checkpoint
        model = torch.load(args.model_path, map_location='cpu')
    else:
        model = create_gpt_model(args.model_size)
        
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Create validator
    validator = ModelValidator(model, tokenizer, args.config)
    
    # Run tests
    if 'all' in args.tests:
        validator.run_all_tests()
    else:
        if 'code' in args.tests:
            validator.test_code_generation()
        if 'conversation' in args.tests:
            validator.test_conversation()
        if 'reasoning' in args.tests:
            validator.test_complex_reasoning()
        if 'multilingual' in args.tests:
            validator.test_multilingual()
        if 'completion' in args.tests:
            validator.test_general_completion()
            
        validator.print_summary()
        
    # Save results
    validator.save_results(args.output)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
