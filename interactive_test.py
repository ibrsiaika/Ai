"""
Interactive CLI tool for testing India's GPT model.
Provides an easy-to-use interface for testing model capabilities.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig


class InteractiveTester:
    """Interactive testing interface for GPT model."""
    
    def __init__(self, model, tokenizer, config_path: Optional[str] = None):
        """Initialize interactive tester.
        
        Args:
            model: GPT model
            tokenizer: Tokenizer
            config_path: Path to configuration file
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerator(model, tokenizer)
        
        # Load advanced config if provided
        self.presets = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                if 'generation' in config_data and 'presets' in config_data['generation']:
                    self.presets = config_data['generation']['presets']
                    
        # Default presets if not loaded
        if not self.presets:
            self.presets = {
                'code': {
                    'max_length': 512,
                    'temperature': 0.2,
                    'top_k': 40,
                    'top_p': 0.9,
                    'do_sample': True
                },
                'creative': {
                    'max_length': 1024,
                    'temperature': 0.8,
                    'top_k': 50,
                    'top_p': 0.95,
                    'do_sample': True
                },
                'conversation': {
                    'max_length': 256,
                    'temperature': 0.7,
                    'top_k': 50,
                    'top_p': 0.9,
                    'do_sample': True
                },
                'accurate': {
                    'max_length': 256,
                    'temperature': 0.1,
                    'top_k': 10,
                    'do_sample': False
                }
            }
            
    def print_menu(self):
        """Print main menu."""
        print("\n" + "="*80)
        print("INDIA'S GPT MODEL - INTERACTIVE TESTER")
        print("="*80)
        print("\nAvailable Commands:")
        print("  1. generate   - Generate text from a prompt")
        print("  2. code       - Generate code")
        print("  3. chat       - Conversational mode")
        print("  4. presets    - List available presets")
        print("  5. config     - Show current configuration")
        print("  6. batch      - Batch generation")
        print("  7. help       - Show detailed help")
        print("  8. exit       - Exit the program")
        print("="*80)
        
    def generate_text(self):
        """Interactive text generation."""
        print("\n" + "-"*80)
        print("TEXT GENERATION")
        print("-"*80)
        
        prompt = input("\nEnter your prompt: ").strip()
        if not prompt:
            print("Empty prompt. Skipping.")
            return
            
        # Ask for preset or custom settings
        print("\nAvailable presets:", ", ".join(self.presets.keys()))
        preset = input("Choose preset (or 'custom'): ").strip().lower()
        
        if preset in self.presets:
            config_dict = self.presets[preset]
        elif preset == 'custom':
            try:
                max_length = int(input("Max length (default 256): ") or "256")
                temperature = float(input("Temperature (default 0.7): ") or "0.7")
                top_k = int(input("Top-k (default 50): ") or "50")
                top_p = float(input("Top-p (default 0.95): ") or "0.95")
                
                config_dict = {
                    'max_length': max_length,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'do_sample': True
                }
            except ValueError:
                print("Invalid input. Using default settings.")
                config_dict = self.presets['conversation']
        else:
            print(f"Unknown preset '{preset}'. Using conversation preset.")
            config_dict = self.presets['conversation']
            
        config = GenerationConfig(**config_dict)
        
        print("\nGenerating...\n")
        print("-"*80)
        
        generated = self.generator.generate(prompt, config)
        print(generated)
        
        print("-"*80)
        print(f"Length: {len(generated)} characters")
        
    def generate_code(self):
        """Interactive code generation."""
        print("\n" + "-"*80)
        print("CODE GENERATION")
        print("-"*80)
        
        print("\nExamples:")
        print("  - def fibonacci(n):")
        print("  - class LinkedList:")
        print("  - // JavaScript function to sort array")
        
        prompt = input("\nEnter code prompt: ").strip()
        if not prompt:
            print("Empty prompt. Skipping.")
            return
            
        config_dict = self.presets.get('code', {
            'max_length': 512,
            'temperature': 0.2,
            'top_k': 40,
            'do_sample': True
        })
        
        config = GenerationConfig(**config_dict)
        
        print("\nGenerating code...\n")
        print("-"*80)
        
        generated = self.generator.generate(prompt, config)
        print(generated)
        
        print("-"*80)
        
    def chat_mode(self):
        """Interactive chat mode."""
        print("\n" + "-"*80)
        print("CONVERSATIONAL MODE")
        print("-"*80)
        print("Type 'quit' to exit chat mode")
        print("-"*80)
        
        config_dict = self.presets.get('conversation', {
            'max_length': 256,
            'temperature': 0.7,
            'do_sample': True
        })
        config = GenerationConfig(**config_dict)
        
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting chat mode.")
                break
                
            if not user_input:
                continue
                
            # Build prompt with conversation history
            prompt = ""
            for turn in conversation_history[-5:]:  # Last 5 turns
                prompt += f"Human: {turn['user']}\nAssistant: {turn['assistant']}\n"
            prompt += f"Human: {user_input}\nAssistant:"
            
            response = self.generator.generate(prompt, config)
            
            # Extract just the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            print(f"\nAssistant: {response}")
            
            conversation_history.append({
                'user': user_input,
                'assistant': response
            })
            
    def list_presets(self):
        """List available generation presets."""
        print("\n" + "-"*80)
        print("AVAILABLE PRESETS")
        print("-"*80)
        
        for name, settings in self.presets.items():
            print(f"\n{name.upper()}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
                
    def show_config(self):
        """Show current model configuration."""
        print("\n" + "-"*80)
        print("MODEL CONFIGURATION")
        print("-"*80)
        
        config = self.model.config
        print(f"\nVocab size: {config.vocab_size}")
        print(f"Context length: {config.context_length}")
        print(f"Model dimension: {config.d_model}")
        print(f"Number of layers: {config.num_layers}")
        print(f"Number of heads: {config.num_heads}")
        print(f"Feed-forward dimension: {config.d_ff}")
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {num_params:,}")
        print(f"Device: {next(self.model.parameters()).device}")
        
    def batch_generation(self):
        """Batch text generation."""
        print("\n" + "-"*80)
        print("BATCH GENERATION")
        print("-"*80)
        
        print("\nEnter prompts (one per line). Enter empty line to finish:")
        
        prompts = []
        while True:
            prompt = input(f"Prompt {len(prompts)+1}: ").strip()
            if not prompt:
                break
            prompts.append(prompt)
            
        if not prompts:
            print("No prompts entered.")
            return
            
        preset = input("\nChoose preset (default: conversation): ").strip().lower() or 'conversation'
        config_dict = self.presets.get(preset, self.presets['conversation'])
        config = GenerationConfig(**config_dict)
        
        print("\nGenerating...\n")
        
        outputs = self.generator.generate_batch(prompts, config)
        
        for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
            print(f"\n{'-'*80}")
            print(f"Prompt {i}: {prompt}")
            print(f"Output: {output}")
            
        print(f"\n{'-'*80}")
        print(f"Generated {len(outputs)} outputs")
        
    def show_help(self):
        """Show detailed help."""
        print("\n" + "="*80)
        print("HELP - INTERACTIVE TESTER")
        print("="*80)
        
        print("""
This tool provides an interactive interface for testing India's GPT model.

COMMANDS:
  generate  - Generate text from a single prompt with various presets
  code      - Specialized mode for code generation
  chat      - Conversational mode with context memory
  presets   - View all available generation presets and their settings
  config    - Display model architecture and parameter count
  batch     - Generate multiple outputs from multiple prompts
  help      - Show this help message
  exit      - Exit the program

PRESETS:
  Each preset has optimized settings for specific tasks:
  - code: Low temperature for accurate code generation
  - creative: High temperature for creative writing
  - conversation: Balanced settings for natural dialogue
  - accurate: Greedy decoding for most likely output

USAGE TIPS:
  1. Use 'code' preset for programming tasks
  2. Use 'creative' for story writing or brainstorming
  3. Use 'conversation' for Q&A and chat
  4. Use 'accurate' when you need deterministic output
  
  You can also use 'custom' to set your own parameters.
        """)
        
    def run(self):
        """Run interactive loop."""
        self.print_menu()
        
        while True:
            try:
                print()
                choice = input("Enter command (or number): ").strip().lower()
                
                if choice in ['1', 'generate', 'gen', 'g']:
                    self.generate_text()
                elif choice in ['2', 'code', 'c']:
                    self.generate_code()
                elif choice in ['3', 'chat', 'conversation']:
                    self.chat_mode()
                elif choice in ['4', 'presets', 'p']:
                    self.list_presets()
                elif choice in ['5', 'config', 'cfg']:
                    self.show_config()
                elif choice in ['6', 'batch', 'b']:
                    self.batch_generation()
                elif choice in ['7', 'help', 'h', '?']:
                    self.show_help()
                elif choice in ['8', 'exit', 'quit', 'q']:
                    print("\nExiting. Goodbye!")
                    break
                else:
                    print(f"Unknown command: '{choice}'. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or type 'exit' to quit.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive testing tool for India's GPT model"
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large', 'xlarge'],
        help='Model size to load'
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
        help='Path to configuration file with presets'
    )
    
    args = parser.parse_args()
    
    print("Loading model...")
    if args.model_path:
        model = torch.load(args.model_path)
    else:
        model = create_gpt_model(args.model_size)
        
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Initializing tester...")
    tester = InteractiveTester(model, tokenizer, args.config)
    
    print("\nReady!")
    tester.run()


if __name__ == '__main__':
    main()
