"""
Example 1: Basic text generation with India's GPT model.
Demonstrates how to generate text using different configurations.
"""

from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("EXAMPLE 1: BASIC TEXT GENERATION")
    print("="*80)
    
    # Load model and tokenizer
    print("\n1. Loading model...")
    model = create_gpt_model('small')  # Use small model for quick demo
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Create generator
    print("2. Creating text generator...")
    generator = TextGenerator(model, tokenizer)
    
    # Example 1: Simple completion
    print("\n" + "-"*80)
    print("Example 1: Simple Text Completion")
    print("-"*80)
    
    prompt1 = "Artificial intelligence is"
    config1 = GenerationConfig(
        max_length=100,
        temperature=0.7,
        do_sample=True
    )
    
    result1 = generator.generate(prompt1, config1)
    print(f"\nPrompt: {prompt1}")
    print(f"Generated: {result1}")
    
    # Example 2: Creative writing
    print("\n" + "-"*80)
    print("Example 2: Creative Writing")
    print("-"*80)
    
    prompt2 = "Once upon a time in a distant land,"
    config2 = GenerationConfig(
        max_length=200,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
    
    result2 = generator.generate(prompt2, config2)
    print(f"\nPrompt: {prompt2}")
    print(f"Generated: {result2}")
    
    # Example 3: Factual completion (low temperature)
    print("\n" + "-"*80)
    print("Example 3: Factual Completion")
    print("-"*80)
    
    prompt3 = "Python is a programming language that"
    config3 = GenerationConfig(
        max_length=100,
        temperature=0.1,
        do_sample=False
    )
    
    result3 = generator.generate(prompt3, config3)
    print(f"\nPrompt: {prompt3}")
    print(f"Generated: {result3}")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
