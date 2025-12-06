"""
Example 2: Code generation with India's GPT model.
Demonstrates how to generate code using optimized settings.
"""

from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("EXAMPLE 2: CODE GENERATION")
    print("="*80)
    
    # Load model and tokenizer
    print("\n1. Loading model...")
    model = create_gpt_model('base')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Create generator with code-optimized settings
    print("2. Creating text generator...")
    generator = TextGenerator(model, tokenizer)
    
    # Configuration optimized for code generation
    code_config = GenerationConfig(
        max_length=512,
        temperature=0.2,  # Low temperature for accurate code
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Example 1: Python function
    print("\n" + "-"*80)
    print("Example 1: Complete Python Function")
    print("-"*80)
    
    prompt1 = '''def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion."""
    if n <= 1:'''
    
    result1 = generator.generate(prompt1, code_config)
    print(f"Generated:\n{result1}")
    
    # Example 2: Class definition
    print("\n" + "-"*80)
    print("Example 2: Complete Python Class")
    print("-"*80)
    
    prompt2 = '''class BinarySearchTree:
    """A binary search tree implementation."""
    def __init__(self):
        self.root = None
    
    def insert(self, value):'''
    
    result2 = generator.generate(prompt2, code_config)
    print(f"Generated:\n{result2}")
    
    # Example 3: Algorithm implementation
    print("\n" + "-"*80)
    print("Example 3: Sorting Algorithm")
    print("-"*80)
    
    prompt3 = '''def quicksort(arr):
    """
    Implement quicksort algorithm.
    Time complexity: O(n log n) average case
    """
    if len(arr) <= 1:'''
    
    result3 = generator.generate(prompt3, code_config)
    print(f"Generated:\n{result3}")
    
    # Example 4: JavaScript code
    print("\n" + "-"*80)
    print("Example 4: JavaScript Function")
    print("-"*80)
    
    prompt4 = '''// Function to debounce function calls
function debounce(func, delay) {
    let timeout;
    return function() {'''
    
    result4 = generator.generate(prompt4, code_config)
    print(f"Generated:\n{result4}")
    
    print("\n" + "="*80)
    print("CODE GENERATION EXAMPLES COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
