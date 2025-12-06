"""
Example 3: Conversational AI with India's GPT model.
Demonstrates how to create realistic conversations.
"""

from models.gpt_model import create_gpt_model
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer


def main():
    print("="*80)
    print("EXAMPLE 3: CONVERSATIONAL AI")
    print("="*80)
    
    # Load model and tokenizer
    print("\n1. Loading model...")
    model = create_gpt_model('base')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Create generator with conversation-optimized settings
    print("2. Creating text generator...")
    generator = TextGenerator(model, tokenizer)
    
    # Configuration optimized for natural conversation
    conversation_config = GenerationConfig(
        max_length=256,
        temperature=0.7,  # Moderate temperature for natural variation
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True
    )
    
    # Example 1: Question answering
    print("\n" + "-"*80)
    print("Example 1: Question Answering")
    print("-"*80)
    
    prompt1 = """Human: What is machine learning and how does it work?
Assistant:"""
    
    result1 = generator.generate(prompt1, conversation_config)
    print(f"{result1}")
    
    # Example 2: Multi-turn conversation
    print("\n" + "-"*80)
    print("Example 2: Multi-turn Conversation")
    print("-"*80)
    
    conversation_history = [
        ("What are neural networks?", "Neural networks are computational models inspired by the human brain..."),
        ("How are they used in AI?", "Neural networks are fundamental to modern AI systems...")
    ]
    
    # Build conversation context
    context = ""
    for user_msg, assistant_msg in conversation_history:
        context += f"Human: {user_msg}\nAssistant: {assistant_msg}\n"
    
    # New question with context
    new_question = "Can you explain backpropagation?"
    prompt2 = context + f"Human: {new_question}\nAssistant:"
    
    result2 = generator.generate(prompt2, conversation_config)
    print(f"Context: [Previous conversation about neural networks]")
    print(f"Human: {new_question}")
    print(f"Assistant: {result2.split('Assistant:')[-1].strip()}")
    
    # Example 3: Helpful assistant
    print("\n" + "-"*80)
    print("Example 3: Helpful Assistant")
    print("-"*80)
    
    prompt3 = """Human: I'm learning Python. What are the best practices for writing clean code?
Assistant:"""
    
    result3 = generator.generate(prompt3, conversation_config)
    print(f"{result3}")
    
    # Example 4: Explanation with examples
    print("\n" + "-"*80)
    print("Example 4: Detailed Explanation")
    print("-"*80)
    
    prompt4 = """Human: Can you explain what recursion is with a simple example?
Assistant:"""
    
    result4 = generator.generate(prompt4, conversation_config)
    print(f"{result4}")
    
    print("\n" + "="*80)
    print("CONVERSATIONAL AI EXAMPLES COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
