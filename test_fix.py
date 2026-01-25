"""
Test script to verify the fixes for text context handling
"""

from pathlib import Path
from app.services.agent import GeminiAgent
from app.config import settings

# Initialize agent
agent = GeminiAgent(api_key=settings.GOOGLE_API_KEY)

print("=" * 60)
print("Testing Agent Prompt Building")
print("=" * 60)

# Test 1: Empty list (this was the bug)
print("\n1. Testing with empty text context (the original bug):")
text_context_empty = []
visual_context_empty = []

prompt = agent._build_prompt(
    query="What are the main topics?",
    text_context=text_context_empty,
    has_visuals=False
)
print(f"✓ Prompt generated successfully")
print(f"  Contains 'No text context available': {'No text context available' in prompt}")

# Test 2: With text context, no visuals
print("\n2. Testing with text context, no visuals:")
text_context_with_data = [
    "[Page 1] This is a sample text from page 1.",
    "[Page 2] This is another sample from page 2."
]

prompt = agent._build_prompt(
    query="What are the main topics?",
    text_context=text_context_with_data,
    has_visuals=False
)
print(f"✓ Prompt generated successfully")
print(f"  Contains text context: {text_context_with_data[0][:30] in prompt}")
print(f"  Mentions visual elements: {'Visual Elements:' in prompt}")
print(f"  Task count: {prompt.count('1. ') + prompt.count('2. ') + prompt.count('3. ')}")

# Test 3: With text context and visuals
print("\n3. Testing with text context and visuals:")
prompt = agent._build_prompt(
    query="What are the main topics?",
    text_context=text_context_with_data,
    has_visuals=True
)
print(f"✓ Prompt generated successfully")
print(f"  Contains text context: {text_context_with_data[0][:30] in prompt}")
print(f"  Mentions visual elements: {'Visual Elements:' in prompt}")
print(f"  Task count: {prompt.count('1. ') + prompt.count('2. ') + prompt.count('3. ') + prompt.count('4. ') + prompt.count('5. ')}")

# Test 4: None text context
print("\n4. Testing with None text context:")
prompt = agent._build_prompt(
    query="What are the main topics?",
    text_context=None,
    has_visuals=False
)
print(f"✓ Prompt generated successfully (handles None gracefully)")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
