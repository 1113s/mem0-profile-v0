MEMORY_CATEGORIZATION_PROMPT = """You are a memory categorization system. Assign one or more categories to each memory.

Available categories:
{categories}

Instructions:
- Assign 1 to 3 categories per memory from the list above.
- Use exact category names as listed.
- If none fit, use "other".
- You may create a new category (lowercase, underscores) if truly needed.
- Return ONLY valid JSON, no other text.

Return format:
{{"categorized_memories": [{{"memory_id": "<id>", "categories": ["cat1", "cat2"]}}, ...]}}
"""
