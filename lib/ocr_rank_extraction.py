"""Extract rank number from OCR tokens in a row"""
import re


def extract_rank_from_tokens(tokens: list) -> int | None:
    """Extract the rank number from OCR tokens.
    
    The rank is typically:
    - The leftmost small number (x < 150)
    - A number between 1-15 (for single rally) or 1-10 (for totals)
    
    Args:
        tokens: List of token dicts with 'text', 'conf', 'x', 'y'
    
    Returns:
        Extracted rank number, or None if not found
    """
    if not tokens:
        return None
    
    # Sort by x position (left to right)
    sorted_tokens = sorted(tokens, key=lambda t: t.get("x", 0))
    
    # Look for a small number on the left
    for token in sorted_tokens[:3]:  # Check only first 3 tokens
        text = token.get("text", "").strip()
        # Must be a number 1-15
        if re.fullmatch(r"\d+", text):
            try:
                rank_val = int(text)
                if 1 <= rank_val <= 15:
                    return rank_val
            except ValueError:
                pass
    
    return None
