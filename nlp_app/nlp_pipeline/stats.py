# Core Functions

def calculate_stats(tokenized_sentences):
    """
    Computes basic statistics for the input text.
    
    Args:
        tokenized_sentences: List of lists of strings.
    
    Returns:
        Dictionary containing calculated metrics.
    """
    if not tokenized_sentences:
        return {
            'total_sentences': 0,
            'total_tokens': 0,
            'avg_sentence_length': 0,
            'lexical_diversity': 0
        }

    # Flatten list to get all tokens
    all_tokens = [t for s in tokenized_sentences for t in s]
    
    total_sentences = len(tokenized_sentences)
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    
    # Average sentence length (Tokens per sentence)
    avg_len = total_tokens / total_sentences if total_sentences > 0 else 0
    
    # Lexical Diversity (Type-Token Ratio)
    # Higher score = richer vocabulary. Lower score = repetitive text.
    diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

    return {
        'total_sentences': total_sentences,
        'total_tokens': total_tokens,
        'avg_sentence_length': round(avg_len, 2),
        'lexical_diversity': round(diversity, 2)  # e.g., 0.45
    }