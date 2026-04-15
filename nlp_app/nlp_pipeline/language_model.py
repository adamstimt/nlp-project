import nltk
from collections import Counter
from nltk.util import ngrams
import math

# Mathematical Core

def calculate_unigram_stats(tokens, vocab_size):
    """
    Calculates Unigram probabilities with Laplace Smoothing.
    P(w) = (Count(w) + 1) / (N + V)
    """
    N = len(tokens)
    counts = Counter(tokens)
    
    stats = []
    for word, count in counts.items():
        # Laplace Smoothing for Unigrams
        prob = (count + 1) / (N + vocab_size)
        
        stats.append({
            'word': word,
            'count': count,
            'prob': prob,  # Keep raw float for sorting
            'prob_fmt': f"{prob:.6f}" # String for display
        })
    
    # Sort by probability (descending)
    stats.sort(key=lambda x: x['prob'], reverse=True)
    return stats, counts

def calculate_bigram_stats(tokens, unigram_counts, vocab_size):
    """
    Calculates Bigram probabilities with Laplace Smoothing.
    P(w2 | w1) = (Count(w1, w2) + 1) / (Count(w1) + V)
    """
    bigrams = list(ngrams(tokens, 2))
    bigram_counts = Counter(bigrams)
    
    stats = []
    # We iterate through unique observed bigrams
    for bg, count in bigram_counts.items():
        w1, w2 = bg
        count_w1 = unigram_counts[w1]
        
        # Laplace Formula
        prob = (count + 1) / (count_w1 + vocab_size)
        
        stats.append({
            'w1': w1,
            'w2': w2,
            'count': count,
            'prob': prob,
            'prob_fmt': f"{prob:.6f}"
        })
        
    stats.sort(key=lambda x: x['prob'], reverse=True)
    return stats, bigram_counts

def score_sentence(sentence_tokens, bigram_counts, unigram_counts, vocab_size):
    """
    Calculates the total probability of a single sentence.
    P(S) = Product( P(wi | wi-1) )
    We use Log-Sum to avoid underflow: Log(P(S)) = Sum( Log(P(wi|wi-1)) )
    """
    if not sentence_tokens:
        return 0, -float('inf') # Log(0) is -inf

    log_prob_sum = 0
    
    # 1. First word probability (Unigram)
    # P(w1)
    w1 = sentence_tokens[0]
    count_w1 = unigram_counts[w1]
    N = sum(unigram_counts.values())
    p_w1 = (count_w1 + 1) / (N + vocab_size)
    log_prob_sum += math.log(p_w1)

    # 2. Subsequent words (Bigrams)
    # P(wi | wi-1)
    # If sentence is ["the", "cell", "divides"]
    # Bigrams: ("the", "cell"), ("cell", "divides")
    
    sentence_bigrams = list(ngrams(sentence_tokens, 2))
    
    for w_prev, w_curr in sentence_bigrams:
        count_w1_w2 = bigram_counts[(w_prev, w_curr)] # Returns 0 if unseen
        count_w1 = unigram_counts[w_prev] # Returns 0 if unseen
        
        # Laplace Formula
        p_bigram = (count_w1_w2 + 1) / (count_w1 + vocab_size)
        
        log_prob_sum += math.log(p_bigram)
        
    # Convert back to normal probability (EXP)
    # Note: This might be extremely small (e.g. 1.2e-20)
    try:
        total_prob = math.exp(log_prob_sum)
    except OverflowError:
        total_prob = 0.0
        
    return total_prob, log_prob_sum

# Pipeline Entry Point

def run_language_model(tokenized_sentences):
    """
    Main function called by views.py
    """
    # 1. Flatten all text for Global Counts
    all_tokens = [t for sent in tokenized_sentences for t in sent]
    N = len(all_tokens)
    
    # 2. Vocabulary (V)
    # Based on the prompt: V is size of vocab AFTER punctuation removal
    vocab = set(all_tokens)
    V = len(vocab)
    
    if N == 0:
        return {}

    # 3. Calculate Global Stats
    unigram_list, unigram_counts = calculate_unigram_stats(all_tokens, V)
    bigram_list, bigram_counts = calculate_bigram_stats(all_tokens, unigram_counts, V)
    
    # 4. Calculate Sentence Probabilities
    # We score each sentence against the model built from the whole text
    sentence_scores = []
    total_log_prob_corpus = 0
    
    for sent in tokenized_sentences:
        prob, log_prob = score_sentence(sent, bigram_counts, unigram_counts, V)
        total_log_prob_corpus += log_prob
        
        sentence_scores.append({
            'text': " ".join(sent),
            'prob': f"{prob:.4e}", # Scientific notation (e.g., 1.23e-05)
            'log_prob': round(log_prob, 4)
        })

    # 5. Calculate Perplexity
    # PP(W) = exp( -1/N * Sum(Log(P)) )
    # Note: N here is total words in corpus
    perplexity = math.exp(-1/N * total_log_prob_corpus)

    return {
        'vocab_size': V,
        'total_tokens': N,
        'perplexity': round(perplexity, 2),
        'unigrams': unigram_list, # List of dicts
        'bigrams': bigram_list,   # List of dicts
        'sentence_probs': sentence_scores # List of dicts
    }