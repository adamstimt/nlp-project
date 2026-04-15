import nltk
from collections import defaultdict
import math

# Data Setup
# --------------------------
# We use the NLTK Treebank corpus to learn the probabilities (Training Phase).
# In a real bio-app, you would swap this for a biomedical corpus (like GENIA).
try:
    nltk.data.find('corpora/treebank')
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('treebank')
    nltk.download('universal_tagset')

class HMMTagger:
    def __init__(self):
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.tags = set()
        self.trained = False

    def train(self):
        """
        Trains the HMM using the NLTK Treebank corpus.
        Calculates Log-Probabilities for Transitions and Emissions.
        """
        # Load tagged sentences using the 'universal' tagset (NOUN, VERB, ADJ, etc.)
        tagged_sentences = nltk.corpus.treebank.tagged_sents(tagset='universal')

        # Frequency counters
        # transition_counts[prev_tag][curr_tag]
        transition_counts = defaultdict(lambda: defaultdict(int))
        # emission_counts[tag][word]
        emission_counts = defaultdict(lambda: defaultdict(int))
        
        # Count occurrences
        for sentence in tagged_sentences:
            prev_tag = "<START>"
            for word, tag in sentence:
                word = word.lower()
                self.tags.add(tag)
                
                transition_counts[prev_tag][tag] += 1
                emission_counts[tag][word] += 1
                
                prev_tag = tag

        # Convert counts to Log Probabilities to prevent underflow
        # Formula: Log(Count(A->B) / Count(A))
        
        # 1. Transition Probabilities
        for prev_tag, next_tag_map in transition_counts.items():
            total_transitions = sum(next_tag_map.values())
            for next_tag, count in next_tag_map.items():
                self.transitions[(prev_tag, next_tag)] = math.log(count / total_transitions)

        # 2. Emission Probabilities
        for tag, word_map in emission_counts.items():
            total_emissions = sum(word_map.values())
            for word, count in word_map.items():
                self.emissions[(tag, word)] = math.log(count / total_emissions)
        
        self.trained = True

    def get_emission_prob(self, tag, word):
        """
        Returns log probability. Handles unknown words with a penalty (Smoothing).
        """
        key = (tag, word.lower())
        if key in self.emissions:
            return self.emissions[key]
        else:
            # If word is unknown, return a very low probability (penalty)
            return -15.0 

    def get_transition_prob(self, prev_tag, curr_tag):
        """
        Returns log probability for tag transition.
        """
        key = (prev_tag, curr_tag)
        if key in self.transitions:
            return self.transitions[key]
        else:
            return -15.0 

    def viterbi(self, tokens):
        """
        The Viterbi Algorithm: Finds the best tag sequence for a list of tokens.
        """
        if not tokens:
            return []
            
        # Initialize tables
        # viterbi[i][tag] = max probability of being at 'tag' at step 'i'
        viterbi = []
        backpointer = []

        # --- Step 1: Initialization ---
        first_viterbi = {}
        first_backpointer = {}
        
        for tag in self.tags:
            # Prob = Transition(<START> -> tag) + Emission(tag -> first_word)
            prob = self.get_transition_prob("<START>", tag) + self.get_emission_prob(tag, tokens[0])
            first_viterbi[tag] = prob
            first_backpointer[tag] = "<START>"
        
        viterbi.append(first_viterbi)
        backpointer.append(first_backpointer)

        # --- Step 2: Recursion ---
        for t in range(1, len(tokens)):
            this_viterbi = {}
            this_backpointer = {}
            
            for curr_tag in self.tags:
                best_prev_prob = -float('inf')
                best_prev_tag = None
                
                emission_p = self.get_emission_prob(curr_tag, tokens[t])
                
                for prev_tag in self.tags:
                    # Calculate probability for this path
                    prob = viterbi[t-1][prev_tag] + \
                           self.get_transition_prob(prev_tag, curr_tag) + \
                           emission_p
                    
                    if prob > best_prev_prob:
                        best_prev_prob = prob
                        best_prev_tag = prev_tag
                
                this_viterbi[curr_tag] = best_prev_prob
                this_backpointer[curr_tag] = best_prev_tag
            
            viterbi.append(this_viterbi)
            backpointer.append(this_backpointer)

        # --- Step 3: Termination ---
        best_final_prob = -float('inf')
        best_last_tag = None
        for tag in self.tags:
            if viterbi[-1][tag] > best_final_prob:
                best_final_prob = viterbi[-1][tag]
                best_last_tag = tag

        # --- Step 4: Backtracking ---
        best_path = [best_last_tag]
        current_tag = best_last_tag
        
        # Traverse backwards from the end
        for t in range(len(tokens) - 1, 0, -1):
            prev_tag = backpointer[t][current_tag]
            best_path.insert(0, prev_tag)
            current_tag = prev_tag

        return list(zip(tokens, best_path))

# Pipeline Entry Point

# Create a single instance so we don't retrain on every request
hmm_tagger = HMMTagger()

def run_pos_tagging(tokenized_sentences):
    """
    Main function to call from Django.
    Args:
        tokenized_sentences: List of list of strings (output from preprocessing)
    Returns:
        List of list of tuples: [[('Word', 'TAG'), ...], ...]
    """
    if not hmm_tagger.trained:
        hmm_tagger.train()
    
    tagged_results = []
    for sentence_tokens in tokenized_sentences:
        tagged = hmm_tagger.viterbi(sentence_tokens)
        tagged_results.append(tagged)
        
    return tagged_results