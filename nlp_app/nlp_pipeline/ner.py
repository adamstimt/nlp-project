import nltk
import re

# ==========================================
# 0. NLTK Resource Check (AUTO-DOWNLOAD)
# ==========================================
# We define a helper to download resources safely
def download_if_missing(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        print(f"Downloading missing NLTK resource: {resource_name}")
        nltk.download(resource_name.split('/')[-1]) # Download the package name

download_if_missing('chunkers/maxent_ne_chunker_tab')
download_if_missing('corpora/words')
download_if_missing('corpora/names') # <--- NEW: For Person names library

# ==========================================
# 1. Knowledge Base (Custom Bio-Dictionary)
# ==========================================

# 1a. Load Person Names from NLTK Library (Result: ~8,000 names)
# We use a set() for instant speed lookup (O(1)) instead of a list
try:
    MALE_NAMES = set(name.lower() for name in nltk.corpus.names.words('male.txt'))
    FEMALE_NAMES = set(name.lower() for name in nltk.corpus.names.words('female.txt'))
    ALL_NAMES = MALE_NAMES.union(FEMALE_NAMES)
except LookupError:
    ALL_NAMES = {"john", "jane", "doe", "abdellah"} # Fallback if library fails

BIO_ENTITIES = {
    "DISEASE": {
        "cancer", "tumor", "diabetes", "hiv", "aids", "leukemia", 
        "melanoma", "carcinoma", "lymphoma", "infection", "hypertension",
        "alzheimer", "parkinson", "covid-19", "sars-cov-2"
    },
    "CHEMICAL": {
        "oxygen", "glucose", "insulin", "dopamine", "serotonin", 
        "cisplatin", "paclitaxel", "hydrogen", "nitrogen", "calcium"
    },
    "GENE_PROTEIN": {
        "p53", "brca1", "brca2", "egfr", "her2", "cd4", "cd8", "myc", "ras",
        "dna", "rna", "mrna"
    },
    "LOCATION": {
        "laboratory", "hospital", "clinic", "university", "center", "institute",
        "department", "ward", "room", "algiers", "paris", "london", "usa", "uk"
    },
    "ORGANIZATION": {
        "who", "cdc", "nih", "fda", "inserm", "pasteur", "pfizer", "moderna",
        "astrazeneca", "cnrs", "google", "facebook", "microsoft"
    },
    "DATE": {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "today", "yesterday", "tomorrow"
    },
    "PERSON": ALL_NAMES # <--- Loaded from Library
}

# ==========================================
# 2. Core Functions
# ==========================================

def get_custom_entity(word):
    """
    Checks if a word exists in our custom bio-dictionary.
    Returns the entity label (e.g., 'DISEASE') or None.
    """
    word_lower = word.lower()
    
    for label, keywords in BIO_ENTITIES.items():
        if word_lower in keywords:
            return label
            
    # Fallback Rules (Regex)
    if word_lower.endswith("ase") and len(word) > 4:
        return "ENZYME"
    if re.search(r'^[A-Z]+-[0-9]+$', word): # Matches patterns like IL-6, CD-19
        return "GENE_PROTEIN"
    if re.search(r'^\d{4}$', word): # Matches years like 1990, 2024
        return "DATE"
        
    return None

def tree_to_bio(tree):
    """
    Converts an NLTK chunk tree into a list of (word, pos, bio_tag).
    Example: [('Steve', 'NNP', 'B-PERSON'), ('Jobs', 'NNP', 'I-PERSON')]
    """
    bio_tags = []
    
    for child in tree:
        if isinstance(child, nltk.Tree):
            # 1. NLTK detected an entity automatically (Standard PER/ORG/GPE)
            label = child.label() 
            
            # Map NLTK labels to our specific format if needed
            if label == "GPE": label = "LOCATION"
            
            for i, (word, pos) in enumerate(child):
                prefix = "B-" if i == 0 else "I-"
                bio_tags.append((word, pos, f"{prefix}{label}"))
                
        else:
            # 2. NLTK missed it (Outside) -> Check our Custom Dictionary
            word, pos = child
            custom_tag = get_custom_entity(word)
            
            if custom_tag:
                bio_tags.append((word, pos, f"B-{custom_tag}"))
            else:
                bio_tags.append((word, pos, "O"))
                
    return bio_tags

def run_ner(sentences_with_tags):
    """
    Main NER pipeline function.
    Combines NLTK's pre-trained model with custom dictionary lookups.
    """
    results = []

    for sentence in sentences_with_tags:
        # 1. Use NLTK's pre-trained NER chunker
        # This builds a Tree structure identifying standard entities
        chunked_tree = nltk.ne_chunk(sentence)
        
        # 2. Convert to BIO format and apply custom dictionary
        bio_sentence = tree_to_bio(chunked_tree)
        
        results.append(bio_sentence)
        
    return results