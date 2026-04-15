import xml.etree.ElementTree as ET
import re
import nltk
import string

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


# Core Functions

def parse_pubmed_xml(xml_content):
    """Parses PubMed XML to extract AbstractText."""
    try:
        root = ET.fromstring(xml_content)
        abstract_parts = []
        for elem in root.iter('AbstractText'):
            if elem.text:
                abstract_parts.append(elem.text.strip())
        return " ".join(abstract_parts)
    except ET.ParseError:
        return ""


def clean_text(text):
    """
    Removes punctuation and lowercases a specific string.
    """
    if not text:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# Pipeline Entry Point

def run_preprocessing(input_data, is_xml=False):
    raw_text = ""

    # 1. Extract Text
    if is_xml:
        raw_text = parse_pubmed_xml(input_data)
    else:
        raw_text = input_data

    # 2. Segment Sentences 
    # We use the raw text to find sentence boundaries (periods, etc.)
    if raw_text:
        raw_sentences = nltk.sent_tokenize(raw_text)
    else:
        raw_sentences = []

    # 3. Clean Each Sentence Individually
    # Now we remove punctuation from each sentence to satisfy the Language Model requirements
    cleaned_sentences = []
    tokenized_sentences = []

    for sent in raw_sentences:
        cleaned = clean_text(sent)
        if cleaned: # Only keep non-empty sentences
            cleaned_sentences.append(cleaned)
            # 4. Tokenize
            tokenized_sentences.append(nltk.word_tokenize(cleaned))

    return {
        'raw_text': raw_text, # Return original for display
        'sentences': cleaned_sentences, # List of strings (no punct)
        'tokenized_sentences': tokenized_sentences # List of lists
    }