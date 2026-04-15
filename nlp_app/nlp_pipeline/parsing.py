import nltk

# Grammar Definition (Updated for Universal Tags)
#--------------
# We updated the rules to look for 'NOUN', 'VERB', 'PRON', etc.
CHUNK_GRAMMAR = r"""
  NP: {<DET|PRON>?<ADJ>*<NOUN|PROPN|X>+} # Noun Phrase: (Det/Pron) + (Adj) + Noun
  NP: {<PRON>}                           # Noun Phrase: just a Pronoun (e.g., "He")
  PP: {<ADP><NP>}                        # Prepositional Phrase: ADP + NP
  VP: {<VERB><NP|PP>*}                   # Verb Phrase: Verb + Object
  CLAUSE: {<NP><VP>}                     # Full Clause
"""

# Helper: Exact Directory-Style Tree Printer

def tree_to_text(node, prefix="", is_last=True, is_root=True):
    """
    Generates a directory-style tree string.
    Splits (Word, Tag) tuples so the Tag becomes a parent node.
    """
    lines = []
    
    # 1. Determine Label & Type
    if isinstance(node, nltk.Tree):
        label = node.label()
        is_leaf_tuple = False
    elif isinstance(node, tuple) and len(node) == 2:
        # This is a leaf node from NLTK: (word, tag)
        word, tag = node
        label = tag # The Tag becomes the node label
        is_leaf_tuple = True
    else:
        label = str(node)
        is_leaf_tuple = False

    # 2. Print Current Node
    if is_root:
        lines.append(label)
    else:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{label}")

    # 3. Handle Children
    if isinstance(node, nltk.Tree):
        child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")
        children = list(node)
        count = len(children)
        
        for i, child in enumerate(children):
            is_last_child = (i == count - 1)
            lines.extend(tree_to_text(child, child_prefix, is_last_child, is_root=False))
            
    elif is_leaf_tuple:
        word = node[0]
        child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")
        connector = "└── "
        lines.append(f"{child_prefix}{connector}{word}")

    return lines

# Core Functions

def run_parsing(tagged_sentences):
    cp = nltk.RegexpParser(CHUNK_GRAMMAR)
    
    parsed_results = []

    for sentence in tagged_sentences:
        try:
            # Filter NER tags back to (word, pos)
            clean_input = [(w, t) for w, t, ner in sentence]
            
            # Generate the Tree
            tree = cp.parse(clean_input)
            
            if tree.label() != 'S':
                tree.set_label('S')
            
            # Generate the text lines
            tree_lines = tree_to_text(tree)
            vertical_tree_string = "\n".join(tree_lines)
            
            parsed_results.append({
                'tree_string': vertical_tree_string
            })
            
        except Exception as e:
            parsed_results.append({
                'tree_string': f"(Error parsing sentence: {e})"
            })

    return parsed_results