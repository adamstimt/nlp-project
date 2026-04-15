import os
from django.shortcuts import render
from django.conf import settings
from .forms import NLPInputForm

# Import all custom NLP modules
from .nlp_pipeline import (
    preprocessing,
    pos_tagging,
    ner,
    parsing,
    language_model,
    topic_modelling,
    stats
)

def index(request):
    """
    Renders the homepage with the input form.
    """
    form = NLPInputForm()
    return render(request, 'nlp_app/index.html', {'form': form})

def analyze(request):
    """
    Main processing view:
    1. Handle Form/File Upload
    2. Run full NLP Pipeline
    3. Save results to Session for the Stats page
    4. Return results to template
    """
    results = {}
    
    if request.method == 'POST':
        form = NLPInputForm(request.POST, request.FILES)
        
        if form.is_valid():
            text_input = ""
            is_xml = False
            
            # --- 1. Input Handling ---
            
            # Check for XML File Upload
            if form.cleaned_data['xml_file']:
                xml_file = form.cleaned_data['xml_file']
                is_xml = True
                
                # Ensure the raw data directory exists
                if not os.path.exists(settings.RAW_DATA_DIR):
                    os.makedirs(settings.RAW_DATA_DIR)
                
                save_path = os.path.join(settings.RAW_DATA_DIR, xml_file.name)
                
                # Write file to disk
                with open(save_path, 'wb+') as destination:
                    for chunk in xml_file.chunks():
                        destination.write(chunk)
                        
                # Read content for processing
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        text_input = f.read()
                except UnicodeDecodeError:
                    # Fallback for different encodings
                    with open(save_path, 'r', encoding='latin-1') as f:
                        text_input = f.read()

            # Check for Raw Text (Fallback)
            elif form.cleaned_data['raw_text']:
                text_input = form.cleaned_data['raw_text']
            
            
            # --- 2. NLP Pipeline Execution ---
            
            if text_input:
                # Step 1: Preprocessing (Cleaning, Punctuation Removal & Tokenization)
                preproc_data = preprocessing.run_preprocessing(text_input, is_xml=is_xml)
                
                # Shortcuts for subsequent steps
                sentences = preproc_data['sentences']
                tokens_list = preproc_data['tokenized_sentences']
                
                # Step 2: POS Tagging (HMM Based)
                pos_data = pos_tagging.run_pos_tagging(tokens_list)
                
                # Step 3: Named Entity Recognition (NER)
                ner_data = ner.run_ner(pos_data)
                
                # Step 4: Syntactic Parsing (Directory-style Text Trees)
                parse_data = parsing.run_parsing(ner_data)
                
                # Step 5: Language Modeling (Laplace Smoothing & Perplexity)
                lm_data = language_model.run_language_model(tokens_list)
                
                # Step 6: Topic Modeling (LDA)
                topic_data = topic_modelling.run_topic_modelling(sentences)
                
                # Step 7: General Statistics (Diversity, Lengths)
                stats_data = stats.calculate_stats(tokens_list)
                
                # --- 3. Final Result Packaging ---
                results = {
                    'original_text': preproc_data['raw_text'],
                    'pos_data': pos_data,
                    'ner_data': ner_data,
                    'parse_data': parse_data,
                    'lm_data': lm_data,
                    'topic_data': topic_data,
                    'stats_data': stats_data,
                }

                # --- 4. Session Linking ---
                # Save results to session so stats.html can access them after redirect or navigation
                request.session['last_analysis'] = results

    else:
        form = NLPInputForm()

    return render(request, 'nlp_app/analyze.html', {'form': form, 'results': results})

def stats_view(request):
    """
    Retrieves the latest analysis results from the session 
    to display global Chart.js visualizations.
    """
    # Get results from session (stored during the last successful analyze POST)
    results = request.session.get('last_analysis', None)
    
    return render(request, 'nlp_app/stats.html', {'results': results})