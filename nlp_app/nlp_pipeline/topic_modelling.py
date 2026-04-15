from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Core Functions

def run_topic_modelling(sentences, n_topics=3, n_words=5):
    """
    Performs LDA Topic Modeling on the provided sentences.
    
    Args:
        sentences: List of sentence strings (output from preprocessing['sentences']).
        n_topics: Number of topics to find.
        n_words: Number of top words to identify per topic.
        
    Returns:
        Dictionary mapping Topic ID -> List of top words.
        Example: {0: ['cell', 'growth'], 1: ['clinical', 'data']}
    """
    # 1. Validation: We need enough data to model topics
    # If text is too short (< 3 sentences), topic modeling isn't useful.
    if not sentences or len(sentences) < 3:
        return {"Error": ["Not enough data for topic modeling (need 3+ sentences)."]}

    try:
        # 2. Vectorization (Bag of Words)
        # Convert sentences to a matrix of token counts
        # stop_words='english' removes common noise like 'the', 'and'
        tf_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        tf = tf_vectorizer.fit_transform(sentences)
        
        # 3. Fit LDA Model
        lda = LatentDirichletAllocation(
            n_components=n_topics, 
            max_iter=10, 
            learning_method='online', 
            random_state=42
        )
        lda.fit(tf)

        # 4. Extract Top Words for each Topic
        feature_names = tf_vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            # Sort words by weight (highest first) and take top 'n_words'
            top_features_ind = topic.argsort()[:-n_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            
            topics[f"Topic {topic_idx + 1}"] = top_features

        return topics

    except ValueError:
        # Handles cases where sentences contain only stop words
        return {"Error": ["Data contained only stopwords or was invalid."]}
    except Exception as e:
        return {"Error": [str(e)]}