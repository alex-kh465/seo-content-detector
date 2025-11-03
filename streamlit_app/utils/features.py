import pandas as pd
import numpy as np
import re
import textstat
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Clean and preprocess text.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text without stopwords
    """
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)


def extract_features(text):
    """
    Extract NLP features from text.
    
    Args:
        text: Input text string
        
    Returns:
        dict with features
    """
    if not text or not isinstance(text, str):
        return {
            'word_count': 0,
            'sentence_count': 0,
            'flesch_reading_ease': 0.0
        }
    
    # Initialize with zeros
    word_count = 0
    sentence_count = 0
    flesch_score = 0.0
    
    try:
        # Word count
        word_count = len(text.split())
        
        # Sentence count - try multiple tokenizers
        try:
            sentences = sent_tokenize(text)
            sentence_count = len(sentences)
        except Exception as e:
            # Fallback: count by periods
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            if sentence_count == 0:
                sentence_count = 1
        
        # Flesch reading ease
        try:
            if word_count > 0:
                flesch_score = textstat.flesch_reading_ease(text)
        except Exception as e:
            # Fallback calculation if textstat fails
            flesch_score = 0.0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'flesch_reading_ease': round(flesch_score, 2)
        }
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        # Return what we calculated so far
        return {
            'word_count': word_count,
            'sentence_count': max(sentence_count, 1) if word_count > 0 else 0,
            'flesch_reading_ease': round(flesch_score, 2)
        }


def extract_top_keywords(texts, top_n=5):
    """
    Extract top keywords using TF-IDF.
    
    Args:
        texts: List of text strings
        top_n: Number of top keywords to extract
        
    Returns:
        List of keyword strings for each text
    """
    if not texts or len(texts) == 0:
        return []
    
    # Preprocess texts
    processed_texts = [remove_stopwords(preprocess_text(text)) for text in texts]
    
    # Remove empty texts
    valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
    valid_texts = [processed_texts[i] for i in valid_indices]
    
    if not valid_texts:
        return ["" for _ in texts]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        keywords_list = [""] * len(texts)
        
        for idx, valid_idx in enumerate(valid_indices):
            # Get top keywords for this document
            tfidf_scores = tfidf_matrix[idx].toarray()[0]
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            keywords_list[valid_idx] = ", ".join(top_keywords)
        
        return keywords_list
    
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return ["" for _ in texts]


def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings using SentenceTransformer.
    
    Args:
        texts: List of text strings
        model_name: Name of the sentence transformer model
        
    Returns:
        numpy array of embeddings
    """
    print(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings...")
    # Convert texts to strings and handle NaN/None values defensively. Ensure
    # every item passed to the encoder is a Python str (avoids float/NaN inputs).
    valid_texts = ["" if pd.isna(text) else str(text) for text in texts]

    embeddings = model.encode(valid_texts, show_progress_bar=True, batch_size=32)
    
    return embeddings


def extract_features_from_dataset(input_csv_path, output_csv_path):
    """
    Extract features from extracted content and save to CSV.
    
    Args:
        input_csv_path: Path to CSV with extracted content
        output_csv_path: Path to save features CSV
        
    Returns:
        DataFrame with features
    """
    print(f"Loading extracted content from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    # Defensive fix: coerce body_text to string and replace NaNs with empty strings.
    # This avoids passing float/NaN values into the sentence transformer which
    # expects string inputs (the originally observed error was a float being
    # passed through to the tokenizer).
    if 'body_text' in df.columns:
        df['body_text'] = df['body_text'].fillna('').astype(str)

    if 'body_text' not in df.columns:
        raise ValueError("Input CSV must contain 'body_text' column")
    
    print(f"Extracting features from {len(df)} pages...")
    
    # Extract basic features
    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        features = extract_features(row['body_text'])
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Add URL
    features_df['url'] = df['url']
    
    # Extract top keywords
    print("Extracting keywords...")
    keywords = extract_top_keywords(df['body_text'].tolist())
    features_df['top_keywords'] = keywords
    
    # Generate embeddings
    embeddings = generate_embeddings(df['body_text'].tolist())
    
    # Convert embeddings to string for CSV storage
    features_df['embedding'] = [np.array2string(emb, separator=',') for emb in embeddings]
    
    # Reorder columns
    features_df = features_df[['url', 'word_count', 'sentence_count', 'flesch_reading_ease', 'top_keywords', 'embedding']]
    
    features_df.to_csv(output_csv_path, index=False)
    print(f"Features saved to {output_csv_path}")
    
    return features_df, embeddings


if __name__ == "__main__":
    extract_features_from_dataset('data/extracted_content.csv', 'data/features.csv')
