import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
import os
import sys

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils.parser import extract_text_from_html
from utils.features import extract_features, preprocess_text, generate_embeddings
from utils.scorer import predict_quality
from utils.similarity import find_similar_pages, parse_embedding


# Page config
st.set_page_config(
    page_title="SEO Content Quality & Duplicate Detector",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load the trained quality model."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'quality_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_dataset():
    """Load features dataset for similarity comparison."""
    features_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        
        # Parse embeddings
        embeddings = []
        valid_indices = []
        for idx, row in df.iterrows():
            emb = parse_embedding(row['embedding'])
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)
        
        embeddings = np.array(embeddings)
        urls = df.iloc[valid_indices]['url'].tolist()
        
        return embeddings, urls
    return None, None


def scrape_url(url, timeout=10):
    """
    Scrape HTML content from URL.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        HTML content string or None on error
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None


def analyze_url(url, model, dataset_embeddings, dataset_urls):
    """
    Analyze a URL and return quality metrics.
    
    Args:
        url: URL to analyze
        model: Trained quality model
        dataset_embeddings: Embeddings from dataset
        dataset_urls: URLs from dataset
        
    Returns:
        Dict with analysis results
    """
    # Scrape URL
    with st.spinner("Fetching page content..."):
        html_content = scrape_url(url)
    
    if not html_content:
        return None
    
    # Extract text
    with st.spinner("Extracting content..."):
        extracted = extract_text_from_html(html_content)
    
    body_text = extracted['body_text']
    word_count = extracted['word_count']
    
    # Debug info
    if not body_text or word_count == 0:
        st.warning("No text content found on the page. The page might be JavaScript-heavy or blocked.")
        st.info(f"Extracted title: {extracted.get('title', 'None')}")
        st.info(f"HTML length: {len(html_content)} characters")
        return None
    
    # Extract features
    with st.spinner("Analyzing content..."):
        features = extract_features(body_text)
    
    # Predict quality
    quality_label = "Unknown"
    if model:
        quality_label = predict_quality(
            model,
            features['word_count'],
            features['sentence_count'],
            features['flesch_reading_ease']
        )
    
    # Check if thin content
    is_thin = word_count < 500
    
    # Generate embedding and find similar pages
    similar_pages = []
    if dataset_embeddings is not None and len(dataset_embeddings) > 0:
        with st.spinner("Finding similar pages..."):
            query_embedding = generate_embeddings([body_text], model_name="all-MiniLM-L6-v2")[0]
            similar_pages = find_similar_pages(
                query_embedding,
                dataset_embeddings,
                dataset_urls,
                top_k=5,
                threshold=0.70
            )
    
    return {
        'url': url,
        'title': extracted['title'],
        'word_count': word_count,
        'sentence_count': features['sentence_count'],
        'readability': features['flesch_reading_ease'],
        'quality_label': quality_label,
        'is_thin': is_thin,
        'similar_to': similar_pages,
        'body_preview': body_text[:500] if body_text else ''
    }


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("SEO Content Quality & Duplicate Detector")
    st.markdown("Analyze web pages for content quality and detect similar/duplicate content.")
    
    # Load model and dataset
    model = load_model()
    dataset_embeddings, dataset_urls = load_dataset()
    
    if model is None:
        st.warning("Quality model not found. Please train the model first by running the pipeline.")
    
    if dataset_embeddings is None:
        st.info("Dataset not loaded. Similar page detection will be unavailable.")
    
    # URL Input
    st.markdown("---")
    url = st.text_input("Enter URL to analyze:", placeholder="https://example.com/article")
    
    analyze_button = st.button("Analyze", type="primary")
    
    if analyze_button and url:
        if not url.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            # Analyze URL
            result = analyze_url(url, model, dataset_embeddings, dataset_urls)
            
            if result:
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Word Count", result['word_count'])
                
                with col2:
                    st.metric("Sentences", result['sentence_count'])
                
                with col3:
                    readability_label = "Good" if 50 <= result['readability'] <= 70 else "Needs Work"
                    st.metric("Readability", f"{result['readability']:.1f}", readability_label)
                
                with col4:
                    quality_color = {
                        'High': '🟢',
                        'Medium': '🟡',
                        'Low': '🔴'
                    }.get(result['quality_label'], '⚪')
                    st.metric("Quality", f"{quality_color} {result['quality_label']}")
                
                # Content flags
                st.markdown("#### Content Flags")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['is_thin']:
                        st.error("⚠️ Thin Content Detected (< 500 words)")
                    else:
                        st.success("✓ Adequate Content Length")
                
                with col2:
                    if 50 <= result['readability'] <= 70:
                        st.success("✓ Optimal Readability")
                    elif result['readability'] < 30:
                        st.error("⚠️ Very Difficult to Read")
                    elif result['readability'] > 90:
                        st.warning("⚠️ May Be Too Simple")
                    else:
                        st.info("ℹ️ Readability Could Be Improved")
                
                # Similar pages
                if result['similar_to'] and len(result['similar_to']) > 0:
                    st.markdown("---")
                    st.subheader("Similar Pages in Dataset")
                    st.markdown("Pages with high similarity may indicate duplicate or near-duplicate content.")
                    
                    similar_df = pd.DataFrame(result['similar_to'])
                    similar_df['similarity'] = (similar_df['similarity'] * 100).round(1).astype(str) + '%'
                    
                    st.dataframe(
                        similar_df,
                        column_config={
                            'url': st.column_config.LinkColumn("URL"),
                            'similarity': "Similarity"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Warning for high similarity
                    max_similarity = max([s['similarity'] for s in result['similar_to']])
                    if max_similarity > 0.80:
                        st.error(f"⚠️ High similarity detected ({max_similarity*100:.1f}%)! This may be duplicate content.")
                    elif max_similarity > 0.70:
                        st.warning(f"⚠️ Moderate similarity detected ({max_similarity*100:.1f}%). Review for potential duplication.")
                
                # Page details
                with st.expander("Page Details"):
                    st.markdown(f"**Title:** {result['title']}")
                    st.markdown(f"**URL:** {result['url']}")
                    if 'body_preview' in result:
                        st.markdown(f"**Text Preview (first 500 chars):**")
                        st.text(result['body_preview'][:500])
    
    # Sidebar - Information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This tool analyzes web page content for:
        - **Content Quality**: Evaluates length and readability
        - **Duplicate Detection**: Finds similar pages in dataset
        - **Thin Content**: Identifies pages with insufficient content
        """)
        
        st.markdown("### Quality Criteria")
        st.markdown("""
        - **High**: 1500+ words, readability 50-70
        - **Medium**: Moderate length and readability
        - **Low**: < 500 words or poor readability
        """)
        
        st.markdown("### Readability Scale")
        st.markdown("""
        - **90-100**: Very Easy
        - **60-70**: Standard
        - **30-50**: Difficult
        - **0-30**: Very Difficult
        """)


if __name__ == "__main__":
    main()
