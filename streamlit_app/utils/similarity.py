import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def parse_embedding(embedding_str):
    """
    Parse embedding string back to numpy array.
    
    Args:
        embedding_str: String representation of numpy array
        
    Returns:
        numpy array
    """
    try:
        # Remove brackets and split by comma
        embedding_str = embedding_str.strip('[]')
        values = [float(x.strip()) for x in embedding_str.split(',')]
        return np.array(values)
    except Exception as e:
        print(f"Error parsing embedding: {str(e)}")
        return None


def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix for embeddings.
    
    Args:
        embeddings: numpy array of embeddings (n_samples, embedding_dim)
        
    Returns:
        Similarity matrix (n_samples, n_samples)
    """
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def detect_duplicates(features_csv_path, output_csv_path, threshold=0.80, thin_threshold=500):
    """
    Detect duplicate content based on embedding similarity.
    
    Args:
        features_csv_path: Path to CSV with features and embeddings
        output_csv_path: Path to save duplicates CSV
        threshold: Similarity threshold for duplicate detection (default: 0.80)
        thin_threshold: Word count threshold for thin content (default: 500)
        
    Returns:
        tuple: (duplicates_df, similarity_matrix, summary_stats)
    """
    print(f"Loading features from {features_csv_path}...")
    df = pd.read_csv(features_csv_path)
    
    if 'embedding' not in df.columns:
        raise ValueError("Input CSV must contain 'embedding' column")
    
    print("Parsing embeddings...")
    embeddings = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing embeddings"):
        emb = parse_embedding(row['embedding'])
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(idx)
    
    embeddings = np.array(embeddings)
    print(f"Parsed {len(embeddings)} embeddings")
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Find duplicate pairs
    print(f"Detecting duplicates (threshold: {threshold})...")
    duplicates = []
    n = len(embeddings)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i, j]
            
            if similarity > threshold:
                idx_i = valid_indices[i]
                idx_j = valid_indices[j]
                
                duplicates.append({
                    'url_1': df.iloc[idx_i]['url'],
                    'url_2': df.iloc[idx_j]['url'],
                    'similarity': round(similarity, 4),
                    'word_count_1': df.iloc[idx_i]['word_count'],
                    'word_count_2': df.iloc[idx_j]['word_count']
                })
    
    duplicates_df = pd.DataFrame(duplicates)
    
    # Identify thin content
    thin_pages = df[df['word_count'] < thin_threshold].copy()
    thin_pages['is_thin'] = True
    
    # Save duplicates
    duplicates_df.to_csv(output_csv_path, index=False)
    print(f"Duplicates saved to {output_csv_path}")
    
    # Summary statistics
    summary_stats = {
        'total_pages': len(df),
        'duplicate_pairs': len(duplicates_df),
        'thin_pages': len(thin_pages),
        'thin_threshold': thin_threshold,
        'similarity_threshold': threshold
    }
    
    print("\n=== Summary Statistics ===")
    print(f"Total pages analyzed: {summary_stats['total_pages']}")
    print(f"Duplicate pairs found (similarity > {threshold}): {summary_stats['duplicate_pairs']}")
    print(f"Thin pages (word count < {thin_threshold}): {summary_stats['thin_pages']}")
    
    return duplicates_df, similarity_matrix, summary_stats


def find_similar_pages(query_embedding, dataset_embeddings, dataset_urls, top_k=5, threshold=0.70):
    """
    Find similar pages to a query embedding.
    
    Args:
        query_embedding: numpy array of query embedding
        dataset_embeddings: numpy array of dataset embeddings
        dataset_urls: list of URLs corresponding to dataset embeddings
        top_k: Number of top similar pages to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of dicts with url and similarity
    """
    # Compute similarity
    query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, dataset_embeddings)[0]
    
    # Get top k similar pages above threshold
    similar_indices = np.argsort(similarities)[::-1][:top_k]
    
    similar_pages = []
    for idx in similar_indices:
        similarity = similarities[idx]
        if similarity >= threshold:
            similar_pages.append({
                'url': dataset_urls[idx],
                'similarity': round(float(similarity), 4)
            })
    
    return similar_pages


if __name__ == "__main__":
    detect_duplicates('data/features.csv', 'data/duplicates.csv')
