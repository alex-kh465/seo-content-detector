import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


def extract_text_from_html(html_content):
    """
    Extract title and body text from HTML content.
    
    Args:
        html_content: Raw HTML string
        
    Returns:
        dict with keys: title, body_text, word_count
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract main content from common tags
        body_text = ""
        
        # Try to find main content areas first
        main_content = soup.find_all(['article', 'main', 'div'], class_=re.compile('content|article|main', re.I))
        
        if main_content:
            for element in main_content:
                paragraphs = element.find_all('p')
                body_text += " ".join([p.get_text(strip=True) for p in paragraphs])
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all('p')
            body_text = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Clean whitespace
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        
        # Calculate word count
        word_count = len(body_text.split()) if body_text else 0
        
        return {
            'title': title,
            'body_text': body_text,
            'word_count': word_count
        }
    
    except Exception as e:
        print(f"Error parsing HTML: {str(e)}")
        return {
            'title': "",
            'body_text': "",
            'word_count': 0
        }


def parse_dataset(input_csv_path, output_csv_path):
    """
    Parse HTML content from dataset and save extracted text.
    
    Args:
        input_csv_path: Path to CSV with columns 'url' and 'html_content'
        output_csv_path: Path to save extracted content CSV
        
    Returns:
        DataFrame with extracted content
    """
    print(f"Loading dataset from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    if 'url' not in df.columns or 'html_content' not in df.columns:
        raise ValueError("Input CSV must contain 'url' and 'html_content' columns")
    
    print(f"Processing {len(df)} pages...")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing HTML"):
        extracted = extract_text_from_html(row['html_content'])
        results.append({
            'url': row['url'],
            'title': extracted['title'],
            'body_text': extracted['body_text'],
            'word_count': extracted['word_count']
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Extracted content saved to {output_csv_path}")
    
    return result_df


if __name__ == "__main__":
    parse_dataset('data/data.csv', 'data/extracted_content.csv')
