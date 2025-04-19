import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import logging
import pandas as pd

# --- Configuration ---
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it's in the project root.")
    exit(1)

# Create logs directory if it doesn't exist
os.makedirs(config.LOG_DIR, exist_ok=True)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.FileHandler(config.PREPROCESS_LOG_FILE)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# --- Download necessary NLTK data ---
def download_nltk_data():
    """Downloads required NLTK models if they don't exist."""
    required_nltk_data = ['punkt', 'stopwords', 'wordnet']
    for item in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
        except nltk.downloader.DownloadError:
            logger.info(f"NLTK data '{item}' not found. Downloading...")
            nltk.download(item, quiet=True)
        except LookupError: # Handles cases like 'corpora/wordnet' lookup error
             logger.info(f"NLTK data '{item}' not found. Downloading...")
             nltk.download(item, quiet=True)

download_nltk_data()

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = config.ENGLISH_STOPWORDS # Use stopwords from config (base English)

# --- Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and preprocesses a single string of text.
    Removes HTML, special characters (keeps letters, numbers, whitespace if config.KEEP_NUMBERS is True),
    converts to lowercase, tokenizes, removes stopwords, and lemmatizes.

    Args:
        text (str): The raw text to preprocess.

    Returns:
        str: The preprocessed text, or an empty string if input is invalid.
    """
    # Check for non-string or empty input
    if not isinstance(text, str) or not text.strip():
        return ''

    try:
        # Remove HTML tags (basic regex)
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters, optionally keeping numbers
        if config.KEEP_NUMBERS:
            # Keep letters, numbers, and whitespace
            text = re.sub(r'[^\w\s]', '', text)
        else:
            # Keep only letters and whitespace
            text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        # Keep tokens that are alphabetic (or alphanumeric if numbers kept) and not in stopwords
        cleaned_tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token.isalnum() and token not in stop_words and len(token) > 1 # Keep tokens > 1 char long
        ]

        # Join tokens back into a string
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text

    except Exception as e:
        logger.error(f"Error preprocessing text snippet '{text[:50]}...': {e}", exc_info=True)
        return "" # Return empty string on error

# --- Main Preprocessing Script ---
def run_preprocessing():
    """
    Loads scraped articles, preprocesses the 'text' column, and saves
    the result to a new CSV file.
    """
    logger.info("--- Starting Text Preprocessing ---")

    # Load data
    try:
        df = pd.read_csv(config.SCRAPED_ARTICLES_CSV)
        logger.info(f"Loaded {len(df)} articles from {config.SCRAPED_ARTICLES_CSV}.")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {config.SCRAPED_ARTICLES_CSV}")
        return
    except Exception as e:
        logger.error(f"Error loading CSV {config.SCRAPED_ARTICLES_CSV}: {e}")
        return

    # Validate required columns
    required_columns = {'outlet', 'category', 'link', 'text'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"Missing required columns in {config.SCRAPED_ARTICLES_CSV}: {missing}")
        return

    # Fill potential NaN in 'text' column before processing
    df['text'].fillna('', inplace=True)

    # Apply preprocessing
    logger.info("Applying preprocessing to 'text' column...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    logger.info("Preprocessing complete.")

    # Keep only necessary columns for the processed file
    output_df = df[['outlet', 'category', 'link', 'processed_text']].copy()

     # Remove rows where processing might have failed or resulted in empty text
    original_len = len(output_df)
    output_df = output_df[output_df['processed_text'].str.strip() != '']
    removed_count = original_len - len(output_df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} articles with empty processed text.")


    # Write preprocessed data to a new CSV
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True) # Ensure directory exists
        output_df.to_csv(config.PREPROCESSED_ARTICLES_CSV, index=False, encoding='utf-8')
        logger.info(f"Preprocessing finished. Saved {len(output_df)} processed articles to {config.PREPROCESSED_ARTICLES_CSV}")
    except Exception as e:
        logger.error(f"Error writing preprocessed data to {config.PREPROCESSED_ARTICLES_CSV}: {e}")

    logger.info("--- Finished Text Preprocessing ---")

if __name__ == "__main__":
    run_preprocessing()