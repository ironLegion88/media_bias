import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import ttest_ind
import re
import os

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# --- Configuration ---
INPUT_CSV = 'data/articles.csv'
OUTPUT_DIR = 'analysis_results'
TOP_N_WORDS = 30  # Number of top words to show

# --- Ensure output directory exists ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helper Functions ---

def get_sentiment(text):
    """Calculates VADER sentiment compound score."""
    analyzer = SentimentIntensityAnalyzer()
    # VADER works best on raw-ish text, but preprocessed is okay too.
    # Handle potential NaN or non-string inputs
    if isinstance(text, str):
        return analyzer.polarity_scores(text)['compound']
    return 0.0 # Return neutral for non-string input

def get_word_frequency(text_series, top_n=TOP_N_WORDS, extra_stopwords=None):
    """Calculates word frequency from a pandas Series of text."""
    if text_series.empty:
        return []

    # Combine all text into one large string
    all_text = ' '.join(text_series.astype(str).tolist())

    # Tokenize (assuming already preprocessed, so mostly space-separated)
    tokens = word_tokenize(all_text.lower()) # Lowercase again to be safe

    # Filter words: alphabetic, length > 2, not stopwords
    stop_words_set = set(stopwords.words('english'))
    if extra_stopwords:
        stop_words_set.update(extra_stopwords)

    filtered_tokens = [
        word for word in tokens
        if word.isalpha() and len(word) > 2 and word not in stop_words_set
    ]

    # Calculate frequency
    freq_dist = Counter(filtered_tokens)
    return freq_dist.most_common(top_n)

def plot_sentiment_distribution(df, group_col, title, filename):
    """Plots sentiment score distribution by group (outlet or bias)."""
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=group_col, y='sentiment_score', palette='coolwarm')
    plt.title(title, fontsize=16)
    plt.ylabel('Sentiment Score (VADER Compound)', fontsize=12)
    plt.xlabel(group_col.capitalize(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot: {filename}")

def plot_average_sentiment(avg_sentiment_series, title, filename):
    """Plots average sentiment scores."""
    plt.figure(figsize=(10, 6))
    avg_sentiment_series.plot(kind='bar', color=sns.color_palette("coolwarm", len(avg_sentiment_series)))
    plt.title(title, fontsize=16)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--') # Zero line for reference
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot: {filename}")

def plot_word_cloud(freq_dist, title, filename):
    """Generates and saves a word cloud."""
    if not freq_dist:
        print(f"No words to generate word cloud for '{title}'. Skipping.")
        return

    wc = WordCloud(width=1000, height=500, background_color='white',
                   max_words=100, colormap='viridis').generate_from_frequencies(dict(freq_dist))
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved word cloud: {filename}")

# --- Main Analysis Script ---
def analyze_articles(filepath=INPUT_CSV):
    """Loads data, performs analysis, and saves results."""
    print(f"--- Starting Analysis on {filepath} ---")

    # 1. Load Data
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} articles.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Basic data cleaning / validation
    if 'text' not in df.columns or 'outlet' not in df.columns or 'bias' not in df.columns:
        print("Error: CSV must contain 'text', 'outlet', and 'bias' columns.")
        return
    df.dropna(subset=['text', 'outlet', 'bias'], inplace=True)
    df = df[df['text'].str.strip() != ''] # Remove rows with empty text after stripping
    df['bias'] = df['bias'].str.lower() # Standardize bias labels
    print(f"Using {len(df)} articles after cleaning.")

    if df.empty:
        print("No valid data found after cleaning. Exiting.")
        return

    # 2. Sentiment Analysis
    print("\n--- Performing Sentiment Analysis ---")
    df['sentiment_score'] = df['text'].apply(get_sentiment)
    print("Sentiment scores calculated.")

    # Aggregate Sentiment
    avg_sentiment_outlet = df.groupby('outlet')['sentiment_score'].mean().sort_values()
    avg_sentiment_bias = df.groupby('bias')['sentiment_score'].mean().sort_values()

    print("\nAverage Sentiment Scores per Outlet:")
    print(avg_sentiment_outlet)
    print("\nAverage Sentiment Scores per Bias Group:")
    print(avg_sentiment_bias)

    # Visualize Sentiment
    plot_sentiment_distribution(df, 'outlet', 'Sentiment Score Distribution by Outlet', 'sentiment_dist_outlet.png')
    plot_sentiment_distribution(df, 'bias', 'Sentiment Score Distribution by Bias Group', 'sentiment_dist_bias.png')
    plot_average_sentiment(avg_sentiment_outlet, 'Average Sentiment Score by Outlet', 'avg_sentiment_outlet.png')
    plot_average_sentiment(avg_sentiment_bias, 'Average Sentiment Score by Bias Group', 'avg_sentiment_bias.png')

    # 3. Word Frequency Analysis
    print("\n--- Performing Word Frequency Analysis ---")
    # Define common political/election context words that might be frequent but not specific bias indicators
    # You might need to refine this list based on your results
    common_context_stopwords = [
        'delhi', 'election', 'bjp', 'aap', 'congress', 'party', 'govt', 'government',
        'state', 'chief', 'minister', 'leader', 'vote', 'voter', 'seat', 'poll',
        'campaign', 'result', 'kejriwal', 'modi', 'rekha', 'gupta', 'atishi', 'verma',
        'said', 'also', 'would', 'could', 'like', 'one', 'two', 'year', 'people'
    ]

    # Overall Frequency
    overall_freq = get_word_frequency(df['text'], extra_stopwords=common_context_stopwords)
    print(f"\nTop {TOP_N_WORDS} Overall Words (excluding common context):")
    print(overall_freq)
    plot_word_cloud(overall_freq, 'Overall Top Words', 'wordcloud_overall.png')

    # Frequency per Bias Group
    bias_groups = df['bias'].unique()
    word_freq_by_bias = {}
    for bias_label in bias_groups:
        print(f"\n--- Analyzing Bias Group: {bias_label.upper()} ---")
        bias_df = df[df['bias'] == bias_label]
        if not bias_df.empty:
            freq = get_word_frequency(bias_df['text'], extra_stopwords=common_context_stopwords)
            word_freq_by_bias[bias_label] = freq
            print(f"Top {TOP_N_WORDS} Words for '{bias_label}' (excluding common context):")
            print(freq)
            plot_word_cloud(freq, f'Top Words for {bias_label.capitalize()} Outlets', f'wordcloud_{bias_label}.png')
        else:
            print(f"No articles found for bias group: {bias_label}")
            word_freq_by_bias[bias_label] = []

    # 4. Statistical Tests (Example: Left vs. Right Sentiment)
    print("\n--- Performing Statistical Tests (Example) ---")
    if 'left' in bias_groups and 'right' in bias_groups:
        sentiment_left = df[df['bias'] == 'left']['sentiment_score']
        sentiment_right = df[df['bias'] == 'right']['sentiment_score']

        if len(sentiment_left) > 1 and len(sentiment_right) > 1:
            t_stat, p_value = ttest_ind(sentiment_left, sentiment_right, equal_var=False, nan_policy='omit') # Welch's t-test
            print(f"\nT-test comparing sentiment (Left vs. Right):")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  Result: Statistically significant difference (p < 0.05)")
            else:
                print("  Result: No statistically significant difference (p >= 0.05)")
        else:
             print("Not enough data in 'left' or 'right' groups to perform t-test.")
    else:
        print("Could not perform Left vs. Right t-test (one or both groups missing).")

    # 5. Save Aggregate Results (Optional: can be used for correlation with survey)
    print("\n--- Saving Aggregate Results ---")
    output_summary_path = os.path.join(OUTPUT_DIR, 'outlet_sentiment_summary.csv')
    avg_sentiment_outlet.to_csv(output_summary_path, header=['average_sentiment'])
    print(f"Saved average sentiment per outlet to {output_summary_path}")

    print("\n--- Analysis Complete ---")
    print(f"Results and plots saved in '{OUTPUT_DIR}' directory.")

# --- Run the Analysis ---
if __name__ == "__main__":
    analyze_articles()