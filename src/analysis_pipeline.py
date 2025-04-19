import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration & Setup ---

# Ensure NLTK data is available (might need download if not already done)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt') # Needed potentially for VADER or advanced tokenization

# Define file paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, '..', 'data', 'preprocessed_articles.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'analysis_results1')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Ideological/Emotionally Loaded Terms (Expand based on your manual analysis and research)
# Using categories based on your proposal and manual analysis structure
IDEOLOGICAL_TERMS = {
    "pro_bjp_modi": [
        "masterstroke", "visionary leadership", "modi’s resolve", "modi magic",
        "strong governance", "development agenda", "nation building", "double engine", "double-engine",
        "governance efficiency", "economic growth", "reform" , "nationalism", "hindutva"
    ],
    "pro_aap_kejriwal": [
        "aam aadmi", "common man", "welfare", "governance model", "delhi model",
        "education reform", "healthcare reform", "mohalla clinic", "anti-corruption",
        "people's mandate", "disruptor", "alternative politics"
    ],
    "critical_bjp_modi": [
        "authoritarian", "centralized control", "erosion of democracy", "unilateral decision",
        "press freedom curbed", "dissent silenced", "government overreach", "censorship",
        "communal", "polarisation", "majoritarian", "institutional bias", "power grab"
    ],
    "critical_aap_kejriwal": [
        "anarchy", "populism", "freebie", "revdi", "u-turn", "confrontational politics",
        "liquor scam", "corruption", "mismanagement", "sheesh mahal", "false promise",
         "governance failure"
    ],
    "neutral_governance": [
        "voter turnout", "urban planning", "fiscal allocation", "policy continuity",
        "electoral outcome", "citizen response", "bureaucratic reshuffle", "infrastructure",
        "civic amenities", "administration", "poll promises"
    ],
    "emotive_political": [
        "political maneuvering", "mobilization tactics", "high-stakes", "battleground",
        "power grab", "betrayal", "scandal", "controversy", "decisive mandate", "rout",
        "setback", "spectacular victory", "existential crisis"
    ]
}

# Define terms specifically associated with political parties for sentiment comparison later
PARTY_TERMS = {
    'BJP': ['bjp', 'bharatiya janata party', 'modi', 'shah', 'yogi', 'adityanath', 'nadda', 'saffron party', 'rekha gupta', 'parvesh verma'],
    'AAP': ['aap', 'aam aadmi party', 'kejriwal', 'atishi', 'sisodia', 'common man party'],
    'Congress': ['congress', 'inc', 'rahul gandhi', 'gandhi', 'dikshit', 'sandeep dikshit']
    # Add other relevant parties/leaders if needed
}

# --- Analysis Functions ---

def load_data(filepath):
    """Loads the preprocessed data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        # Handle potential missing text data gracefully
        df['text'] = df['text'].fillna('')
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_sentiment_analysis(df):
    """
    Performs sentiment analysis on the 'text' column using TextBlob
    and adds 'polarity' and 'subjectivity' columns.
    """
    if df is None or 'text' not in df.columns:
        print("Error: DataFrame is invalid or missing 'text' column for sentiment analysis.")
        return df

    print("\nPerforming Sentiment Analysis...")
    polarities = []
    subjectivities = []

    for text in df['text']:
        if isinstance(text, str) and text.strip(): # Check if text is a non-empty string
            try:
                blob = TextBlob(text)
                polarities.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            except Exception as e:
                print(f"Error processing text for sentiment: {text[:50]}... Error: {e}")
                polarities.append(None) # Append None or 0 for problematic texts
                subjectivities.append(None)
        else:
            polarities.append(0.0) # Assign neutral sentiment to empty texts
            subjectivities.append(0.0)

    df['polarity'] = polarities
    df['subjectivity'] = subjectivities
    print("Sentiment analysis complete.")
    return df

def analyze_word_frequency(df, top_n=50):
    """
    Performs overall word frequency analysis on the combined preprocessed text.
    Excludes common English stopwords again just in case preprocessing missed some context.
    """
    if df is None or 'text' not in df.columns:
        print("Error: DataFrame is invalid or missing 'text' column for word frequency analysis.")
        return None, None

    print(f"\nPerforming Word Frequency Analysis (Top {top_n})...")
    # Combine all preprocessed text into one large string
    # Ensure only non-empty strings are joined
    all_text = ' '.join(df['text'].astype(str).tolist())

    if not all_text.strip():
        print("Warning: No text content found for word frequency analysis.")
        return Counter(), {} # Return empty counter and dict

    # Tokenize - assuming text is already reasonably clean from preprocessing.py
    words = re.findall(r'\b\w+\b', all_text.lower()) # Simple word tokenization

    # Use NLTK stopwords, plus potentially some domain-specific ones if needed
    stop_words_set = set(stopwords.words('english'))
    # Add any other words you want to ignore (e.g., common but uninformative like 'also', 'said')
    custom_stopwords = {'also', 'said', 'would', 'could', 'like', 'one', 'delhi', 'party', 'bjp', 'aap', 'congress'} # Add common political entities if you want to focus on *other* words
    stop_words_set.update(custom_stopwords)

    # Filter words
    filtered_words = [word for word in words if word not in stop_words_set and len(word) > 2] # Keep words longer than 2 chars

    # Calculate frequency distribution
    word_counts = Counter(filtered_words)

    most_common = word_counts.most_common(top_n)
    print(f"Word frequency analysis complete. Found {len(word_counts)} unique words.")

    return word_counts, most_common

def analyze_ideological_term_frequency(df, term_dict):
    """
    Counts the frequency of specific ideological terms within the articles.
    Returns counts per outlet and overall.
    """
    if df is None or 'text' not in df.columns:
        print("Error: DataFrame is invalid for ideological term analysis.")
        return None

    print("\nAnalyzing Ideological Term Frequency...")
    # Initialize dictionaries to store counts
    term_counts_overall = {category: Counter() for category in term_dict}
    term_counts_per_outlet = {outlet: {category: Counter() for category in term_dict} for outlet in df['outlet'].unique()}

    # Combine all terms for efficient searching
    all_search_terms = set()
    for category_terms in term_dict.values():
        all_search_terms.update(category_terms)

    # Pre-compile regex for efficiency if using regex search extensively (optional)

    for index, row in df.iterrows():
        outlet = row['outlet']
        text = row['text']

        if not isinstance(text, str) or not text.strip():
            continue # Skip rows with empty text

        # Simple word tokenization for counting
        words_in_article = re.findall(r'\b\w+\b', text.lower())
        article_word_counts = Counter(words_in_article)

        for category, terms in term_dict.items():
            for term in terms:
                # Count occurrences of the term in this article's words
                count = article_word_counts[term]
                if count > 0:
                    term_counts_overall[category][term] += count
                    term_counts_per_outlet[outlet][category][term] += count

    print("Ideological term frequency analysis complete.")
    return term_counts_overall, term_counts_per_outlet

def analyze_sentiment_towards_parties(df, party_term_dict):
    """
    Calculates average sentiment of articles mentioning specific parties.
    This is a basic approach; more advanced methods exist (e.g., aspect-based sentiment).
    """
    if df is None or 'text' not in df.columns or 'polarity' not in df.columns:
        print("Error: DataFrame is invalid for party sentiment analysis (requires text and polarity).")
        return None

    print("\nAnalyzing Sentiment Towards Political Parties...")
    party_sentiment = {party: {'total_polarity': 0, 'article_count': 0, 'mentioned_articles': []} for party in party_term_dict}

    for index, row in df.iterrows():
        text_lower = str(row['text']).lower() # Ensure text is string and lowercased
        polarity = row['polarity']

        if pd.isna(polarity) or not text_lower.strip():
            continue # Skip if polarity is NaN or text is empty

        for party, terms in party_term_dict.items():
            # Check if any term for the party is present in the article text
            if any(f' {term} ' in f' {text_lower} ' for term in terms): # Basic check using spaces to avoid partial matches
                party_sentiment[party]['total_polarity'] += polarity
                party_sentiment[party]['article_count'] += 1
                party_sentiment[party]['mentioned_articles'].append(row['link']) # Store links for reference

    # Calculate average sentiment
    avg_party_sentiment = {}
    for party, data in party_sentiment.items():
        if data['article_count'] > 0:
            avg_sentiment = data['total_polarity'] / data['article_count']
            avg_party_sentiment[party] = {
                'average_polarity': avg_sentiment,
                'article_count': data['article_count']
                # 'articles': data['mentioned_articles'] # Uncomment to see specific articles
                }
        else:
             avg_party_sentiment[party] = {
                'average_polarity': 0.0,
                'article_count': 0
                }

    print("Party sentiment analysis complete.")
    return avg_party_sentiment


def display_results(df_sentiment, avg_sentiment_outlet, overall_word_counts, most_common_words, term_counts_overall, term_counts_per_outlet, avg_party_sentiment):
    """Displays the analysis results in a structured format."""

    print("\n\n--- Analysis Results ---")

    # --- Sentiment Analysis Results ---
    print("\n=== Sentiment Analysis ===")
    if df_sentiment is not None:
        print("\nSample Articles with Sentiment Scores:")
        print(df_sentiment[['outlet', 'link', 'polarity', 'subjectivity']].head())

        print("\nAverage Sentiment Polarity per Outlet:")
        print(avg_sentiment_outlet.to_string())

        # Optional: Save sentiment data to CSV
        sentiment_output_path = os.path.join(OUTPUT_DIR, 'articles_with_sentiment.csv')
        try:
            df_sentiment.to_csv(sentiment_output_path, index=False, encoding='utf-8')
            print(f"\nFull sentiment data saved to: {sentiment_output_path}")
        except Exception as e:
            print(f"Error saving sentiment data: {e}")

        # Simple Plot (Optional)
        if not avg_sentiment_outlet.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=avg_sentiment_outlet.index, y=avg_sentiment_outlet.values)
            plt.title('Average Sentiment Polarity per News Outlet')
            plt.xlabel('News Outlet')
            plt.ylabel('Average Polarity (TextBlob)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, 'avg_sentiment_per_outlet.png')
            try:
                plt.savefig(plot_path)
                print(f"Sentiment plot saved to: {plot_path}")
                # plt.show() # Uncomment to display plot directly
                plt.close()
            except Exception as e:
                print(f"Error saving sentiment plot: {e}")


    # --- Word Frequency Results ---
    print("\n=== Word Frequency Analysis ===")
    if most_common_words:
        print(f"\nTop {len(most_common_words)} Most Common Words (Overall, excluding stopwords):")
        for word, count in most_common_words:
            print(f"- {word}: {count}")
    else:
        print("No word frequency data to display.")

    # --- Ideological Term Frequency Results ---
    print("\n=== Ideological Term Frequency ===")
    if term_counts_overall:
        print("\nOverall Frequency of Ideological Terms:")
        for category, counter in term_counts_overall.items():
            if counter: # Only print categories with found terms
                print(f"\n  Category: {category}")
                # Sort terms by frequency within the category
                sorted_terms = sorted(counter.items(), key=lambda item: item[1], reverse=True)
                for term, count in sorted_terms:
                    print(f"  - {term}: {count}")

        # Optional: Print per outlet details (can be very verbose)
        # print("\nFrequency of Ideological Terms per Outlet:")
        # for outlet, categories in term_counts_per_outlet.items():
        #     print(f"\n  Outlet: {outlet}")
        #     for category, counter in categories.items():
        #          if counter:
        #             print(f"    Category: {category}")
        #             sorted_terms = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        #             for term, count in sorted_terms:
        #                 print(f"    - {term}: {count}")
    else:
        print("No ideological term frequency data to display.")

    # --- Party Sentiment Results ---
    print("\n=== Sentiment Towards Political Parties ===")
    if avg_party_sentiment:
        print("\nAverage Sentiment Polarity in Articles Mentioning Parties:")
        # Convert to DataFrame for nice printing
        party_sentiment_df = pd.DataFrame.from_dict(avg_party_sentiment, orient='index')
        party_sentiment_df = party_sentiment_df.sort_values('average_polarity', ascending=False)
        print(party_sentiment_df.to_string())

        # Optional Plot
        if not party_sentiment_df.empty:
             plt.figure(figsize=(8, 5))
             sns.barplot(x=party_sentiment_df.index, y=party_sentiment_df['average_polarity'], palette='viridis')
             plt.title('Average Sentiment Towards Political Parties (in Mentioning Articles)')
             plt.xlabel('Party')
             plt.ylabel('Average Polarity (TextBlob)')
             plt.xticks(rotation=0)
             plt.tight_layout()
             plot_path = os.path.join(OUTPUT_DIR, 'avg_sentiment_per_party.png')
             try:
                 plt.savefig(plot_path)
                 print(f"Party sentiment plot saved to: {plot_path}")
                 # plt.show() # Uncomment to display plot directly
                 plt.close()
             except Exception as e:
                 print(f"Error saving party sentiment plot: {e}")

    else:
        print("No party sentiment data to display.")


    print("\n--- End of Analysis ---")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Media Bias Analysis Pipeline...")

    # 1. Load Data
    df = load_data(INPUT_CSV)

    if df is not None:
        # Add a check for empty dataframe
        if df.empty:
            print("Error: The loaded DataFrame is empty. Cannot proceed with analysis.")
        else:
            # 2. Perform Sentiment Analysis
            df_sentiment = perform_sentiment_analysis(df.copy()) # Use copy to avoid modifying original df if needed later

            avg_sentiment_per_outlet = pd.Series(dtype=float) # Initialize empty series
            if df_sentiment is not None and 'outlet' in df_sentiment.columns and 'polarity' in df_sentiment.columns:
                 # Ensure polarity is numeric before grouping
                df_sentiment['polarity'] = pd.to_numeric(df_sentiment['polarity'], errors='coerce')
                df_sentiment.dropna(subset=['polarity'], inplace=True) # Drop rows where polarity couldn't be calculated
                if not df_sentiment.empty:
                     avg_sentiment_per_outlet = df_sentiment.groupby('outlet')['polarity'].mean().sort_values()


            # 3. Perform Word Frequency Analysis
            overall_word_counts, most_common_words = analyze_word_frequency(df)

            # 4. Analyze Ideological Term Frequency
            term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, IDEOLOGICAL_TERMS)

            # 5. Analyze Sentiment Towards Parties
            avg_party_sentiment = analyze_sentiment_towards_parties(df_sentiment, PARTY_TERMS)


            # 6. Display Results
            display_results(
                df_sentiment,
                avg_sentiment_per_outlet,
                overall_word_counts,
                most_common_words,
                term_counts_overall,
                term_counts_per_outlet,
                avg_party_sentiment
            )
    else:
        print("Failed to load data. Exiting.")

    print("\nPipeline execution finished.")