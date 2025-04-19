import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import f_oneway, chi2_contingency, ttest_ind
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import re
import os
import logging
import json # For saving complex dicts like topic models

# --- Configuration ---
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it's in the project root.")
    exit(1)

# Create output and logs directories if they don't exist
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log_handler = logging.FileHandler(config.ANALYSIS_VADER_LOG_FILE)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent adding handler multiple times if script is re-run in same session
if not logger.hasHandlers():
    logger.addHandler(log_handler)
# Optional: Add console handler
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# logger.addHandler(console_handler)


# --- Download necessary NLTK data ---
def download_nltk_data():
    """Downloads required NLTK models if they don't exist."""
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    for item in required_nltk_data:
        try:
            if item == 'vader_lexicon':
                 nltk.data.find(f'sentiment/{item}.zip')
            else:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
        except (LookupError, nltk.downloader.DownloadError):
            logger.info(f"NLTK data '{item}' not found or outdated. Downloading...")
            nltk.download(item, quiet=True)

download_nltk_data()

# --- Load spaCy Model ---
NLP = None
try:
    NLP = spacy.load(config.SPACY_MODEL)
    logger.info(f"spaCy model '{config.SPACY_MODEL}' loaded successfully.")
except OSError:
    logger.warning(f"spaCy model '{config.SPACY_MODEL}' not found. Downloading...")
    try:
        spacy.cli.download(config.SPACY_MODEL)
        NLP = spacy.load(config.SPACY_MODEL)
        logger.info(f"spaCy model '{config.SPACY_MODEL}' downloaded and loaded.")
    except Exception as e:
        logger.error(f"Failed to download or load spaCy model '{config.SPACY_MODEL}'. NER features will be disabled. Error: {e}")
except ImportError:
     logger.error("spaCy library not installed. NER features will be disabled. Run 'pip install spacy'.")


# === Helper & Analysis Functions ===

def load_data(filepath):
    """Loads the preprocessed data from a CSV file."""
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['processed_text'] = df['processed_text'].fillna('') # Handle potential NaNs
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        if df.empty:
            logger.warning("Loaded DataFrame is empty.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def perform_vader_sentiment_analysis(df):
    """
    Performs sentiment analysis on the 'processed_text' column using VADER
    and adds 'vader_compound' score column.
    """
    logger.info("Performing VADER Sentiment Analysis...")
    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid or missing 'processed_text' column for VADER analysis.")
        return df

    analyzer = SentimentIntensityAnalyzer()
    df['vader_compound'] = df['processed_text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
    logger.info("VADER sentiment analysis complete.")
    return df

def analyze_word_frequency(text_series, top_n=config.TOP_N_WORDS, stopwords_set=config.COMBINED_STOPWORDS):
    """
    Performs word frequency analysis on a pandas Series of preprocessed text.

    Args:
        text_series (pd.Series): Series containing preprocessed text documents.
        top_n (int): Number of most frequent words to return.
        stopwords_set (set): Set of stopwords to exclude.

    Returns:
        list: A list of tuples (word, count) for the top N words.
              Returns empty list if input is empty or invalid.
    """
    logger.info(f"Performing Word Frequency Analysis (Top {top_n})...")
    if text_series is None or text_series.empty:
        logger.warning("Input text series is empty for word frequency analysis.")
        return []

    # Combine all text, ensure it's string type
    try:
        all_text = ' '.join(text_series.astype(str).tolist())
    except Exception as e:
        logger.error(f"Error combining text series for word frequency: {e}")
        return []

    if not all_text.strip():
        logger.warning("No text content found for word frequency analysis after joining.")
        return []

    # Tokenize (simple split assuming preprocessed text)
    tokens = all_text.split() # Using split() as text is already joined processed tokens

    # Filter words: length > 2, not stopwords
    # No need to check isalpha() here as preprocessing should have handled it
    filtered_tokens = [
        word for word in tokens if len(word) > 2 and word not in stopwords_set
    ]

    # Calculate frequency
    word_counts = Counter(filtered_tokens)
    most_common = word_counts.most_common(top_n)
    logger.info(f"Word frequency analysis complete. Found {len(word_counts)} unique words (after filtering).")

    return most_common

def analyze_ideological_term_frequency(df, term_dict):
    """
    Counts the frequency of specific ideological terms within the processed articles.

    Args:
        df (pd.DataFrame): DataFrame with 'outlet' and 'processed_text' columns.
        term_dict (dict): Dictionary where keys are categories and values are lists of terms.

    Returns:
        tuple: (term_counts_overall, term_counts_per_outlet)
               - term_counts_overall (dict): {category: Counter(term: count)}
               - term_counts_per_outlet (dict): {outlet: {category: Counter(term: count)}}
               Returns (None, None) on error.
    """
    logger.info("Analyzing Ideological Term Frequency...")
    if df is None or not {'outlet', 'processed_text'}.issubset(df.columns):
        logger.error("DataFrame invalid or missing columns for ideological term analysis.")
        return None, None

    term_counts_overall = {category: Counter() for category in term_dict}
    outlets = df['outlet'].unique()
    term_counts_per_outlet = {outlet: {category: Counter() for category in term_dict} for outlet in outlets}

    # Pre-process terms in the dictionary (e.g., lemmatize if needed, assuming processed_text is lemmatized)
    processed_term_dict = {}
    lemmatizer = WordNetLemmatizer() # Assuming WordNetLemmatizer was used in preprocessing
    for category, terms in term_dict.items():
         processed_term_dict[category] = [lemmatizer.lemmatize(term.lower()) for term in terms]


    for index, row in df.iterrows():
        outlet = row['outlet']
        text = row['processed_text']

        if not isinstance(text, str) or not text.strip():
            continue

        # Simple split is sufficient as text is preprocessed
        words_in_article = text.split()
        article_word_counts = Counter(words_in_article)

        for category, processed_terms in processed_term_dict.items():
            original_terms = term_dict[category] # Keep original terms for reporting
            for i, term in enumerate(processed_terms):
                original_term = original_terms[i] # Get corresponding original term
                # Count occurrences of the processed term in this article's words
                count = article_word_counts[term]
                if count > 0:
                    # Store counts using the ORIGINAL term for readability
                    term_counts_overall[category][original_term] += count
                    if outlet in term_counts_per_outlet: # Check if outlet exists (handles potential data issues)
                         term_counts_per_outlet[outlet][category][original_term] += count

    logger.info("Ideological term frequency analysis complete.")
    return term_counts_overall, term_counts_per_outlet


def analyze_sentiment_towards_parties_vader(df, party_term_dict):
    """
    Calculates average VADER sentiment of sentences mentioning specific parties.

    Args:
        df (pd.DataFrame): DataFrame containing 'processed_text'. Original text might be better
                           but requires adding it back or modifying preprocessing. Using processed for now.
        party_term_dict (dict): Dictionary mapping party names to lists of keywords.

    Returns:
        dict: Average VADER compound sentiment per party for mentioning sentences.
              Returns None on error.
    """
    logger.info("Analyzing Sentiment Towards Political Parties (VADER - Sentence Level)...")
    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid or missing 'processed_text' for party sentiment analysis.")
        return None

    analyzer = SentimentIntensityAnalyzer()
    party_sentiments = {party: {'total_compound': 0.0, 'sentence_count': 0} for party in party_term_dict}

    # Pre-process party terms (e.g., lemmatize) to match processed_text
    processed_party_term_dict = {}
    lemmatizer = WordNetLemmatizer()
    for party, terms in party_term_dict.items():
         processed_party_term_dict[party] = [lemmatizer.lemmatize(term.lower()) for term in terms]

    total_articles = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0 or i == total_articles - 1: # Log progress periodically
             logger.info(f"Processing party sentiment in article {i+1}/{total_articles}")

        text = row['processed_text']
        if not isinstance(text, str) or not text.strip():
            continue

        try:
            sentences = sent_tokenize(text) # Sentence tokenization on processed text
        except Exception as e:
            logger.warning(f"Could not sentence-tokenize article snippet: {text[:100]}... Error: {e}")
            continue

        for sentence in sentences:
            sentence_lower = sentence.lower() # Lowercase for matching
            sentence_words = set(sentence_lower.split()) # Use set for faster lookups

            for party, processed_terms in processed_party_term_dict.items():
                # Check if any term for the party is present in the sentence words
                if any(term in sentence_words for term in processed_terms):
                    try:
                        # Calculate VADER score for the *original sentence* if possible,
                        # or the processed sentence. Using processed sentence here.
                        vs = analyzer.polarity_scores(sentence)
                        party_sentiments[party]['total_compound'] += vs['compound']
                        party_sentiments[party]['sentence_count'] += 1
                        # Optimization: if a sentence mentions multiple parties, it contributes to both
                    except Exception as e:
                         logger.warning(f"VADER error on sentence: {sentence[:50]}... Error: {e}")


    # Calculate average sentiment
    avg_party_sentiment = {}
    for party, data in party_sentiments.items():
        if data['sentence_count'] > 0:
            avg_sentiment = data['total_compound'] / data['sentence_count']
            avg_party_sentiment[party] = {
                'average_vader_compound': avg_sentiment,
                'sentence_count': data['sentence_count']
            }
        else:
             avg_party_sentiment[party] = {
                'average_vader_compound': 0.0,
                'sentence_count': 0
            }
    logger.info("Party sentiment analysis (VADER - Sentence Level) complete.")
    return avg_party_sentiment

def perform_ner(df, top_n=config.TOP_N_ENTITIES):
    """
    Performs Named Entity Recognition using spaCy to find PERSON and ORG entities.

    Args:
        df (pd.DataFrame): DataFrame with 'processed_text'. Using original 'text'
                           (if available) is generally better for NER accuracy.
                           Requires modification if only processed_text is available.
                           Let's assume original text needs to be loaded or passed.
        top_n (int): Number of top entities to report per category.

    Returns:
        tuple: (ner_persons_overall, ner_orgs_overall, ner_per_outlet)
               Returns (None, None, None) if spaCy model failed to load or on error.
    """
    logger.info("Performing Named Entity Recognition (NER)...")
    if NLP is None:
        logger.error("spaCy model not loaded. Cannot perform NER.")
        return None, None, None
    # This function ideally needs the *original* text for best NER results.
    # Let's modify the structure slightly to load the original text if needed.
    # We'll assume 'processed_text' is passed for now, but flag this limitation.
    logger.warning("NER is being performed on 'processed_text'. Results may be less accurate than on original text.")

    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid or missing 'processed_text' for NER analysis.")
        return None, None, None

    ner_persons_overall = Counter()
    ner_orgs_overall = Counter()
    outlets = df['outlet'].unique()
    ner_per_outlet = {outlet: {'PERSON': Counter(), 'ORG': Counter()} for outlet in outlets}

    total_articles = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0 or i == total_articles - 1: # Log progress
             logger.info(f"Processing NER in article {i+1}/{total_articles}")

        outlet = row['outlet']
        text = row['processed_text'] # Using processed text - limitation!

        if not isinstance(text, str) or not text.strip():
            continue

        try:
            doc = NLP(text)
            for ent in doc.ents:
                entity_text = ent.text.strip()
                # Basic cleaning: skip very short entities or purely numeric ones
                if len(entity_text) > 2 and not entity_text.isdigit():
                    if ent.label_ == 'PERSON':
                        ner_persons_overall[entity_text] += 1
                        if outlet in ner_per_outlet:
                             ner_per_outlet[outlet]['PERSON'][entity_text] += 1
                    elif ent.label_ == 'ORG':
                        # Normalize common org variations if needed (e.g., 'BJP', 'Bharatiya Janata Party')
                        # For now, just count as found
                        ner_orgs_overall[entity_text] += 1
                        if outlet in ner_per_outlet:
                             ner_per_outlet[outlet]['ORG'][entity_text] += 1
        except Exception as e:
            logger.warning(f"spaCy NER error on article snippet: {text[:100]}... Error: {e}")

    # Get top N
    top_persons = ner_persons_overall.most_common(top_n)
    top_orgs = ner_orgs_overall.most_common(top_n)

    logger.info("NER analysis complete.")
    return top_persons, top_orgs, ner_per_outlet # Returning top N overall and full dict per outlet


def perform_topic_modeling(text_series, n_topics=config.N_TOPICS, n_top_words=10, model_type='lda'):
    """
    Performs Topic Modeling (LDA or NMF) on the text data.

    Args:
        text_series (pd.Series): Series containing preprocessed text documents.
        n_topics (int): The number of topics to extract.
        n_top_words (int): The number of top words to display for each topic.
        model_type (str): 'lda' or 'nmf'.

    Returns:
        tuple: (model, vectorizer, topics) where topics is a dict {topic_id: "word1 word2..."}
               Returns (None, None, None) on error or if text_series is empty.
    """
    logger.info(f"Performing Topic Modeling ({model_type.upper()}) with {n_topics} topics...")
    if text_series is None or text_series.empty:
        logger.warning("Input text series is empty for topic modeling.")
        return None, None, None

    # Ensure all entries are strings
    text_series = text_series.astype(str)

    # Create TF-IDF or Count Vectorizer (LDA often uses Counts, NMF often uses TF-IDF)
    if model_type == 'lda':
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=list(config.COMBINED_STOPWORDS), max_features=1000)
        dtm = vectorizer.fit_transform(text_series)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1) # Use multiple cores
    elif model_type == 'nmf':
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(config.COMBINED_STOPWORDS), max_features=1000)
        dtm = vectorizer.fit_transform(text_series)
        model = NMF(n_components=n_topics, random_state=42, max_iter=300, init='nndsvda') # NMF can be sensitive to init
    else:
        logger.error(f"Invalid model_type '{model_type}'. Choose 'lda' or 'nmf'.")
        return None, None, None

    logger.info("Fitting topic model...")
    try:
        model.fit(dtm)
    except ValueError as e:
         logger.error(f"Error fitting topic model. Often caused by empty vocabulary or documents after vectorization: {e}")
         return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error fitting topic model: {e}")
        return None, None, None
    logger.info("Topic model fitting complete.")


    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics[topic_idx] = " ".join(top_words)
        logger.info(f"Topic {topic_idx}: {topics[topic_idx]}")

    logger.info("Topic modeling complete.")
    return model, vectorizer, topics


def analyze_omission_proxy(df, omission_topics_keywords):
    """
    Basic proxy for omission analysis by checking keyword presence per outlet.
    NOTE: This is highly simplified and requires well-defined keywords.

    Args:
        df (pd.DataFrame): DataFrame with 'outlet' and 'processed_text'.
        omission_topics_keywords (dict): Dict mapping topic names to lists of keywords.

    Returns:
        dict: Dictionary mapping outlets to counters of topic mentions.
              {outlet: Counter(topic_name: mention_count)}
              Returns None if input is invalid or keywords are not defined.
    """
    logger.info("Performing Omission Analysis Proxy (Keyword Presence)...")
    if not omission_topics_keywords:
        logger.warning("OMISSION_TOPICS keywords not defined in config. Skipping omission analysis.")
        return None
    if df is None or not {'outlet', 'processed_text'}.issubset(df.columns):
        logger.error("DataFrame invalid or missing columns for omission analysis.")
        return None

    # Pre-process keywords (assuming text is lemmatized)
    processed_keywords = {}
    lemmatizer = WordNetLemmatizer()
    for topic, keywords in omission_topics_keywords.items():
         processed_keywords[topic] = [lemmatizer.lemmatize(kw.lower()) for kw in keywords]

    omission_counts = {outlet: Counter() for outlet in df['outlet'].unique()}

    for index, row in df.iterrows():
        outlet = row['outlet']
        text = row['processed_text']
        if not isinstance(text, str) or not text.strip():
            continue

        words_in_article = set(text.split()) # Use set for faster checking

        for topic, keywords in processed_keywords.items():
            # Check if *any* keyword for the topic is present
            if any(kw in words_in_article for kw in keywords):
                 if outlet in omission_counts:
                    omission_counts[outlet][topic] += 1

    logger.info("Omission analysis proxy complete.")
    return omission_counts

def perform_statistical_tests(df, alpha=config.STAT_ALPHA):
    """
    Performs statistical tests (ANOVA, T-tests) on sentiment scores.

    Args:
        df (pd.DataFrame): DataFrame with 'outlet' and 'vader_compound' columns.
        alpha (float): Significance level.

    Returns:
        dict: A dictionary containing test results and interpretations.
    """
    logger.info("Performing Statistical Tests on VADER Sentiment...")
    results = {}
    if df is None or not {'outlet', 'vader_compound'}.issubset(df.columns) or df['vader_compound'].isnull().all():
        logger.error("DataFrame invalid, missing columns, or all sentiment scores are null for statistical tests.")
        return results

    df_clean = df.dropna(subset=['vader_compound'])
    if len(df_clean) < 2:
        logger.warning("Not enough valid data points for statistical tests.")
        return results

    outlets = df_clean['outlet'].unique()

    # 1. ANOVA (Compare mean sentiment across all outlets)
    if len(outlets) > 1:
        sentiment_groups = [df_clean['vader_compound'][df_clean['outlet'] == outlet] for outlet in outlets]
        # Filter out groups with insufficient data for ANOVA
        sentiment_groups_filtered = [g for g in sentiment_groups if len(g) >= 2]
        if len(sentiment_groups_filtered) > 1:
             logger.info(f"Running ANOVA on sentiment across {len(sentiment_groups_filtered)} outlets...")
             try:
                 f_stat, p_value = f_oneway(*sentiment_groups_filtered)
                 interpretation = f"Significant difference (p < {alpha})" if p_value < alpha else f"No significant difference (p >= {alpha})"
                 results['ANOVA_sentiment_by_outlet'] = {
                     'f_statistic': f_stat,
                     'p_value': p_value,
                     'interpretation': f"ANOVA comparing mean VADER sentiment across outlets: {interpretation}."
                 }
                 logger.info(results['ANOVA_sentiment_by_outlet'])
             except Exception as e:
                  logger.error(f"Error during ANOVA: {e}")
        else:
             logger.warning(f"Could not run ANOVA: Need at least 2 outlets with >= 2 articles each.")

    # 2. Example T-test (e.g., comparing two specific outlets if needed)
    #    Let's compare the first two outlets with enough data as an example
    outlets_with_data = [outlet for outlet in outlets if len(df_clean[df_clean['outlet'] == outlet]) >= 2]
    if len(outlets_with_data) >= 2:
        outlet1_name = outlets_with_data[0]
        outlet2_name = outlets_with_data[1]
        group1 = df_clean['vader_compound'][df_clean['outlet'] == outlet1_name]
        group2 = df_clean['vader_compound'][df_clean['outlet'] == outlet2_name]

        logger.info(f"Running T-test comparing sentiment between '{outlet1_name}' and '{outlet2_name}'...")
        try:
            # Welch's T-test (doesn't assume equal variance)
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            interpretation = f"Significant difference (p < {alpha})" if p_value < alpha else f"No significant difference (p >= {alpha})"
            test_name = f"TTest_{outlet1_name}_vs_{outlet2_name}"
            results[test_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'interpretation': f"T-test comparing mean VADER sentiment ({outlet1_name} vs {outlet2_name}): {interpretation}."
            }
            logger.info(results[test_name])
        except Exception as e:
            logger.error(f"Error during T-test ({outlet1_name} vs {outlet2_name}): {e}")


    # 3. Chi-squared Test (Example: Ideological Term Category Presence)
    # This requires the ideological term counts per outlet
    term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)

    if term_counts_per_outlet:
         # Example: Compare presence of 'pro_bjp_modi' vs 'critical_bjp_modi' terms across outlets
         cat1 = 'pro_bjp_modi'
         cat2 = 'critical_bjp_modi'
         contingency_table_data = []
         valid_outlets_for_chi2 = []

         for outlet, categories in term_counts_per_outlet.items():
             count1 = sum(categories[cat1].values())
             count2 = sum(categories[cat2].values())
             # Only include outlets that have *some* mentions in either category for a meaningful test
             if count1 > 0 or count2 > 0:
                  contingency_table_data.append([count1, count2])
                  valid_outlets_for_chi2.append(outlet)

         if len(contingency_table_data) >= 2: # Need at least 2 outlets
            logger.info(f"Running Chi-squared test for '{cat1}' vs '{cat2}' presence across {len(valid_outlets_for_chi2)} outlets...")
            contingency_table = np.array(contingency_table_data)
            try:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                interpretation = f"Significant difference (p < {alpha})" if p < alpha else f"No significant difference (p >= {alpha})"
                results['Chi2_BJP_Terms_by_Outlet'] = {
                    'chi2_statistic': chi2,
                    'p_value': p,
                    'degrees_of_freedom': dof,
                    # 'expected_freq': expected.tolist(), # Can be large
                    'interpretation': f"Chi-squared test comparing distribution of '{cat1}' vs '{cat2}' terms across outlets: {interpretation}."
                }
                logger.info(results['Chi2_BJP_Terms_by_Outlet'])
            except ValueError as e:
                 logger.error(f"Error during Chi-squared test (often due to low expected frequencies): {e}")
            except Exception as e:
                 logger.error(f"Unexpected error during Chi-squared test: {e}")
         else:
            logger.warning(f"Could not run Chi-squared test for '{cat1}' vs '{cat2}': Need counts from at least 2 outlets.")

    logger.info("Statistical tests complete.")
    return results


def save_plot(plt_object, filename):
    """Saves the current matplotlib plot to the output directory."""
    try:
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        plt_object.savefig(filepath)
        logger.info(f"Plot saved to {filepath}")
        plt_object.close() # Close the plot to free memory
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")
        plt_object.close() # Still try to close it


def generate_visualizations(df, avg_sentiment_outlet, most_common_overall, word_freq_by_outlet, ner_persons_overall, ner_orgs_overall, avg_party_sentiment, topics, omission_counts):
    """Generates and saves various plots."""
    logger.info("Generating visualizations...")

    # 1. Sentiment Distribution by Outlet
    if df is not None and 'vader_compound' in df.columns and not df['vader_compound'].isnull().all():
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df, x='outlet', y='vader_compound', palette='viridis')
        plt.title('VADER Sentiment Score Distribution by Outlet', fontsize=16)
        plt.ylabel('VADER Compound Score', fontsize=12)
        plt.xlabel('Outlet', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot(plt, 'vader_sentiment_dist_outlet.png')

        plt.figure(figsize=(12, 7))
        if not avg_sentiment_outlet.empty:
            avg_sentiment_outlet.plot(kind='bar', color=sns.color_palette("viridis", len(avg_sentiment_outlet)))
            plt.title('Average VADER Sentiment Score by Outlet', fontsize=16)
            plt.ylabel('Average VADER Compound Score', fontsize=12)
            plt.xlabel('')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            plt.tight_layout()
            save_plot(plt, 'vader_avg_sentiment_outlet.png')
        else:
             logger.warning("Skipping average sentiment plot - no data.")


    # 2. Overall Word Cloud
    if most_common_overall:
        try:
            wc = WordCloud(width=1200, height=600, background_color='white',
                           max_words=100, colormap='magma').generate_from_frequencies(dict(most_common_overall))
            plt.figure(figsize=(15, 7))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Overall Top Words (Excluding Stopwords)', fontsize=16)
            plt.tight_layout()
            save_plot(plt, 'vader_wordcloud_overall.png')
        except Exception as e:
             logger.error(f"Error generating overall word cloud: {e}")
    else:
        logger.warning("Skipping overall word cloud - no frequency data.")

    # 3. Word Clouds per Outlet (Optional - can generate many files)
    # Consider enabling only if needed, as it can be slow and create many files
    # if word_freq_by_outlet:
    #     for outlet, freq_dist in word_freq_by_outlet.items():
    #         if freq_dist:
    #             try:
    #                 wc = WordCloud(width=800, height=400, background_color='white', max_words=75, colormap='plasma').generate_from_frequencies(dict(freq_dist))
    #                 plt.figure(figsize=(10, 5))
    #                 plt.imshow(wc, interpolation='bilinear')
    #                 plt.axis('off')
    #                 plt.title(f'Top Words for {outlet}', fontsize=14)
    #                 plt.tight_layout()
    #                 safe_outlet_name = re.sub(r'\W+', '_', outlet) # Make filename safe
    #                 save_plot(plt, f'vader_wordcloud_outlet_{safe_outlet_name}.png')
    #             except Exception as e:
    #                 logger.error(f"Error generating word cloud for outlet {outlet}: {e}")


    # 4. Top Entities Plots
    if ner_persons_overall:
        ner_df = pd.DataFrame(ner_persons_overall, columns=['Person', 'Count'])
        plt.figure(figsize=(12, 8))
        sns.barplot(data=ner_df, y='Person', x='Count', palette='coolwarm')
        plt.title(f'Top {len(ner_df)} PERSON Entities Mentioned Overall', fontsize=16)
        plt.tight_layout()
        save_plot(plt, 'vader_ner_top_persons.png')

    if ner_orgs_overall:
        ner_df = pd.DataFrame(ner_orgs_overall, columns=['Organization', 'Count'])
        plt.figure(figsize=(12, 8))
        sns.barplot(data=ner_df, y='Organization', x='Count', palette='coolwarm')
        plt.title(f'Top {len(ner_df)} ORGANIZATION Entities Mentioned Overall', fontsize=16)
        plt.tight_layout()
        save_plot(plt, 'vader_ner_top_orgs.png')

    # 5. Party Sentiment Plot
    if avg_party_sentiment:
         party_sentiment_df = pd.DataFrame.from_dict(avg_party_sentiment, orient='index')
         party_sentiment_df = party_sentiment_df.sort_values('average_vader_compound', ascending=False)
         if not party_sentiment_df.empty:
             plt.figure(figsize=(10, 6))
             sns.barplot(x=party_sentiment_df.index, y=party_sentiment_df['average_vader_compound'], palette='crest')
             plt.title('Average VADER Sentiment in Sentences Mentioning Parties', fontsize=14)
             plt.xlabel('Party', fontsize=12)
             plt.ylabel('Average VADER Compound Score (Sentence Level)', fontsize=12)
             plt.xticks(rotation=0)
             plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
             plt.tight_layout()
             save_plot(plt, 'vader_avg_sentiment_per_party_sentence.png')

    # 6. Omission Proxy Plot (Example: Bar chart of topic mentions per outlet)
    if omission_counts:
        omission_df = pd.DataFrame.from_dict(omission_counts, orient='index').fillna(0)
        if not omission_df.empty:
             omission_df.plot(kind='bar', figsize=(15, 8), colormap='tab20')
             plt.title('Proxy Omission Analysis: Topic Keyword Mentions per Outlet', fontsize=16)
             plt.xlabel('Outlet', fontsize=12)
             plt.ylabel('Number of Articles Mentioning Topic Keywords', fontsize=12)
             plt.xticks(rotation=45, ha='right')
             plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
             plt.tight_layout()
             save_plot(plt, 'vader_omission_proxy_mentions.png')

    logger.info("Visualizations generation complete.")


def save_results(df, avg_sentiment_outlet, most_common_overall, term_counts_overall, term_counts_per_outlet, ner_persons_overall, ner_orgs_overall, ner_per_outlet, avg_party_sentiment, topics_lda, topics_nmf, stat_results, omission_counts):
    """Saves numerical results and logs to files."""
    logger.info("Saving analysis results...")

    # Save main dataframe with sentiment
    if df is not None:
        try:
            df.to_csv(os.path.join(config.OUTPUT_DIR, 'vader_articles_with_sentiment.csv'), index=False, encoding='utf-8')
            logger.info("Saved main dataframe with VADER scores.")
        except Exception as e:
            logger.error(f"Error saving main dataframe: {e}")

    # Save aggregate sentiment
    if not avg_sentiment_outlet.empty:
        try:
            avg_sentiment_outlet.to_csv(os.path.join(config.OUTPUT_DIR, 'vader_avg_sentiment_outlet.csv'), header=['average_vader_compound'])
            logger.info("Saved average VADER sentiment per outlet.")
        except Exception as e:
            logger.error(f"Error saving average outlet sentiment: {e}")

    # Save top words
    if most_common_overall:
        try:
            pd.DataFrame(most_common_overall, columns=['word', 'count']).to_csv(os.path.join(config.OUTPUT_DIR, 'vader_top_words_overall.csv'), index=False)
            logger.info("Saved overall top words.")
        except Exception as e:
            logger.error(f"Error saving top words: {e}")

    # Save ideological term counts (using JSON for nested structure)
    results_to_save = {
        'term_counts_overall': {cat: dict(cnt) for cat, cnt in term_counts_overall.items()} if term_counts_overall else {},
        'term_counts_per_outlet': {
            outlet: {cat: dict(cnt) for cat, cnt in cats.items()}
            for outlet, cats in term_counts_per_outlet.items()
        } if term_counts_per_outlet else {}
    }
    try:
        with open(os.path.join(config.OUTPUT_DIR, 'vader_ideological_term_counts.json'), 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4)
        logger.info("Saved ideological term counts (overall and per outlet).")
    except Exception as e:
        logger.error(f"Error saving ideological term counts: {e}")


    # Save NER results
    ner_results = {
        'top_persons_overall': ner_persons_overall if ner_persons_overall else [],
        'top_orgs_overall': ner_orgs_overall if ner_orgs_overall else [],
        'ner_per_outlet': {
             outlet: {etype: dict(cnt) for etype, cnt in types.items()}
             for outlet, types in ner_per_outlet.items()
        } if ner_per_outlet else {}
    }
    try:
        with open(os.path.join(config.OUTPUT_DIR, 'vader_ner_results.json'), 'w', encoding='utf-8') as f:
            json.dump(ner_results, f, indent=4)
        logger.info("Saved NER results.")
    except Exception as e:
        logger.error(f"Error saving NER results: {e}")

    # Save party sentiment
    if avg_party_sentiment:
        try:
            pd.DataFrame.from_dict(avg_party_sentiment, orient='index').to_csv(os.path.join(config.OUTPUT_DIR, 'vader_avg_party_sentiment_sentence.csv'))
            logger.info("Saved average party sentiment (sentence level).")
        except Exception as e:
            logger.error(f"Error saving party sentiment: {e}")

    # Save Topic Modeling results
    topic_results = {
        'lda_topics': topics_lda if topics_lda else {},
        'nmf_topics': topics_nmf if topics_nmf else {}
    }
    try:
        with open(os.path.join(config.OUTPUT_DIR, 'vader_topic_modeling_results.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_results, f, indent=4)
        logger.info("Saved topic modeling results.")
    except Exception as e:
        logger.error(f"Error saving topic modeling results: {e}")

    # Save Statistical Test results
    if stat_results:
        try:
            with open(os.path.join(config.OUTPUT_DIR, 'vader_statistical_test_results.json'), 'w', encoding='utf-8') as f:
                # Convert numpy types to standard Python types for JSON serialization
                 serializable_results = {}
                 for key, value_dict in stat_results.items():
                      serializable_results[key] = {
                           k: (float(v) if isinstance(v, (np.float_, np.int_)) else v)
                           for k, v in value_dict.items()
                           }
                 json.dump(serializable_results, f, indent=4)
            logger.info("Saved statistical test results.")
        except Exception as e:
            logger.error(f"Error saving statistical test results: {e}")

     # Save Omission Proxy results
    if omission_counts:
         try:
             omission_df = pd.DataFrame.from_dict(omission_counts, orient='index').fillna(0)
             omission_df.to_csv(os.path.join(config.OUTPUT_DIR, 'vader_omission_proxy_counts.csv'))
             logger.info("Saved omission proxy counts.")
         except Exception as e:
             logger.error(f"Error saving omission proxy counts: {e}")


    logger.info(f"All available results saved to {config.OUTPUT_DIR}")
    logger.info(f"Detailed logs available in {config.ANALYSIS_VADER_LOG_FILE}")


# === Main Execution ===
if __name__ == "__main__":
    logger.info("--- Starting VADER Analysis Pipeline ---")

    # 1. Load Data
    df = load_data(config.PREPROCESSED_ARTICLES_CSV)

    if df is not None and not df.empty:
        # 2. Perform Sentiment Analysis
        df = perform_vader_sentiment_analysis(df)
        avg_sentiment_outlet = df.groupby('outlet')['vader_compound'].mean().sort_values() if 'vader_compound' in df.columns else pd.Series(dtype=float)

        # 3. Perform Word Frequency Analysis
        most_common_overall = analyze_word_frequency(df['processed_text'])
        # Word freq per outlet (optional, can be slow)
        word_freq_by_outlet = {}
        # for outlet_name in df['outlet'].unique():
        #      outlet_texts = df[df['outlet'] == outlet_name]['processed_text']
        #      word_freq_by_outlet[outlet_name] = analyze_word_frequency(outlet_texts)

        # 4. Analyze Ideological Term Frequency
        term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)

        # 5. Analyze Sentiment Towards Parties (Sentence Level)
        avg_party_sentiment = analyze_sentiment_towards_parties_vader(df, config.PARTY_TERMS)

        # 6. Perform NER
        # Ideally, load original text here if available and needed for better NER
        ner_persons_overall, ner_orgs_overall, ner_per_outlet = perform_ner(df)

        # 7. Perform Topic Modeling (LDA and NMF)
        lda_model, lda_vectorizer, topics_lda = perform_topic_modeling(df['processed_text'], model_type='lda')
        nmf_model, nmf_vectorizer, topics_nmf = perform_topic_modeling(df['processed_text'], model_type='nmf')

        # 8. Perform Omission Analysis Proxy
        omission_counts = analyze_omission_proxy(df, config.OMISSION_TOPICS)

        # 9. Perform Statistical Tests
        stat_results = perform_statistical_tests(df)

        # 10. Generate Visualizations
        generate_visualizations(df, avg_sentiment_outlet, most_common_overall, word_freq_by_outlet, ner_persons_overall, ner_orgs_overall, avg_party_sentiment, topics_lda, omission_counts) # Passing LDA topics for now

        # 11. Save Results
        save_results(df, avg_sentiment_outlet, most_common_overall, term_counts_overall, term_counts_per_outlet, ner_persons_overall, ner_orgs_overall, ner_per_outlet, avg_party_sentiment, topics_lda, topics_nmf, stat_results, omission_counts)

    else:
        logger.error("Failed to load or process data. Exiting VADER analysis.")

    logger.info("--- Finished VADER Analysis Pipeline ---")