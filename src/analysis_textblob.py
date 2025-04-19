import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob # Use TextBlob
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
# Use a different log file name
log_handler = logging.FileHandler(config.ANALYSIS_TEXTBLOB_LOG_FILE)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__ + "_textblob") # Use a unique logger name
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
# Optional: Add console handler
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# logger.addHandler(console_handler)


# --- Download necessary NLTK data ---
def download_nltk_data():
    """Downloads required NLTK models if they don't exist."""
    # TextBlob uses NLTK models implicitly, ensure they are present
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'brown'] # TextBlob needs tagger/brown
    for item in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}' if item in ['stopwords', 'wordnet', 'brown'] else f'taggers/{item}')
        except (LookupError, nltk.downloader.DownloadError):
            logger.info(f"NLTK data '{item}' for TextBlob not found or outdated. Downloading...")
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


# === Helper & Analysis Functions (Many are identical to VADER version) ===

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

def perform_textblob_sentiment_analysis(df):
    """
    Performs sentiment analysis on the 'processed_text' column using TextBlob
    and adds 'polarity' and 'subjectivity' columns.
    """
    logger.info("Performing TextBlob Sentiment Analysis...")
    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid or missing 'processed_text' column for TextBlob analysis.")
        return df

    polarities = []
    subjectivities = []
    for text in df['processed_text']:
        try:
            blob = TextBlob(str(text))
            polarities.append(blob.sentiment.polarity)
            subjectivities.append(blob.sentiment.subjectivity)
        except Exception as e:
            logger.warning(f"TextBlob error processing text snippet '{str(text)[:50]}...': {e}")
            polarities.append(None) # Append None or 0 for problematic texts
            subjectivities.append(None)

    df['polarity'] = polarities
    df['subjectivity'] = subjectivities
    # Drop rows where sentiment calculation failed
    df.dropna(subset=['polarity', 'subjectivity'], inplace=True)
    logger.info("TextBlob sentiment analysis complete.")
    return df

def analyze_word_frequency(text_series, top_n=config.TOP_N_WORDS, stopwords_set=config.COMBINED_STOPWORDS):
    """Identical to VADER version"""
    logger.info(f"Performing Word Frequency Analysis (Top {top_n})...")
    if text_series is None or text_series.empty:
        logger.warning("Input text series is empty for word frequency analysis.")
        return []
    try:
        all_text = ' '.join(text_series.astype(str).tolist())
    except Exception as e:
        logger.error(f"Error combining text series for word frequency: {e}")
        return []
    if not all_text.strip():
        logger.warning("No text content found for word frequency analysis after joining.")
        return []
    tokens = all_text.split()
    filtered_tokens = [word for word in tokens if len(word) > 2 and word not in stopwords_set]
    word_counts = Counter(filtered_tokens)
    most_common = word_counts.most_common(top_n)
    logger.info(f"Word frequency analysis complete. Found {len(word_counts)} unique words (after filtering).")
    return most_common

def analyze_ideological_term_frequency(df, term_dict):
    """Identical to VADER version"""
    logger.info("Analyzing Ideological Term Frequency...")
    if df is None or not {'outlet', 'processed_text'}.issubset(df.columns):
        logger.error("DataFrame invalid or missing columns for ideological term analysis.")
        return None, None
    term_counts_overall = {category: Counter() for category in term_dict}
    outlets = df['outlet'].unique()
    term_counts_per_outlet = {outlet: {category: Counter() for category in term_dict} for outlet in outlets}
    processed_term_dict = {}
    lemmatizer = WordNetLemmatizer()
    for category, terms in term_dict.items():
         processed_term_dict[category] = [lemmatizer.lemmatize(term.lower()) for term in terms]

    for index, row in df.iterrows():
        outlet = row['outlet']
        text = row['processed_text']
        if not isinstance(text, str) or not text.strip(): continue
        words_in_article = text.split()
        article_word_counts = Counter(words_in_article)
        for category, processed_terms in processed_term_dict.items():
            original_terms = term_dict[category]
            for i, term in enumerate(processed_terms):
                original_term = original_terms[i]
                count = article_word_counts[term]
                if count > 0:
                    term_counts_overall[category][original_term] += count
                    if outlet in term_counts_per_outlet:
                         term_counts_per_outlet[outlet][category][original_term] += count
    logger.info("Ideological term frequency analysis complete.")
    return term_counts_overall, term_counts_per_outlet

def analyze_sentiment_towards_parties_textblob(df, party_term_dict):
    """
    Calculates average TextBlob polarity/subjectivity of sentences mentioning specific parties.
    """
    logger.info("Analyzing Sentiment Towards Political Parties (TextBlob - Sentence Level)...")
    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid or missing 'processed_text' for party sentiment analysis.")
        return None

    party_sentiments = {party: {'total_polarity': 0.0, 'total_subjectivity': 0.0, 'sentence_count': 0} for party in party_term_dict}

    processed_party_term_dict = {}
    lemmatizer = WordNetLemmatizer()
    for party, terms in party_term_dict.items():
         processed_party_term_dict[party] = [lemmatizer.lemmatize(term.lower()) for term in terms]

    total_articles = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0 or i == total_articles - 1:
             logger.info(f"Processing party sentiment in article {i+1}/{total_articles}")

        text = row['processed_text']
        if not isinstance(text, str) or not text.strip():
            continue

        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Could not sentence-tokenize article snippet: {text[:100]}... Error: {e}")
            continue

        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())

            for party, processed_terms in processed_party_term_dict.items():
                if any(term in sentence_words for term in processed_terms):
                    try:
                        blob = TextBlob(sentence) # Analyze the sentence
                        party_sentiments[party]['total_polarity'] += blob.sentiment.polarity
                        party_sentiments[party]['total_subjectivity'] += blob.sentiment.subjectivity
                        party_sentiments[party]['sentence_count'] += 1
                    except Exception as e:
                         logger.warning(f"TextBlob error on sentence: {sentence[:50]}... Error: {e}")

    # Calculate average sentiment
    avg_party_sentiment = {}
    for party, data in party_sentiments.items():
        if data['sentence_count'] > 0:
            avg_pol = data['total_polarity'] / data['sentence_count']
            avg_subj = data['total_subjectivity'] / data['sentence_count']
            avg_party_sentiment[party] = {
                'average_polarity': avg_pol,
                'average_subjectivity': avg_subj,
                'sentence_count': data['sentence_count']
            }
        else:
             avg_party_sentiment[party] = {
                'average_polarity': 0.0,
                'average_subjectivity': 0.0,
                'sentence_count': 0
            }
    logger.info("Party sentiment analysis (TextBlob - Sentence Level) complete.")
    return avg_party_sentiment


def perform_ner(df, top_n=config.TOP_N_ENTITIES):
    """Identical to VADER version"""
    logger.info("Performing Named Entity Recognition (NER)...")
    if NLP is None:
        logger.error("spaCy model not loaded. Cannot perform NER.")
        return None, None, None
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
        if (i + 1) % 10 == 0 or i == total_articles - 1:
             logger.info(f"Processing NER in article {i+1}/{total_articles}")
        outlet = row['outlet']
        text = row['processed_text']
        if not isinstance(text, str) or not text.strip(): continue
        try:
            doc = NLP(text)
            for ent in doc.ents:
                entity_text = ent.text.strip()
                if len(entity_text) > 2 and not entity_text.isdigit():
                    if ent.label_ == 'PERSON':
                        ner_persons_overall[entity_text] += 1
                        if outlet in ner_per_outlet: ner_per_outlet[outlet]['PERSON'][entity_text] += 1
                    elif ent.label_ == 'ORG':
                        ner_orgs_overall[entity_text] += 1
                        if outlet in ner_per_outlet: ner_per_outlet[outlet]['ORG'][entity_text] += 1
        except Exception as e:
            logger.warning(f"spaCy NER error on article snippet: {text[:100]}... Error: {e}")

    top_persons = ner_persons_overall.most_common(top_n)
    top_orgs = ner_orgs_overall.most_common(top_n)
    logger.info("NER analysis complete.")
    return top_persons, top_orgs, ner_per_outlet


def perform_topic_modeling(text_series, n_topics=config.N_TOPICS, n_top_words=10, model_type='lda'):
    """Identical to VADER version"""
    logger.info(f"Performing Topic Modeling ({model_type.upper()}) with {n_topics} topics...")
    if text_series is None or text_series.empty:
        logger.warning("Input text series is empty for topic modeling.")
        return None, None, None
    text_series = text_series.astype(str)
    if model_type == 'lda':
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=list(config.COMBINED_STOPWORDS), max_features=1000)
        dtm = vectorizer.fit_transform(text_series)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
    elif model_type == 'nmf':
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(config.COMBINED_STOPWORDS), max_features=1000)
        dtm = vectorizer.fit_transform(text_series)
        model = NMF(n_components=n_topics, random_state=42, max_iter=300, init='nndsvda')
    else:
        logger.error(f"Invalid model_type '{model_type}'. Choose 'lda' or 'nmf'.")
        return None, None, None
    logger.info("Fitting topic model...")
    try:
        model.fit(dtm)
    except ValueError as e:
         logger.error(f"Error fitting topic model: {e}")
         return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error fitting topic model: {e}")
        return None, None, None
    logger.info("Topic model fitting complete.")
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
    """Identical to VADER version"""
    logger.info("Performing Omission Analysis Proxy (Keyword Presence)...")
    if not omission_topics_keywords:
        logger.warning("OMISSION_TOPICS keywords not defined in config. Skipping omission analysis.")
        return None
    if df is None or not {'outlet', 'processed_text'}.issubset(df.columns):
        logger.error("DataFrame invalid or missing columns for omission analysis.")
        return None
    processed_keywords = {}
    lemmatizer = WordNetLemmatizer()
    for topic, keywords in omission_topics_keywords.items():
         processed_keywords[topic] = [lemmatizer.lemmatize(kw.lower()) for kw in keywords]
    omission_counts = {outlet: Counter() for outlet in df['outlet'].unique()}
    for index, row in df.iterrows():
        outlet = row['outlet']
        text = row['processed_text']
        if not isinstance(text, str) or not text.strip(): continue
        words_in_article = set(text.split())
        for topic, keywords in processed_keywords.items():
            if any(kw in words_in_article for kw in keywords):
                 if outlet in omission_counts: omission_counts[outlet][topic] += 1
    logger.info("Omission analysis proxy complete.")
    return omission_counts


def perform_statistical_tests(df, alpha=config.STAT_ALPHA):
    """
    Performs statistical tests (ANOVA, T-tests) on TextBlob polarity scores.
    """
    logger.info("Performing Statistical Tests on TextBlob Polarity...")
    results = {}
    # Use 'polarity' column from TextBlob
    sentiment_col = 'polarity'
    if df is None or not {'outlet', sentiment_col}.issubset(df.columns) or df[sentiment_col].isnull().all():
        logger.error(f"DataFrame invalid, missing columns, or all {sentiment_col} scores are null for statistical tests.")
        return results

    df_clean = df.dropna(subset=[sentiment_col])
    if len(df_clean) < 2:
        logger.warning("Not enough valid data points for statistical tests.")
        return results

    outlets = df_clean['outlet'].unique()

    # 1. ANOVA (Compare mean polarity across all outlets)
    if len(outlets) > 1:
        sentiment_groups = [df_clean[sentiment_col][df_clean['outlet'] == outlet] for outlet in outlets]
        sentiment_groups_filtered = [g for g in sentiment_groups if len(g) >= 2]
        if len(sentiment_groups_filtered) > 1:
             logger.info(f"Running ANOVA on polarity across {len(sentiment_groups_filtered)} outlets...")
             try:
                 f_stat, p_value = f_oneway(*sentiment_groups_filtered)
                 interpretation = f"Significant difference (p < {alpha})" if p_value < alpha else f"No significant difference (p >= {alpha})"
                 results['ANOVA_polarity_by_outlet'] = {
                     'f_statistic': f_stat,
                     'p_value': p_value,
                     'interpretation': f"ANOVA comparing mean TextBlob polarity across outlets: {interpretation}."
                 }
                 logger.info(results['ANOVA_polarity_by_outlet'])
             except Exception as e:
                  logger.error(f"Error during ANOVA: {e}")
        else:
             logger.warning(f"Could not run ANOVA: Need at least 2 outlets with >= 2 articles each.")


    # 2. Example T-test (Comparing two specific outlets)
    outlets_with_data = [outlet for outlet in outlets if len(df_clean[df_clean['outlet'] == outlet]) >= 2]
    if len(outlets_with_data) >= 2:
        outlet1_name = outlets_with_data[0]
        outlet2_name = outlets_with_data[1]
        group1 = df_clean[sentiment_col][df_clean['outlet'] == outlet1_name]
        group2 = df_clean[sentiment_col][df_clean['outlet'] == outlet2_name]
        logger.info(f"Running T-test comparing polarity between '{outlet1_name}' and '{outlet2_name}'...")
        try:
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            interpretation = f"Significant difference (p < {alpha})" if p_value < alpha else f"No significant difference (p >= {alpha})"
            test_name = f"TTest_polarity_{outlet1_name}_vs_{outlet2_name}"
            results[test_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'interpretation': f"T-test comparing mean TextBlob polarity ({outlet1_name} vs {outlet2_name}): {interpretation}."
            }
            logger.info(results[test_name])
        except Exception as e:
            logger.error(f"Error during T-test ({outlet1_name} vs {outlet2_name}): {e}")

    # 3. Chi-squared Test (Example: Ideological Term Category Presence) - Identical logic to VADER version
    term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)
    if term_counts_per_outlet:
         cat1 = 'pro_bjp_modi'
         cat2 = 'critical_bjp_modi'
         contingency_table_data = []
         valid_outlets_for_chi2 = []
         for outlet, categories in term_counts_per_outlet.items():
             count1 = sum(categories[cat1].values())
             count2 = sum(categories[cat2].values())
             if count1 > 0 or count2 > 0:
                  contingency_table_data.append([count1, count2])
                  valid_outlets_for_chi2.append(outlet)
         if len(contingency_table_data) >= 2:
            logger.info(f"Running Chi-squared test for '{cat1}' vs '{cat2}' presence across {len(valid_outlets_for_chi2)} outlets...")
            contingency_table = np.array(contingency_table_data)
            try:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                interpretation = f"Significant difference (p < {alpha})" if p < alpha else f"No significant difference (p >= {alpha})"
                results['Chi2_BJP_Terms_by_Outlet'] = { # Keep key consistent for comparison
                    'chi2_statistic': chi2, 'p_value': p, 'degrees_of_freedom': dof,
                    'interpretation': f"Chi-squared test comparing distribution of '{cat1}' vs '{cat2}' terms across outlets: {interpretation}."
                }
                logger.info(results['Chi2_BJP_Terms_by_Outlet'])
            except ValueError as e: logger.error(f"Error during Chi-squared test: {e}")
            except Exception as e: logger.error(f"Unexpected error during Chi-squared test: {e}")
         else: logger.warning(f"Could not run Chi-squared test for '{cat1}' vs '{cat2}': Need counts from at least 2 outlets.")

    logger.info("Statistical tests complete.")
    return results


def save_plot(plt_object, filename):
    """Saves the current matplotlib plot to the output directory."""
    try:
        # Add 'textblob_' prefix to filename
        filepath = os.path.join(config.OUTPUT_DIR, f"textblob_{filename}")
        plt_object.savefig(filepath)
        logger.info(f"Plot saved to {filepath}")
        plt_object.close()
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")
        plt_object.close()

def generate_visualizations(df, avg_sentiment_outlet, most_common_overall, word_freq_by_outlet, ner_persons_overall, ner_orgs_overall, avg_party_sentiment, topics, omission_counts):
    """Generates and saves various plots, adapted for TextBlob."""
    logger.info("Generating visualizations (TextBlob)...")
    sentiment_col = 'polarity' # Use polarity for main sentiment plots

    # 1. Polarity Distribution by Outlet
    if df is not None and sentiment_col in df.columns and not df[sentiment_col].isnull().all():
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df, x='outlet', y=sentiment_col, palette='viridis')
        plt.title('TextBlob Polarity Score Distribution by Outlet', fontsize=16)
        plt.ylabel('TextBlob Polarity Score', fontsize=12)
        plt.xlabel('Outlet', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot(plt, 'polarity_dist_outlet.png')

        plt.figure(figsize=(12, 7))
        if not avg_sentiment_outlet.empty:
            avg_sentiment_outlet.plot(kind='bar', color=sns.color_palette("viridis", len(avg_sentiment_outlet)))
            plt.title('Average TextBlob Polarity Score by Outlet', fontsize=16)
            plt.ylabel('Average TextBlob Polarity Score', fontsize=12)
            plt.xlabel('')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            plt.tight_layout()
            save_plot(plt, 'avg_polarity_outlet.png')
        else:
             logger.warning("Skipping average polarity plot - no data.")

        # Optional: Subjectivity plot
        if 'subjectivity' in df.columns and not df['subjectivity'].isnull().all():
             plt.figure(figsize=(14, 8))
             sns.boxplot(data=df, x='outlet', y='subjectivity', palette='magma')
             plt.title('TextBlob Subjectivity Score Distribution by Outlet', fontsize=16)
             plt.ylabel('TextBlob Subjectivity Score', fontsize=12)
             plt.xlabel('Outlet', fontsize=12)
             plt.xticks(rotation=45, ha='right')
             plt.tight_layout()
             save_plot(plt, 'subjectivity_dist_outlet.png')


    # 2. Overall Word Cloud - Identical to VADER version
    if most_common_overall:
        try:
            wc = WordCloud(width=1200, height=600, background_color='white', max_words=100, colormap='magma').generate_from_frequencies(dict(most_common_overall))
            plt.figure(figsize=(15, 7))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Overall Top Words (Excluding Stopwords)', fontsize=16)
            plt.tight_layout()
            save_plot(plt, 'wordcloud_overall.png') # Keep name generic? Or add prefix? Let's add prefix
        except Exception as e: logger.error(f"Error generating overall word cloud: {e}")
    else: logger.warning("Skipping overall word cloud - no frequency data.")


    # 3. Top Entities Plots - Identical to VADER version
    if ner_persons_overall:
        ner_df = pd.DataFrame(ner_persons_overall, columns=['Person', 'Count'])
        plt.figure(figsize=(12, 8)); sns.barplot(data=ner_df, y='Person', x='Count', palette='coolwarm')
        plt.title(f'Top {len(ner_df)} PERSON Entities Mentioned Overall', fontsize=16); plt.tight_layout()
        save_plot(plt, 'ner_top_persons.png')
    if ner_orgs_overall:
        ner_df = pd.DataFrame(ner_orgs_overall, columns=['Organization', 'Count'])
        plt.figure(figsize=(12, 8)); sns.barplot(data=ner_df, y='Organization', x='Count', palette='coolwarm')
        plt.title(f'Top {len(ner_df)} ORGANIZATION Entities Mentioned Overall', fontsize=16); plt.tight_layout()
        save_plot(plt, 'ner_top_orgs.png')

    # 4. Party Sentiment Plot (Polarity)
    if avg_party_sentiment:
         party_sentiment_df = pd.DataFrame.from_dict(avg_party_sentiment, orient='index')
         party_sentiment_df = party_sentiment_df.sort_values('average_polarity', ascending=False)
         if not party_sentiment_df.empty:
             plt.figure(figsize=(10, 6))
             sns.barplot(x=party_sentiment_df.index, y=party_sentiment_df['average_polarity'], palette='crest')
             plt.title('Average TextBlob Polarity in Sentences Mentioning Parties', fontsize=14)
             plt.xlabel('Party', fontsize=12); plt.ylabel('Average Polarity (Sentence Level)', fontsize=12)
             plt.xticks(rotation=0); plt.axhline(0, color='grey', linewidth=0.8, linestyle='--'); plt.tight_layout()
             save_plot(plt, 'avg_polarity_per_party_sentence.png')

    # 5. Omission Proxy Plot - Identical logic to VADER version
    if omission_counts:
        omission_df = pd.DataFrame.from_dict(omission_counts, orient='index').fillna(0)
        if not omission_df.empty:
             omission_df.plot(kind='bar', figsize=(15, 8), colormap='tab20')
             plt.title('Proxy Omission Analysis: Topic Keyword Mentions per Outlet', fontsize=16)
             plt.xlabel('Outlet', fontsize=12); plt.ylabel('Number of Articles Mentioning Topic Keywords', fontsize=12)
             plt.xticks(rotation=45, ha='right'); plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
             save_plot(plt, 'omission_proxy_mentions.png')

    logger.info("Visualizations generation complete (TextBlob).")

def save_results(df, avg_sentiment_outlet, most_common_overall, term_counts_overall, term_counts_per_outlet, ner_persons_overall, ner_orgs_overall, ner_per_outlet, avg_party_sentiment, topics_lda, topics_nmf, stat_results, omission_counts):
    """Saves numerical results and logs to files, adding 'textblob_' prefix."""
    logger.info("Saving analysis results (TextBlob)...")
    prefix = "textblob_"

    # Save main dataframe with sentiment
    if df is not None:
        try:
            df.to_csv(os.path.join(config.OUTPUT_DIR, f'{prefix}articles_with_sentiment.csv'), index=False, encoding='utf-8')
            logger.info("Saved main dataframe with TextBlob scores.")
        except Exception as e: logger.error(f"Error saving main dataframe: {e}")

    # Save aggregate sentiment (polarity)
    if not avg_sentiment_outlet.empty:
        try:
            avg_sentiment_outlet.to_csv(os.path.join(config.OUTPUT_DIR, f'{prefix}avg_polarity_outlet.csv'), header=['average_polarity'])
            logger.info("Saved average TextBlob polarity per outlet.")
        except Exception as e: logger.error(f"Error saving average outlet polarity: {e}")

    # Save top words
    if most_common_overall:
        try:
            pd.DataFrame(most_common_overall, columns=['word', 'count']).to_csv(os.path.join(config.OUTPUT_DIR, f'{prefix}top_words_overall.csv'), index=False)
            logger.info("Saved overall top words.")
        except Exception as e: logger.error(f"Error saving top words: {e}")

    # Save ideological term counts (JSON) - Keep structure same as VADER for comparison
    results_to_save = {
        'term_counts_overall': {cat: dict(cnt) for cat, cnt in term_counts_overall.items()} if term_counts_overall else {},
        'term_counts_per_outlet': {outlet: {cat: dict(cnt) for cat, cnt in cats.items()} for outlet, cats in term_counts_per_outlet.items()} if term_counts_per_outlet else {}
    }
    try:
        with open(os.path.join(config.OUTPUT_DIR, f'{prefix}ideological_term_counts.json'), 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4)
        logger.info("Saved ideological term counts.")
    except Exception as e: logger.error(f"Error saving ideological term counts: {e}")

    # Save NER results (JSON) - Keep structure same as VADER
    ner_results = {
        'top_persons_overall': ner_persons_overall if ner_persons_overall else [],
        'top_orgs_overall': ner_orgs_overall if ner_orgs_overall else [],
        'ner_per_outlet': {outlet: {etype: dict(cnt) for etype, cnt in types.items()} for outlet, types in ner_per_outlet.items()} if ner_per_outlet else {}
    }
    try:
        with open(os.path.join(config.OUTPUT_DIR, f'{prefix}ner_results.json'), 'w', encoding='utf-8') as f:
            json.dump(ner_results, f, indent=4)
        logger.info("Saved NER results.")
    except Exception as e: logger.error(f"Error saving NER results: {e}")

    # Save party sentiment
    if avg_party_sentiment:
        try:
            pd.DataFrame.from_dict(avg_party_sentiment, orient='index').to_csv(os.path.join(config.OUTPUT_DIR, f'{prefix}avg_party_sentiment_sentence.csv'))
            logger.info("Saved average party sentiment (sentence level).")
        except Exception as e: logger.error(f"Error saving party sentiment: {e}")

    # Save Topic Modeling results (JSON) - Keep structure same as VADER
    topic_results = {'lda_topics': topics_lda if topics_lda else {}, 'nmf_topics': topics_nmf if topics_nmf else {}}
    try:
        with open(os.path.join(config.OUTPUT_DIR, f'{prefix}topic_modeling_results.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_results, f, indent=4)
        logger.info("Saved topic modeling results.")
    except Exception as e: logger.error(f"Error saving topic modeling results: {e}")

    # Save Statistical Test results (JSON)
    if stat_results:
        try:
            with open(os.path.join(config.OUTPUT_DIR, f'{prefix}statistical_test_results.json'), 'w', encoding='utf-8') as f:
                 serializable_results = {}
                 for key, value_dict in stat_results.items():
                      serializable_results[key] = { k: (float(v) if isinstance(v, (np.float_, np.int_)) else v) for k, v in value_dict.items() }
                 json.dump(serializable_results, f, indent=4)
            logger.info("Saved statistical test results.")
        except Exception as e: logger.error(f"Error saving statistical test results: {e}")

     # Save Omission Proxy results
    if omission_counts:
         try:
             omission_df = pd.DataFrame.from_dict(omission_counts, orient='index').fillna(0)
             omission_df.to_csv(os.path.join(config.OUTPUT_DIR, f'{prefix}omission_proxy_counts.csv'))
             logger.info("Saved omission proxy counts.")
         except Exception as e: logger.error(f"Error saving omission proxy counts: {e}")

    logger.info(f"All available results saved to {config.OUTPUT_DIR}")
    logger.info(f"Detailed logs available in {config.ANALYSIS_TEXTBLOB_LOG_FILE}")


# === Main Execution ===
if __name__ == "__main__":
    logger.info("--- Starting TextBlob Analysis Pipeline ---")

    # 1. Load Data
    df = load_data(config.PREPROCESSED_ARTICLES_CSV)

    if df is not None and not df.empty:
        # 2. Perform Sentiment Analysis
        df = perform_textblob_sentiment_analysis(df)
        avg_sentiment_outlet = df.groupby('outlet')['polarity'].mean().sort_values() if 'polarity' in df.columns else pd.Series(dtype=float)

        # 3. Perform Word Frequency Analysis
        most_common_overall = analyze_word_frequency(df['processed_text'])
        word_freq_by_outlet = {} # Optional: Implement loop if needed

        # 4. Analyze Ideological Term Frequency
        term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)

        # 5. Analyze Sentiment Towards Parties (Sentence Level)
        avg_party_sentiment = analyze_sentiment_towards_parties_textblob(df, config.PARTY_TERMS)

        # 6. Perform NER
        ner_persons_overall, ner_orgs_overall, ner_per_outlet = perform_ner(df)

        # 7. Perform Topic Modeling (LDA and NMF)
        lda_model, lda_vectorizer, topics_lda = perform_topic_modeling(df['processed_text'], model_type='lda')
        nmf_model, nmf_vectorizer, topics_nmf = perform_topic_modeling(df['processed_text'], model_type='nmf')

        # 8. Perform Omission Analysis Proxy
        omission_counts = analyze_omission_proxy(df, config.OMISSION_TOPICS)

        # 9. Perform Statistical Tests
        stat_results = perform_statistical_tests(df)

        # 10. Generate Visualizations
        generate_visualizations(df, avg_sentiment_outlet, most_common_overall, word_freq_by_outlet, ner_persons_overall, ner_orgs_overall, avg_party_sentiment, topics_lda, omission_counts)

        # 11. Save Results
        save_results(df, avg_sentiment_outlet, most_common_overall, term_counts_overall, term_counts_per_outlet, ner_persons_overall, ner_orgs_overall, ner_per_outlet, avg_party_sentiment, topics_lda, topics_nmf, stat_results, omission_counts)

    else:
        logger.error("Failed to load or process data. Exiting TextBlob analysis.")

    logger.info("--- Finished TextBlob Analysis Pipeline ---")