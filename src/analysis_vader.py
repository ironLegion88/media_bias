# src/analysis_vader.py
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
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
from nltk.stem import WordNetLemmatizer # Added for term processing

# --- Configuration ---
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it's in the project root or src directory if running directly.")
    # Attempt to load from parent directory if running script from src/
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        import config
    except ImportError:
        print("Critical Error: config.py not found. Please ensure it's in the project root.")
        exit(1)


# Create output and logs directories if they don't exist
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log_handler = logging.FileHandler(config.ANALYSIS_VADER_LOG_FILE)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__) # Unique logger per module
logger.setLevel(logging.INFO)
if not logger.hasHandlers(): # Prevent adding handler multiple times
    logger.addHandler(log_handler)
    # console_handler = logging.StreamHandler() # Optional: for console output
    # console_handler.setFormatter(log_formatter)
    # logger.addHandler(console_handler)


# --- Download necessary NLTK data ---
def download_nltk_data():
    """Downloads required NLTK models if they don't exist."""
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    for item in required_nltk_data:
        try:
            if item == 'vader_lexicon': path = f'sentiment/{item}.zip'
            elif item == 'punkt': path = f'tokenizers/{item}'
            else: path = f'corpora/{item}'
            nltk.data.find(path)
            logger.debug(f"NLTK data '{item}' found.")
        except LookupError:
            logger.info(f"NLTK data '{item}' not found. Attempting download...")
            try:
                nltk.download(item, quiet=True)
                logger.info(f"NLTK data '{item}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download NLTK data '{item}'. Error: {e}")
                if item in ['punkt', 'vader_lexicon']:
                     logger.error(f"Critical NLTK resource '{item}' missing. Exiting.")
                     # sys.exit(1) # Consider uncommenting to halt if critical data is missing

download_nltk_data()

# --- Load spaCy Model ---
NLP = None
try:
    NLP = spacy.load(config.SPACY_MODEL)
    logger.info(f"spaCy model '{config.SPACY_MODEL}' loaded successfully.")
except OSError:
    logger.warning(f"spaCy model '{config.SPACY_MODEL}' not found. Attempting download...")
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
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['processed_text'] = df['processed_text'].fillna('')
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def perform_vader_sentiment_analysis(df):
    logger.info("Performing VADER Sentiment Analysis...")
    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid or missing 'processed_text' column for VADER analysis.")
        return df
    analyzer = SentimentIntensityAnalyzer()
    df['vader_compound'] = df['processed_text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
    logger.info("VADER sentiment analysis complete.")
    return df

def analyze_word_frequency(text_series, top_n=config.TOP_N_WORDS, stopwords_set=config.COMBINED_STOPWORDS):
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

def analyze_sentiment_towards_parties_vader(df, party_term_dict):
    """
    Calculates average VADER sentiment of sentences mentioning specific parties,
    returning detailed per-article, per-party sentiment.
    """
    logger.info("Analyzing Sentiment Towards Political Parties (VADER - Sentence Level)...")
    if df is None or 'processed_text' not in df.columns or 'link' not in df.columns or 'outlet' not in df.columns:
        logger.error("DataFrame invalid or missing required columns for party sentiment analysis.")
        return [] # Return empty list on error

    analyzer = SentimentIntensityAnalyzer()
    party_sentiment_details = []
    # Temp dict: Key: (link, party), Value: {'total_compound': 0.0, 'sentence_count': 0, 'outlet': outlet_name}
    temp_article_party_sentiment = {}

    # Pre-process party terms (lemmatize) to match processed_text
    processed_party_term_dict = {}
    lemmatizer = WordNetLemmatizer()
    for party, terms in party_term_dict.items():
         processed_party_term_dict[party] = {lemmatizer.lemmatize(term.lower()) for term in terms} # Use set for faster lookup

    total_articles = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 20 == 0 or i == total_articles - 1:
             logger.info(f"Processing party sentiment in article {i+1}/{total_articles}")

        text = row['processed_text']
        link = row['link']
        outlet_name = row['outlet']

        if not isinstance(text, str) or not text.strip():
            continue

        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Could not sentence-tokenize article snippet: {text[:100]}... for link {link}. Error: {e}")
            continue

        for sentence in sentences:
            sentence_words = set(sentence.lower().split()) # For efficient checking

            for party, processed_terms_set in processed_party_term_dict.items():
                # Check if any lemmatized party term is in the sentence words
                if not processed_terms_set.isdisjoint(sentence_words): # Efficient check for intersection
                    try:
                        vs = analyzer.polarity_scores(sentence) # VADER works well on somewhat raw sentences
                        key = (link, party)
                        if key not in temp_article_party_sentiment:
                            temp_article_party_sentiment[key] = {
                                'total_compound': 0.0, 'sentence_count': 0, 'outlet': outlet_name
                            }
                        temp_article_party_sentiment[key]['total_compound'] += vs['compound']
                        temp_article_party_sentiment[key]['sentence_count'] += 1
                    except Exception as e:
                         logger.warning(f"VADER error on sentence: {sentence[:50]}... Error: {e}")

    # Calculate the average per article/party and create the final list
    for (link, party), data in temp_article_party_sentiment.items():
        if data['sentence_count'] > 0:
            avg_compound = data['total_compound'] / data['sentence_count']
            party_sentiment_details.append({
                'link': link,
                'outlet': data['outlet'],
                'party': party,
                'avg_sentence_sentiment_vader': avg_compound, # Specific name
                'sentence_count': data['sentence_count']
            })

    logger.info(f"Party sentence sentiment calculation (VADER) complete. Found {len(party_sentiment_details)} party mentions with sentiment.")
    return party_sentiment_details

def categorize_and_aggregate_party_sentiment(party_sentiment_details, sentiment_col='avg_sentence_sentiment_vader', tool_name='VADER'):
    logger.info(f"Categorizing article sentiment towards parties using {tool_name} scores...")
    if not party_sentiment_details:
        logger.warning("No detailed party sentiment data provided for categorization.")
        return pd.DataFrame()

    df_details = pd.DataFrame(party_sentiment_details)
    if df_details.empty or sentiment_col not in df_details.columns:
        logger.error(f"Sentiment column '{sentiment_col}' not found or df_details is empty.")
        return pd.DataFrame()

    if tool_name == 'VADER': pos_thresh, neg_thresh = 0.05, -0.05
    elif tool_name == 'TextBlob': pos_thresh, neg_thresh = 0.1, -0.1 # Polarity
    else: pos_thresh, neg_thresh = 0.05, -0.05
    def categorize(score):
        if score > pos_thresh: return 'Positive'
        elif score < neg_thresh: return 'Negative'
        else: return 'Neutral'
    df_details['sentiment_category'] = df_details[sentiment_col].apply(categorize)
    agg_counts = df_details.groupby(['outlet', 'party', 'sentiment_category']).size().unstack(fill_value=0)
    for cat in ['Positive', 'Neutral', 'Negative']:
        if cat not in agg_counts.columns: agg_counts[cat] = 0
    agg_counts.rename(columns={'Positive': 'Positive_Count', 'Neutral': 'Neutral_Count', 'Negative': 'Negative_Count'}, inplace=True)
    agg_counts['Total_Articles_Mentioning'] = agg_counts['Positive_Count'] + agg_counts['Neutral_Count'] + agg_counts['Negative_Count']
    if 'Total_Articles_Mentioning' in agg_counts and agg_counts['Total_Articles_Mentioning'].sum() > 0 :
        agg_counts['Positive_Proportion'] = agg_counts['Positive_Count'] / agg_counts['Total_Articles_Mentioning']
        agg_counts['Neutral_Proportion'] = agg_counts['Neutral_Count'] / agg_counts['Total_Articles_Mentioning']
        agg_counts['Negative_Proportion'] = agg_counts['Negative_Count'] / agg_counts['Total_Articles_Mentioning']
    else:
        agg_counts['Positive_Proportion'] = 0
        agg_counts['Neutral_Proportion'] = 0
        agg_counts['Negative_Proportion'] = 0

    logger.info("Aggregation of categorized party sentiment complete.")
    return agg_counts[['Positive_Count', 'Neutral_Count', 'Negative_Count', 'Total_Articles_Mentioning', 'Positive_Proportion', 'Neutral_Proportion', 'Negative_Proportion']].copy()


def perform_ner(df, top_n=config.TOP_N_ENTITIES):
    logger.info("Performing Named Entity Recognition (NER)...")
    if NLP is None: logger.error("spaCy model not loaded. Cannot perform NER."); return None, None, None
    logger.warning("NER is being performed on 'processed_text'. Results may be less accurate than on original text.")
    if df is None or 'processed_text' not in df.columns: logger.error("DataFrame invalid for NER."); return None, None, None
    ner_persons_overall, ner_orgs_overall = Counter(), Counter()
    outlets = df['outlet'].unique()
    ner_per_outlet = {outlet: {'PERSON': Counter(), 'ORG': Counter()} for outlet in outlets}
    total_articles = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 20 == 0 or i == total_articles - 1: logger.info(f"Processing NER in article {i+1}/{total_articles}")
        outlet, text = row['outlet'], row['processed_text']
        if not isinstance(text, str) or not text.strip(): continue
        try:
            doc = NLP(text)
            for ent in doc.ents:
                entity_text = ent.text.strip().lower() # Lowercase for consistent counting
                if len(entity_text) > 2 and not entity_text.isdigit() and entity_text not in config.COMBINED_STOPWORDS: # Avoid stopwords as entities
                    if ent.label_ == 'PERSON':
                        ner_persons_overall[entity_text] += 1
                        if outlet in ner_per_outlet: ner_per_outlet[outlet]['PERSON'][entity_text] += 1
                    elif ent.label_ == 'ORG':
                        ner_orgs_overall[entity_text] += 1
                        if outlet in ner_per_outlet: ner_per_outlet[outlet]['ORG'][entity_text] += 1
        except Exception as e: logger.warning(f"spaCy NER error: {text[:100]}... Error: {e}")
    top_persons = ner_persons_overall.most_common(top_n)
    top_orgs = ner_orgs_overall.most_common(top_n)
    logger.info("NER analysis complete.")
    return top_persons, top_orgs, ner_per_outlet

def perform_topic_modeling(text_series, n_topics=config.N_TOPICS, n_top_words=10, model_type='lda'):
    logger.info(f"Performing Topic Modeling ({model_type.upper()}) with {n_topics} topics...")
    if text_series is None or text_series.empty or text_series.str.strip().eq('').all():
        logger.warning("Input text series is empty or contains only whitespace for topic modeling.")
        return None, None, None
    text_series = text_series.astype(str)
    # Filter out empty strings after casting, as vectorizer might fail
    text_series = text_series[text_series.str.strip() != '']
    if text_series.empty:
        logger.warning("Text series became empty after filtering whitespace for topic modeling.")
        return None, None, None

    if model_type == 'lda':
        vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words=list(config.COMBINED_STOPWORDS), max_features=1000)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
    elif model_type == 'nmf':
        vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, stop_words=list(config.COMBINED_STOPWORDS), max_features=1000)
        model = NMF(n_components=n_topics, random_state=42, max_iter=400, init='nndsvda')
    else: logger.error(f"Invalid model_type '{model_type}'."); return None, None, None
    try:
        dtm = vectorizer.fit_transform(text_series)
        if dtm.shape[0] == 0 or dtm.shape[1] == 0 : # Check if dtm is empty
             logger.error("Document-term matrix is empty. Cannot fit topic model. Check min_df/stopwords/text quality.")
             return None, None, None
        model.fit(dtm)
    except ValueError as e: logger.error(f"Error fitting topic model (check min_df/corpus size): {e}"); return None, None, None
    except Exception as e: logger.error(f"Unexpected error fitting topic model: {e}"); return None, None, None
    logger.info("Topic model fitting complete.")
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic_dist in enumerate(model.components_):
        top_words_indices = topic_dist.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics[topic_idx] = " ".join(top_words)
        logger.info(f"Topic {topic_idx}: {topics[topic_idx]}")
    logger.info("Topic modeling complete.")
    return model, vectorizer, topics

def analyze_omission_proxy(df, omission_topics_keywords):
    logger.info("Performing Omission Analysis Proxy (Keyword Presence)...")
    if not omission_topics_keywords: logger.warning("OMISSION_TOPICS not defined. Skipping."); return None
    if df is None or not {'outlet', 'processed_text'}.issubset(df.columns): logger.error("DataFrame invalid."); return None
    processed_keywords = {}
    lemmatizer = WordNetLemmatizer()
    for topic, keywords in omission_topics_keywords.items():
         processed_keywords[topic] = [lemmatizer.lemmatize(kw.lower()) for kw in keywords]
    omission_counts = {outlet: Counter() for outlet in df['outlet'].unique()}
    for index, row in df.iterrows():
        outlet, text = row['outlet'], row['processed_text']
        if not isinstance(text, str) or not text.strip(): continue
        words_in_article = set(text.split())
        for topic, keywords in processed_keywords.items():
            if any(kw in words_in_article for kw in keywords):
                 if outlet in omission_counts: omission_counts[outlet][topic] += 1
    logger.info("Omission analysis proxy complete.")
    return omission_counts

def perform_statistical_tests(df, alpha=config.STAT_ALPHA):
    logger.info("Performing Statistical Tests on VADER Sentiment...")
    results = {}
    sentiment_col = 'vader_compound'
    if df is None or not {'outlet', sentiment_col}.issubset(df.columns) or df[sentiment_col].isnull().all():
        logger.error(f"DataFrame invalid for statistical tests on {sentiment_col}.")
        return results
    df_clean = df.dropna(subset=[sentiment_col])
    if len(df_clean) < 2: logger.warning("Not enough valid data for stats."); return results
    outlets = df_clean['outlet'].unique()
    if len(outlets) > 1:
        sentiment_groups = [df_clean[sentiment_col][df_clean['outlet'] == o] for o in outlets]
        sentiment_groups_filtered = [g for g in sentiment_groups if len(g) >= 2]
        if len(sentiment_groups_filtered) > 1:
             logger.info(f"Running ANOVA on sentiment across {len(sentiment_groups_filtered)} outlets...")
             try:
                 f_stat, p_value = f_oneway(*sentiment_groups_filtered)
                 interp = f"Significant (p<{alpha})" if p_value<alpha else f"Not significant (p>={alpha})"
                 results['ANOVA_sentiment_by_outlet'] = {'f':f_stat, 'p':p_value, 'interp':f"ANOVA ({sentiment_col}): {interp}."}
                 logger.info(results['ANOVA_sentiment_by_outlet'])
             except Exception as e: logger.error(f"Error during ANOVA: {e}")
        else: logger.warning(f"ANOVA: Need >=2 outlets with >=2 articles.")
    outlets_with_data = [o for o in outlets if len(df_clean[df_clean['outlet']==o]) >= 2]
    if len(outlets_with_data) >= 2:
        o1,o2 = outlets_with_data[0], outlets_with_data[1]
        g1,g2 = df_clean[sentiment_col][df_clean['outlet']==o1], df_clean[sentiment_col][df_clean['outlet']==o2]
        logger.info(f"T-test: {sentiment_col} ({o1} vs {o2})...")
        try:
            t_stat, p_value = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
            interp = f"Significant (p<{alpha})" if p_value<alpha else f"Not significant (p>={alpha})"
            results[f"TTest_{o1}_vs_{o2}"] = {'t':t_stat, 'p':p_value, 'interp':f"T-test ({o1} vs {o2}): {interp}."}
            logger.info(results[f"TTest_{o1}_vs_{o2}"])
        except Exception as e: logger.error(f"Error T-test ({o1} vs {o2}): {e}")
    term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)
    if term_counts_per_outlet:
         cat1, cat2 = 'pro_bjp_modi', 'critical_bjp_modi'
         ct_data, valid_outlets_chi2 = [], []
         for o, cats_data in term_counts_per_outlet.items():
             c1 = sum(cats_data.get(cat1, Counter()).values())
             c2 = sum(cats_data.get(cat2, Counter()).values())
             if c1 > 0 or c2 > 0: ct_data.append([c1,c2]); valid_outlets_chi2.append(o)
         if len(ct_data) >= 2 and np.array(ct_data).shape[1] == 2: # Ensure table is 2D
            logger.info(f"Chi2 test: '{cat1}' vs '{cat2}' across {len(valid_outlets_chi2)} outlets...")
            try:
                chi2, p, dof, _ = chi2_contingency(np.array(ct_data))
                interp = f"Significant (p<{alpha})" if p<alpha else f"Not significant (p>={alpha})"
                results['Chi2_BJP_Terms_Outlet'] = {'chi2':chi2, 'p':p, 'dof':dof, 'interp':f"Chi2 ('{cat1}' vs '{cat2}'): {interp}."}
                logger.info(results['Chi2_BJP_Terms_Outlet'])
            except ValueError as e: logger.error(f"Error Chi2 (low freq?): {e}")
            except Exception as e: logger.error(f"Error Chi2: {e}")
         else: logger.warning(f"Chi2 '{cat1}' vs '{cat2}': Need >=2 outlets with data.")
    logger.info("Statistical tests complete.")
    return results

def save_plot(plt_object, filename_suffix):
    try:
        filepath = os.path.join(config.OUTPUT_DIR, f"vader_{filename_suffix}")
        plt_object.savefig(filepath)
        logger.info(f"Plot saved to {filepath}")
        plt_object.close()
    except Exception as e:
        logger.error(f"Error saving plot {filepath}: {e}")
        plt_object.close()

def generate_visualizations(df_sentiment_scores, avg_sentiment_outlet, most_common_overall,
                            ner_persons_overall, ner_orgs_overall,
                            party_sentiment_details, agg_party_counts,
                            topics_lda, topics_nmf, omission_counts):
    logger.info("Generating visualizations (VADER)...")
    sentiment_col = 'vader_compound'
    tool_name = 'VADER'

    if df_sentiment_scores is not None and sentiment_col in df_sentiment_scores.columns and not df_sentiment_scores[sentiment_col].isnull().all():
        plt.figure(figsize=(14, 8)); sns.boxplot(data=df_sentiment_scores, x='outlet', y=sentiment_col, palette='viridis')
        plt.title(f'{tool_name} Sentiment Score Distribution by Outlet', fontsize=16); plt.ylabel(f'{tool_name} Compound Score'); plt.xlabel('Outlet')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout(); save_plot(plt, 'sentiment_dist_outlet.png')

        if not avg_sentiment_outlet.empty:
            plt.figure(figsize=(12, 7)); avg_sentiment_outlet.plot(kind='bar', color=sns.color_palette("viridis", len(avg_sentiment_outlet)))
            plt.title(f'Average {tool_name} Sentiment Score by Outlet', fontsize=16); plt.ylabel(f'Average {tool_name} Compound Score'); plt.xlabel('')
            plt.xticks(rotation=45, ha='right'); plt.axhline(0, color='grey',ls='--'); plt.tight_layout(); save_plot(plt, 'avg_sentiment_outlet.png')

    if most_common_overall:
        try:
            wc = WordCloud(width=1200, height=600, background_color='white', max_words=100, colormap='magma').generate_from_frequencies(dict(most_common_overall))
            plt.figure(figsize=(15, 7)); plt.imshow(wc, interpolation='bilinear'); plt.axis('off')
            plt.title('Overall Top Words', fontsize=16); plt.tight_layout(); save_plot(plt, 'wordcloud_overall.png')
        except Exception as e: logger.error(f"Error generating overall word cloud: {e}")

    if ner_persons_overall:
        ner_df = pd.DataFrame(ner_persons_overall, columns=['Person', 'Count'])
        plt.figure(figsize=(12, max(8, len(ner_df) * 0.4))); sns.barplot(data=ner_df, y='Person', x='Count', palette='coolwarm')
        plt.title(f'Top PERSON Entities', fontsize=16); plt.tight_layout(); save_plot(plt, 'ner_top_persons.png')
    if ner_orgs_overall:
        ner_df = pd.DataFrame(ner_orgs_overall, columns=['Organization', 'Count'])
        plt.figure(figsize=(12, max(8, len(ner_df) * 0.4))); sns.barplot(data=ner_df, y='Organization', x='Count', palette='coolwarm')
        plt.title(f'Top ORGANIZATION Entities', fontsize=16); plt.tight_layout(); save_plot(plt, 'ner_top_orgs.png')

    # New Party Sentiment Plots
    if not agg_party_counts.empty:
        plot_party_sentiment_categories(agg_party_counts, tool_name=tool_name, plot_type='proportion', prefix="vader_")
        plot_party_sentiment_categories(agg_party_counts, tool_name=tool_name, plot_type='count', prefix="vader_")

    if party_sentiment_details:
        plot_party_sentiment_distributions(party_sentiment_details, sentiment_col='avg_sentence_sentiment_vader', tool_name=tool_name, prefix="vader_")

    if omission_counts:
        omission_df = pd.DataFrame.from_dict(omission_counts, orient='index').fillna(0)
        if not omission_df.empty:
             omission_df.plot(kind='bar', figsize=(15, 8), colormap='tab20')
             plt.title('Omission Proxy: Topic Mentions/Outlet', fontsize=16); plt.xlabel('Outlet'); plt.ylabel('# Articles Mentioning Topic')
             plt.xticks(rotation=45, ha='right'); plt.legend(title='Topics', bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout()
             save_plot(plt, 'omission_proxy_mentions.png')
    logger.info("Visualizations generation complete (VADER).")


def plot_party_sentiment_categories(agg_party_counts, tool_name='VADER', plot_type='proportion', prefix=""):
    if agg_party_counts.empty: logger.warning(f"{prefix}No aggregated party sentiment counts to plot."); return
    cols = ['Positive_Proportion','Neutral_Proportion','Negative_Proportion'] if plot_type=='proportion' else ['Positive_Count','Neutral_Count','Negative_Count']
    ylab = 'Proportion of Articles' if plot_type=='proportion' else 'Number of Articles'
    title_sfx = 'Proportions' if plot_type=='proportion' else 'Counts'
    parties = agg_party_counts.index.get_level_values('party').unique()
    for party in parties:
        party_data = agg_party_counts.xs(party, level='party')
        if party_data.empty: continue
        plt.figure(figsize=(max(10, len(party_data.index)*0.8),7))
        party_data[cols].plot(kind='bar',stacked=True,colormap='coolwarm_r',ax=plt.gca())
        plt.title(f'{tool_name} Sentiment Towards {party} by Outlet ({title_sfx})',fontsize=16); plt.xlabel('Outlet'); plt.ylabel(ylab)
        plt.xticks(rotation=45,ha='right'); plt.legend(title='Sentiment',bbox_to_anchor=(1.05,1),loc='upper left'); plt.tight_layout(rect=[0,0,0.85,1])
        safe_party_name = re.sub(r'\W+','_',party); save_plot(plt,f'{prefix}party_sentiment_categories_{safe_party_name}_{plot_type}.png')

def plot_party_sentiment_distributions(party_sentiment_details, sentiment_col, tool_name='VADER', prefix=""):
    if not party_sentiment_details: logger.warning(f"{prefix}No detailed party sentiment data for distribution plots."); return
    df_details = pd.DataFrame(party_sentiment_details)
    if df_details.empty or sentiment_col not in df_details.columns: logger.error(f"{prefix}Invalid data for distribution plots."); return
    parties = df_details['party'].unique()
    for party in parties:
        party_data = df_details[df_details['party']==party]
        if party_data.empty or party_data[sentiment_col].isnull().all(): continue
        plt.figure(figsize=(14,7))
        sns.violinplot(data=party_data, x='outlet', y=sentiment_col, palette='viridis', inner='quartile')
        plt.title(f'{tool_name} Sentiment Dist. Towards {party} by Outlet',fontsize=16); plt.ylabel(f'{tool_name} Score (Avg Sentence)'); plt.xlabel('Outlet')
        plt.axhline(0,color='grey',ls='--'); plt.xticks(rotation=45,ha='right'); plt.tight_layout()
        safe_party_name = re.sub(r'\W+','_',party); save_plot(plt,f'{prefix}party_sentiment_distribution_{safe_party_name}.png')


def save_results(df_scores, avg_sent_outlet, common_overall, term_overall, term_per_outlet,
                 ner_p_overall, ner_o_overall, ner_per_out,
                 party_sent_details, agg_party_cnts, # New params
                 topics_lda, topics_nmf, stat_res, omission_cnts, prefix="vader_"):
    logger.info(f"Saving {prefix}analysis results...")
    out_dir = config.OUTPUT_DIR

    if df_scores is not None: pd.DataFrame(df_scores).to_csv(os.path.join(out_dir, f'{prefix}articles_with_sentiment.csv'), index=False)
    if not avg_sent_outlet.empty: avg_sent_outlet.to_csv(os.path.join(out_dir, f'{prefix}avg_sentiment_outlet.csv'), header=[f'avg_{prefix}score'])
    if common_overall: pd.DataFrame(common_overall, columns=['word','count']).to_csv(os.path.join(out_dir,f'{prefix}top_words_overall.csv'), index=False)

    term_res = {'overall': {c:dict(cnt) for c,cnt in term_overall.items()} if term_overall else {},
                'per_outlet': {o:{c:dict(cnt) for c,cnt in cats.items()} for o,cats in term_per_outlet.items()} if term_per_outlet else {}}
    with open(os.path.join(out_dir,f'{prefix}ideological_term_counts.json'),'w') as f: json.dump(term_res,f,indent=4)

    ner_res = {'persons_overall':ner_p_overall if ner_p_overall else [], 'orgs_overall':ner_o_overall if ner_o_overall else [],
               'per_outlet':{o:{t:dict(cnt) for t,cnt in types.items()} for o,types in ner_per_out.items()} if ner_per_out else {}}
    with open(os.path.join(out_dir,f'{prefix}ner_results.json'),'w') as f: json.dump(ner_res,f,indent=4)

    # Save new party sentiment results
    if party_sent_details: pd.DataFrame(party_sent_details).to_csv(os.path.join(out_dir, f'{prefix}party_sentiment_details.csv'), index=False)
    if not agg_party_cnts.empty: agg_party_cnts.to_csv(os.path.join(out_dir, f'{prefix}party_sentiment_aggregated_counts.csv'))

    topic_res = {'lda':topics_lda if topics_lda else {}, 'nmf':topics_nmf if topics_nmf else {}}
    with open(os.path.join(out_dir,f'{prefix}topic_modeling_results.json'),'w') as f: json.dump(topic_res,f,indent=4)

    if stat_res:
        serializable_stat_res = {k:{sk:(float(sv) if isinstance(sv,(np.float_,np.int_)) else sv) for sk,sv in sd.items()} for k,sd in stat_res.items()}
        with open(os.path.join(out_dir,f'{prefix}statistical_test_results.json'),'w') as f: json.dump(serializable_stat_res,f,indent=4)
    if omission_cnts: pd.DataFrame.from_dict(omission_cnts,orient='index').fillna(0).to_csv(os.path.join(out_dir,f'{prefix}omission_proxy_counts.csv'))
    logger.info(f"All available {prefix}results saved to {out_dir}")


# === Main Execution ===
if __name__ == "__main__":
    logger.info("--- Starting VADER Analysis Pipeline ---")
    df = load_data(config.PREPROCESSED_ARTICLES_CSV)

    if df is not None and not df.empty:
        df = perform_vader_sentiment_analysis(df)
        avg_sentiment_outlet = df.groupby('outlet')['vader_compound'].mean().sort_values() if 'vader_compound' in df.columns else pd.Series(dtype=float)
        most_common_overall = analyze_word_frequency(df['processed_text'])
        term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)

        # Updated Party Sentiment Analysis
        party_sentiment_details_vader = analyze_sentiment_towards_parties_vader(df, config.PARTY_TERMS)
        agg_party_counts_vader = categorize_and_aggregate_party_sentiment(
            party_sentiment_details_vader,
            sentiment_col='avg_sentence_sentiment_vader', # Ensure this matches output key
            tool_name='VADER'
        )

        ner_persons_overall, ner_orgs_overall, ner_per_outlet = perform_ner(df)
        lda_model, lda_vectorizer, topics_lda = perform_topic_modeling(df['processed_text'], model_type='lda')
        nmf_model, nmf_vectorizer, topics_nmf = perform_topic_modeling(df['processed_text'], model_type='nmf')
        omission_counts = analyze_omission_proxy(df, config.OMISSION_TOPICS)
        stat_results = perform_statistical_tests(df)

        generate_visualizations(df, avg_sentiment_outlet, most_common_overall,
                                ner_persons_overall, ner_orgs_overall,
                                party_sentiment_details_vader, agg_party_counts_vader,
                                topics_lda, topics_nmf, omission_counts)

        save_results(df, avg_sentiment_outlet, most_common_overall, term_counts_overall, term_counts_per_outlet,
                     ner_persons_overall, ner_orgs_overall, ner_per_outlet,
                     party_sentiment_details_vader, agg_party_counts_vader,
                     topics_lda, topics_nmf, stat_results, omission_counts, prefix="vader_")
    else:
        logger.error("Failed to load or process data. Exiting VADER analysis.")
    logger.info("--- Finished VADER Analysis Pipeline ---")