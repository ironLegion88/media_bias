# src/analysis_textblob.py
import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob # Use TextBlob
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
from nltk.stem import WordNetLemmatizer # Added

# --- Configuration ---
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it's in the project root or src directory if running directly.")
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        import config
    except ImportError:
        print("Critical Error: config.py not found. Please ensure it's in the project root.")
        exit(1)


os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log_handler = logging.FileHandler(config.ANALYSIS_TEXTBLOB_LOG_FILE)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__ + "_textblob") # Unique logger name
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(log_formatter)
    # logger.addHandler(console_handler)

# --- Download necessary NLTK data ---
def download_nltk_data():
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'brown']
    for item in required_nltk_data:
        try:
            if item == 'punkt': path = f'tokenizers/{item}'
            elif item == 'averaged_perceptron_tagger': path = f'taggers/{item}'
            else: path = f'corpora/{item}' # stopwords, wordnet, brown
            nltk.data.find(path)
            logger.debug(f"NLTK data '{item}' found.")
        except LookupError:
            logger.info(f"NLTK data '{item}' for TextBlob not found. Attempting download...")
            try:
                nltk.download(item, quiet=True)
                logger.info(f"NLTK data '{item}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download NLTK data '{item}'. Error: {e}")
                if item in ['punkt', 'averaged_perceptron_tagger']:
                     logger.error(f"Critical NLTK resource '{item}' for TextBlob missing. Exiting.")
                     # sys.exit(1)

download_nltk_data()

# --- Load spaCy Model --- (Identical to VADER script)
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
    except Exception as e: logger.error(f"Failed to download or load spaCy model. NER disabled. Error: {e}")
except ImportError: logger.error("spaCy library not installed. NER disabled.")


# === Helper & Analysis Functions (Many are identical to VADER version) ===

def load_data(filepath): # Identical
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath); df['processed_text'] = df['processed_text'].fillna('');
        logger.info(f"Successfully loaded data. Shape: {df.shape}"); return df
    except FileNotFoundError: logger.error(f"File not found: {filepath}"); return None
    except Exception as e: logger.error(f"Error loading data from {filepath}: {e}"); return None

def perform_textblob_sentiment_analysis(df):
    logger.info("Performing TextBlob Sentiment Analysis...")
    if df is None or 'processed_text' not in df.columns:
        logger.error("DataFrame invalid for TextBlob analysis.")
        return df
    polarities, subjectivities = [], []
    for text in df['processed_text']:
        try:
            blob = TextBlob(str(text)); polarities.append(blob.sentiment.polarity); subjectivities.append(blob.sentiment.subjectivity)
        except Exception as e:
            logger.warning(f"TextBlob error: '{str(text)[:50]}...': {e}"); polarities.append(None); subjectivities.append(None)
    df['polarity'] = polarities; df['subjectivity'] = subjectivities
    df.dropna(subset=['polarity', 'subjectivity'], inplace=True)
    logger.info("TextBlob sentiment analysis complete.")
    return df

def analyze_word_frequency(text_series, top_n=config.TOP_N_WORDS, stopwords_set=config.COMBINED_STOPWORDS): # Identical
    logger.info(f"Performing Word Frequency Analysis (Top {top_n})...")
    if text_series is None or text_series.empty: logger.warning("Input empty for word freq."); return []
    try: all_text = ' '.join(text_series.astype(str).tolist())
    except Exception as e: logger.error(f"Error combining text series for word freq: {e}"); return []
    if not all_text.strip(): logger.warning("No text content for word freq."); return []
    tokens = all_text.split()
    filtered_tokens = [w for w in tokens if len(w) > 2 and w not in stopwords_set]
    word_counts = Counter(filtered_tokens)
    most_common = word_counts.most_common(top_n)
    logger.info(f"Word freq complete. Found {len(word_counts)} unique words (filtered).")
    return most_common

def analyze_ideological_term_frequency(df, term_dict): # Identical logic
    logger.info("Analyzing Ideological Term Frequency...")
    if df is None or not {'outlet', 'processed_text'}.issubset(df.columns): logger.error("Invalid DF."); return None, None
    term_counts_overall = {c: Counter() for c in term_dict}
    outlets = df['outlet'].unique()
    term_counts_per_outlet = {o: {c: Counter() for c in term_dict} for o in outlets}
    processed_term_dict = {}; lemmatizer = WordNetLemmatizer()
    for cat, terms in term_dict.items(): processed_term_dict[cat] = [lemmatizer.lemmatize(t.lower()) for t in terms]
    for _, row in df.iterrows():
        outlet, text = row['outlet'], row['processed_text']
        if not isinstance(text, str) or not text.strip(): continue
        article_word_counts = Counter(text.split())
        for cat, p_terms in processed_term_dict.items():
            orig_terms = term_dict[cat]
            for i, term in enumerate(p_terms):
                count = article_word_counts[term]
                if count > 0:
                    term_counts_overall[cat][orig_terms[i]] += count
                    if outlet in term_counts_per_outlet: term_counts_per_outlet[outlet][cat][orig_terms[i]] += count
    logger.info("Ideological term freq analysis complete.")
    return term_counts_overall, term_counts_per_outlet


def analyze_sentiment_towards_parties_textblob(df, party_term_dict):
    """
    Calculates average TextBlob polarity/subjectivity of sentences mentioning specific parties,
    returning detailed per-article, per-party sentiment.
    """
    logger.info("Analyzing Sentiment Towards Political Parties (TextBlob - Sentence Level)...")
    if df is None or 'processed_text' not in df.columns or 'link' not in df.columns or 'outlet' not in df.columns:
        logger.error("DataFrame invalid or missing required columns for party sentiment analysis (TextBlob).")
        return []

    party_sentiment_details = []
    temp_article_party_sentiment = {} # Key: (link, party), Value for sums and counts

    processed_party_term_dict = {}
    lemmatizer = WordNetLemmatizer()
    for party, terms in party_term_dict.items():
         processed_party_term_dict[party] = {lemmatizer.lemmatize(term.lower()) for term in terms}

    total_articles = len(df)
    for i, row in df.iterrows():
        if (i + 1) % 20 == 0 or i == total_articles - 1:
             logger.info(f"Processing TextBlob party sentiment in article {i+1}/{total_articles}")

        text = row['processed_text']
        link = row['link']
        outlet_name = row['outlet']

        if not isinstance(text, str) or not text.strip():
            continue

        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Could not sentence-tokenize (TextBlob): {text[:100]}... for link {link}. Error: {e}")
            continue

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())

            for party, processed_terms_set in processed_party_term_dict.items():
                if not processed_terms_set.isdisjoint(sentence_words):
                    try:
                        blob = TextBlob(sentence)
                        key = (link, party)
                        if key not in temp_article_party_sentiment:
                            temp_article_party_sentiment[key] = {
                                'total_polarity': 0.0, 'total_subjectivity': 0.0, 'sentence_count': 0, 'outlet': outlet_name
                            }
                        temp_article_party_sentiment[key]['total_polarity'] += blob.sentiment.polarity
                        temp_article_party_sentiment[key]['total_subjectivity'] += blob.sentiment.subjectivity
                        temp_article_party_sentiment[key]['sentence_count'] += 1
                    except Exception as e:
                         logger.warning(f"TextBlob error on sentence: {sentence[:50]}... Error: {e}")

    for (link, party), data in temp_article_party_sentiment.items():
        if data['sentence_count'] > 0:
            avg_pol = data['total_polarity'] / data['sentence_count']
            avg_subj = data['total_subjectivity'] / data['sentence_count']
            party_sentiment_details.append({
                'link': link,
                'outlet': data['outlet'],
                'party': party,
                'avg_sentence_sentiment_textblob': avg_pol, # Specific name for polarity
                'avg_sentence_subjectivity_textblob': avg_subj,
                'sentence_count': data['sentence_count']
            })

    logger.info(f"Party sentence sentiment calculation (TextBlob) complete. Found {len(party_sentiment_details)} party mentions.")
    return party_sentiment_details


def categorize_and_aggregate_party_sentiment(party_sentiment_details, sentiment_col='avg_sentence_sentiment_textblob', tool_name='TextBlob'):
    # This function can be reused, just ensure sentiment_col matches the TextBlob output
    logger.info(f"Categorizing article sentiment towards parties using {tool_name} scores...")
    if not party_sentiment_details:
        logger.warning("No detailed party sentiment data provided for categorization.")
        return pd.DataFrame()

    df_details = pd.DataFrame(party_sentiment_details)
    if df_details.empty or sentiment_col not in df_details.columns:
        logger.error(f"Sentiment column '{sentiment_col}' not found or df_details is empty for TextBlob.")
        return pd.DataFrame()

    if tool_name == 'VADER': pos_thresh, neg_thresh = 0.05, -0.05
    elif tool_name == 'TextBlob': pos_thresh, neg_thresh = 0.1, -0.1 # Using polarity
    else: pos_thresh, neg_thresh = 0.05, -0.05
    def categorize(score):
        if score > pos_thresh: return 'Positive'
        elif score < neg_thresh: return 'Negative'
        else: return 'Neutral'
    df_details['sentiment_category'] = df_details[sentiment_col].apply(categorize)
    agg_counts = df_details.groupby(['outlet', 'party', 'sentiment_category']).size().unstack(fill_value=0)
    for cat in ['Positive', 'Neutral', 'Negative']:
        if cat not in agg_counts.columns: agg_counts[cat] = 0
    agg_counts.rename(columns={'Positive':'Positive_Count', 'Neutral':'Neutral_Count', 'Negative':'Negative_Count'}, inplace=True)
    agg_counts['Total_Articles_Mentioning'] = agg_counts['Positive_Count'] + agg_counts['Neutral_Count'] + agg_counts['Negative_Count']
    if 'Total_Articles_Mentioning' in agg_counts and agg_counts['Total_Articles_Mentioning'].sum() > 0 :
        agg_counts['Positive_Proportion'] = agg_counts['Positive_Count'] / agg_counts['Total_Articles_Mentioning']
        agg_counts['Neutral_Proportion'] = agg_counts['Neutral_Count'] / agg_counts['Total_Articles_Mentioning']
        agg_counts['Negative_Proportion'] = agg_counts['Negative_Count'] / agg_counts['Total_Articles_Mentioning']
    else:
        agg_counts['Positive_Proportion'] = 0; agg_counts['Neutral_Proportion'] = 0; agg_counts['Negative_Proportion'] = 0

    logger.info("Aggregation of categorized party sentiment complete (TextBlob).")
    return agg_counts[['Positive_Count','Neutral_Count','Negative_Count','Total_Articles_Mentioning', 'Positive_Proportion','Neutral_Proportion','Negative_Proportion']].copy()


def perform_ner(df, top_n=config.TOP_N_ENTITIES): # Identical to VADER version
    logger.info("Performing Named Entity Recognition (NER)...")
    if NLP is None: logger.error("spaCy model not loaded."); return None,None,None
    logger.warning("NER on 'processed_text'. Less accurate than original.")
    if df is None or 'processed_text' not in df.columns: logger.error("Invalid DF for NER."); return None,None,None
    ner_p_overall, ner_o_overall = Counter(), Counter()
    outlets = df['outlet'].unique()
    ner_per_out = {o: {'PERSON': Counter(), 'ORG': Counter()} for o in outlets}
    total_articles = len(df)
    for i, row in df.iterrows():
        if (i+1)%20==0 or i==total_articles-1: logger.info(f"NER article {i+1}/{total_articles}")
        o, txt = row['outlet'], row['processed_text']
        if not isinstance(txt, str) or not txt.strip(): continue
        try:
            doc = NLP(txt)
            for ent in doc.ents:
                ent_txt = ent.text.strip().lower()
                if len(ent_txt)>2 and not ent_txt.isdigit() and ent_txt not in config.COMBINED_STOPWORDS:
                    if ent.label_=='PERSON': ner_p_overall[ent_txt]+=1; ner_per_out[o]['PERSON'][ent_txt]+=1
                    elif ent.label_=='ORG': ner_o_overall[ent_txt]+=1; ner_per_out[o]['ORG'][ent_txt]+=1
        except Exception as e: logger.warning(f"spaCy NER error: {txt[:100]}... Error: {e}")
    top_p = ner_p_overall.most_common(top_n); top_o = ner_o_overall.most_common(top_n)
    logger.info("NER analysis complete.")
    return top_p, top_o, ner_per_out

def perform_topic_modeling(text_series, n_topics=config.N_TOPICS, n_top_words=10, model_type='lda'): # Identical
    logger.info(f"Topic Modeling ({model_type.upper()}) {n_topics} topics...")
    if text_series is None or text_series.empty or text_series.str.strip().eq('').all(): logger.warning("Empty text series for topics."); return None,None,None
    text_series = text_series.astype(str)[text_series.str.strip()!='']
    if text_series.empty: logger.warning("Text series empty after filter for topics."); return None,None,None
    if model_type=='lda':
        vec = CountVectorizer(max_df=0.90,min_df=5,stop_words=list(config.COMBINED_STOPWORDS),max_features=1000)
        model = LatentDirichletAllocation(n_components=n_topics,random_state=42,n_jobs=-1)
    elif model_type=='nmf':
        vec = TfidfVectorizer(max_df=0.90,min_df=5,stop_words=list(config.COMBINED_STOPWORDS),max_features=1000)
        model = NMF(n_components=n_topics,random_state=42,max_iter=400,init='nndsvda')
    else: logger.error(f"Invalid model_type '{model_type}'."); return None,None,None
    try:
        dtm = vec.fit_transform(text_series)
        if dtm.shape[0]==0 or dtm.shape[1]==0: logger.error("DTM empty for topics."); return None,None,None
        model.fit(dtm)
    except ValueError as e: logger.error(f"Error fitting topic model: {e}"); return None,None,None
    except Exception as e: logger.error(f"Unexpected error topic model: {e}"); return None,None,None
    logger.info("Topic model fit complete.")
    feats = vec.get_feature_names_out(); topics = {}
    for idx, topic_dist in enumerate(model.components_):
        top_w_idx = topic_dist.argsort()[:-n_top_words-1:-1]
        topics[idx] = " ".join([feats[i] for i in top_w_idx])
        logger.info(f"Topic {idx}: {topics[idx]}")
    logger.info("Topic modeling complete.")
    return model, vec, topics

def analyze_omission_proxy(df, omission_topics_keywords): # Identical
    logger.info("Omission Analysis Proxy...")
    if not omission_topics_keywords: logger.warning("OMISSION_TOPICS not defined."); return None
    if df is None or not {'outlet','processed_text'}.issubset(df.columns): logger.error("Invalid DF."); return None
    proc_kw = {}; lem = WordNetLemmatizer()
    for topic, kws in omission_topics_keywords.items(): proc_kw[topic] = [lem.lemmatize(kw.lower()) for kw in kws]
    om_counts = {o: Counter() for o in df['outlet'].unique()}
    for _, row in df.iterrows():
        o, txt = row['outlet'], row['processed_text']
        if not isinstance(txt,str) or not txt.strip(): continue
        words_art = set(txt.split())
        for topic, kws in proc_kw.items():
            if any(kw in words_art for kw in kws):
                if o in om_counts: om_counts[o][topic] += 1
    logger.info("Omission proxy complete.")
    return om_counts

def perform_statistical_tests(df, alpha=config.STAT_ALPHA): # Adapted for TextBlob 'polarity'
    logger.info("Performing Statistical Tests on TextBlob Polarity...")
    results = {}
    sentiment_col = 'polarity' # Key change for TextBlob
    if df is None or not {'outlet',sentiment_col}.issubset(df.columns) or df[sentiment_col].isnull().all():
        logger.error(f"DF invalid for stats on {sentiment_col}."); return results
    df_clean = df.dropna(subset=[sentiment_col])
    if len(df_clean) < 2: logger.warning("Not enough data for stats."); return results
    outlets = df_clean['outlet'].unique()
    if len(outlets) > 1:
        s_groups = [df_clean[sentiment_col][df_clean['outlet']==o] for o in outlets]
        s_groups_f = [g for g in s_groups if len(g) >= 2]
        if len(s_groups_f) > 1:
            logger.info(f"ANOVA on polarity across {len(s_groups_f)} outlets...")
            try:
                f, p = f_oneway(*s_groups_f)
                interp = f"Significant (p<{alpha})" if p<alpha else f"Not significant (p>={alpha})"
                results['ANOVA_polarity_by_outlet'] = {'f':f,'p':p,'interp':f"ANOVA ({sentiment_col}): {interp}."}
                logger.info(results['ANOVA_polarity_by_outlet'])
            except Exception as e: logger.error(f"Error ANOVA: {e}")
        else: logger.warning("ANOVA: Need >=2 outlets with >=2 articles.")
    # T-test and Chi2 logic would be identical to VADER's, just ensure using 'polarity' for T-test
    # For brevity, I'll skip rewriting the identical Chi2 part here, but it should be included.
    # T-test example:
    outlets_with_data = [o for o in outlets if len(df_clean[df_clean['outlet']==o]) >= 2]
    if len(outlets_with_data) >= 2:
        o1,o2 = outlets_with_data[0], outlets_with_data[1]
        g1,g2 = df_clean[sentiment_col][df_clean['outlet']==o1], df_clean[sentiment_col][df_clean['outlet']==o2]
        logger.info(f"T-test: {sentiment_col} ({o1} vs {o2})...")
        try:
            t_stat, p_value = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
            interp = f"Significant (p<{alpha})" if p_value<alpha else f"Not significant (p>={alpha})"
            results[f"TTest_polarity_{o1}_vs_{o2}"] = {'t':t_stat,'p':p_value,'interp':f"T-test polarity ({o1} vs {o2}): {interp}."}
            logger.info(results[f"TTest_polarity_{o1}_vs_{o2}"])
        except Exception as e: logger.error(f"Error T-test polarity ({o1} vs {o2}): {e}")

    # Chi-squared (identical logic as vader script, ensure term_counts_per_outlet is generated)
    term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)
    if term_counts_per_outlet:
         cat1, cat2 = 'pro_bjp_modi', 'critical_bjp_modi' # Example categories
         ct_data, valid_outlets_chi2 = [], []
         for o, cats_data in term_counts_per_outlet.items():
             c1 = sum(cats_data.get(cat1, Counter()).values())
             c2 = sum(cats_data.get(cat2, Counter()).values())
             if c1 > 0 or c2 > 0: ct_data.append([c1,c2]); valid_outlets_chi2.append(o)
         if len(ct_data) >= 2 and np.array(ct_data).shape[1] == 2:
            logger.info(f"Chi2 test: '{cat1}' vs '{cat2}' across {len(valid_outlets_chi2)} outlets...")
            try:
                chi2,p,dof,_=chi2_contingency(np.array(ct_data))
                interp = f"Significant (p<{alpha})" if p<alpha else f"Not sig (p>={alpha})"
                results['Chi2_BJP_Terms_Outlet_TextBlob'] = {'chi2':chi2,'p':p,'dof':dof,'interp':f"Chi2 ('{cat1}' vs '{cat2}'): {interp}."}
                logger.info(results['Chi2_BJP_Terms_Outlet_TextBlob'])
            except ValueError as e: logger.error(f"Error Chi2 TextBlob (low freq?): {e}")
            except Exception as e: logger.error(f"Error Chi2 TextBlob: {e}")
         else: logger.warning(f"Chi2 TextBlob '{cat1}' vs '{cat2}': Need >=2 outlets with data.")

    logger.info("Statistical tests complete (TextBlob).")
    return results

def save_plot(plt_object, filename_suffix): # Adapted for TextBlob prefix
    try:
        filepath = os.path.join(config.OUTPUT_DIR, f"textblob_{filename_suffix}")
        plt_object.savefig(filepath); logger.info(f"Plot saved to {filepath}"); plt_object.close()
    except Exception as e: logger.error(f"Error saving plot {filepath}: {e}"); plt_object.close()


def generate_visualizations(df_sentiment_scores, avg_sentiment_outlet, most_common_overall,
                            ner_persons_overall, ner_orgs_overall,
                            party_sentiment_details, agg_party_counts, # New params
                            topics_lda, topics_nmf, omission_counts):
    logger.info("Generating visualizations (TextBlob)...")
    sentiment_col = 'polarity' # For TextBlob
    tool_name = 'TextBlob'

    if df_sentiment_scores is not None and sentiment_col in df_sentiment_scores.columns and not df_sentiment_scores[sentiment_col].isnull().all():
        plt.figure(figsize=(14,8)); sns.boxplot(data=df_sentiment_scores,x='outlet',y=sentiment_col,palette='viridis')
        plt.title(f'{tool_name} Polarity Score Distribution by Outlet',fontsize=16); plt.ylabel(f'{tool_name} Polarity'); plt.xlabel('Outlet')
        plt.xticks(rotation=45,ha='right'); plt.tight_layout(); save_plot(plt,'polarity_dist_outlet.png')
        if not avg_sentiment_outlet.empty:
            plt.figure(figsize=(12,7)); avg_sentiment_outlet.plot(kind='bar',color=sns.color_palette("viridis",len(avg_sentiment_outlet)))
            plt.title(f'Average {tool_name} Polarity by Outlet',fontsize=16); plt.ylabel(f'Average {tool_name} Polarity'); plt.xlabel('')
            plt.xticks(rotation=45,ha='right'); plt.axhline(0,color='grey',ls='--'); plt.tight_layout(); save_plot(plt,'avg_polarity_outlet.png')
        if 'subjectivity' in df_sentiment_scores.columns and not df_sentiment_scores['subjectivity'].isnull().all():
            plt.figure(figsize=(14,8)); sns.boxplot(data=df_sentiment_scores,x='outlet',y='subjectivity',palette='magma')
            plt.title(f'{tool_name} Subjectivity Score Distribution by Outlet',fontsize=16); plt.ylabel(f'{tool_name} Subjectivity'); plt.xlabel('Outlet')
            plt.xticks(rotation=45,ha='right'); plt.tight_layout(); save_plot(plt,'subjectivity_dist_outlet.png')

    # Word cloud, NER, Omission plots are identical in logic, just use the textblob_ prefix via save_plot
    if most_common_overall:
        try:
            wc=WordCloud(width=1200,height=600,background_color='white',max_words=100,colormap='magma').generate_from_frequencies(dict(most_common_overall))
            plt.figure(figsize=(15,7)); plt.imshow(wc,interpolation='bilinear'); plt.axis('off')
            plt.title('Overall Top Words',fontsize=16); plt.tight_layout(); save_plot(plt,'wordcloud_overall.png')
        except Exception as e: logger.error(f"Error generating overall word cloud (TextBlob): {e}")
    if ner_persons_overall:
        df_p=pd.DataFrame(ner_persons_overall,columns=['Person','Count']); plt.figure(figsize=(12,max(8,len(df_p)*0.4)))
        sns.barplot(data=df_p,y='Person',x='Count',palette='coolwarm'); plt.title(f'Top PERSON Entities',fontsize=16)
        plt.tight_layout(); save_plot(plt,'ner_top_persons.png')
    if ner_orgs_overall:
        df_o=pd.DataFrame(ner_orgs_overall,columns=['Organization','Count']); plt.figure(figsize=(12,max(8,len(df_o)*0.4)))
        sns.barplot(data=df_o,y='Organization',x='Count',palette='coolwarm'); plt.title(f'Top ORGANIZATION Entities',fontsize=16)
        plt.tight_layout(); save_plot(plt,'ner_top_orgs.png')

    # New Party Sentiment Plots for TextBlob
    if not agg_party_counts.empty:
        plot_party_sentiment_categories(agg_party_counts, tool_name=tool_name, plot_type='proportion', prefix="textblob_")
        plot_party_sentiment_categories(agg_party_counts, tool_name=tool_name, plot_type='count', prefix="textblob_")

    if party_sentiment_details: # Ensure this contains the 'avg_sentence_sentiment_textblob' column
        plot_party_sentiment_distributions(party_sentiment_details, sentiment_col='avg_sentence_sentiment_textblob', tool_name=tool_name, prefix="textblob_")

    if omission_counts: # Identical logic
        om_df=pd.DataFrame.from_dict(omission_counts,orient='index').fillna(0)
        if not om_df.empty:
            om_df.plot(kind='bar',figsize=(15,8),colormap='tab20')
            plt.title('Omission Proxy: Topic Mentions/Outlet',fontsize=16);plt.xlabel('Outlet');plt.ylabel('# Articles Mentioning Topic')
            plt.xticks(rotation=45,ha='right');plt.legend(title='Topics',bbox_to_anchor=(1.05,1),loc='upper left');plt.tight_layout()
            save_plot(plt,'omission_proxy_mentions.png')
    logger.info("Visualizations generation complete (TextBlob).")


def plot_party_sentiment_categories(agg_party_counts, tool_name='TextBlob', plot_type='proportion', prefix=""):
    # Reusing the VADER plot function structure, just ensure 'prefix' is used in save_plot
    if agg_party_counts.empty: logger.warning(f"{prefix}No aggregated party sentiment counts to plot."); return
    cols=['Positive_Proportion','Neutral_Proportion','Negative_Proportion'] if plot_type=='proportion' else ['Positive_Count','Neutral_Count','Negative_Count']
    ylab='Proportion of Articles' if plot_type=='proportion' else 'Number of Articles'; title_sfx = 'Proportions' if plot_type=='proportion' else 'Counts'
    parties = agg_party_counts.index.get_level_values('party').unique()
    for party in parties:
        party_data = agg_party_counts.xs(party, level='party')
        if party_data.empty: continue
        plt.figure(figsize=(max(10,len(party_data.index)*0.8),7))
        party_data[cols].plot(kind='bar',stacked=True,colormap='coolwarm_r',ax=plt.gca())
        plt.title(f'{tool_name} Sentiment Towards {party} by Outlet ({title_sfx})',fontsize=16);plt.xlabel('Outlet');plt.ylabel(ylab)
        plt.xticks(rotation=45,ha='right');plt.legend(title='Sentiment',bbox_to_anchor=(1.05,1),loc='upper left');plt.tight_layout(rect=[0,0,0.85,1])
        safe_party_name = re.sub(r'\W+','_',party); save_plot(plt,f'{prefix}party_sentiment_categories_{safe_party_name}_{plot_type}.png')

def plot_party_sentiment_distributions(party_sentiment_details, sentiment_col, tool_name='TextBlob', prefix=""):
    # Reusing the VADER plot function structure
    if not party_sentiment_details: logger.warning(f"{prefix}No detailed party sentiment data for distribution plots."); return
    df_details = pd.DataFrame(party_sentiment_details)
    if df_details.empty or sentiment_col not in df_details.columns: logger.error(f"{prefix}Invalid data for distribution plots."); return
    parties = df_details['party'].unique()
    for party in parties:
        party_data = df_details[df_details['party']==party]
        if party_data.empty or party_data[sentiment_col].isnull().all(): continue
        plt.figure(figsize=(14,7))
        sns.violinplot(data=party_data, x='outlet', y=sentiment_col, palette='viridis', inner='quartile')
        plt.title(f'{tool_name} Sentiment Dist. Towards {party} by Outlet',fontsize=16);plt.ylabel(f'{tool_name} Score (Avg Sentence)');plt.xlabel('Outlet')
        plt.axhline(0,color='grey',ls='--');plt.xticks(rotation=45,ha='right');plt.tight_layout()
        safe_party_name = re.sub(r'\W+','_',party); save_plot(plt,f'{prefix}party_sentiment_distribution_{safe_party_name}.png')


def save_results(df_scores, avg_sent_outlet, common_overall, term_overall, term_per_outlet,
                 ner_p_overall, ner_o_overall, ner_per_out,
                 party_sent_details, agg_party_cnts, # New params
                 topics_lda, topics_nmf, stat_res, omission_cnts, prefix="textblob_"): # Default prefix
    logger.info(f"Saving {prefix}analysis results...")
    out_dir = config.OUTPUT_DIR # Uses global config

    # Save main dataframe with sentiment (polarity and subjectivity for TextBlob)
    if df_scores is not None: pd.DataFrame(df_scores).to_csv(os.path.join(out_dir, f'{prefix}articles_with_sentiment.csv'), index=False)
    # Save aggregate sentiment (polarity)
    if not avg_sent_outlet.empty: avg_sent_outlet.to_csv(os.path.join(out_dir, f'{prefix}avg_polarity_outlet.csv'), header=[f'avg_polarity'])

    # Identical saving logic for common_overall, term_counts, ner_results, topic_results, stat_results, omission_counts
    if common_overall: pd.DataFrame(common_overall,columns=['word','count']).to_csv(os.path.join(out_dir,f'{prefix}top_words_overall.csv'),index=False)
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
    logger.info("--- Starting TextBlob Analysis Pipeline ---")
    df = load_data(config.PREPROCESSED_ARTICLES_CSV)

    if df is not None and not df.empty:
        df = perform_textblob_sentiment_analysis(df) # Use TextBlob specific function
        avg_sentiment_outlet = df.groupby('outlet')['polarity'].mean().sort_values() if 'polarity' in df.columns else pd.Series(dtype=float)
        most_common_overall = analyze_word_frequency(df['processed_text'])
        term_counts_overall, term_counts_per_outlet = analyze_ideological_term_frequency(df, config.IDEOLOGICAL_TERMS)

        # Updated Party Sentiment Analysis for TextBlob
        party_sentiment_details_textblob = analyze_sentiment_towards_parties_textblob(df, config.PARTY_TERMS)
        agg_party_counts_textblob = categorize_and_aggregate_party_sentiment(
            party_sentiment_details_textblob,
            sentiment_col='avg_sentence_sentiment_textblob', # Ensure this matches output key
            tool_name='TextBlob'
        )

        ner_persons_overall, ner_orgs_overall, ner_per_outlet = perform_ner(df)
        lda_model, lda_vectorizer, topics_lda = perform_topic_modeling(df['processed_text'], model_type='lda')
        nmf_model, nmf_vectorizer, topics_nmf = perform_topic_modeling(df['processed_text'], model_type='nmf')
        omission_counts = analyze_omission_proxy(df, config.OMISSION_TOPICS)
        stat_results = perform_statistical_tests(df) # Uses 'polarity' by default for TextBlob now

        generate_visualizations(df, avg_sentiment_outlet, most_common_overall,
                                ner_persons_overall, ner_orgs_overall,
                                party_sentiment_details_textblob, agg_party_counts_textblob,
                                topics_lda, topics_nmf, omission_counts)

        save_results(df, avg_sentiment_outlet, most_common_overall, term_counts_overall, term_counts_per_outlet,
                     ner_persons_overall, ner_orgs_overall, ner_per_outlet,
                     party_sentiment_details_textblob, agg_party_counts_textblob,
                     topics_lda, topics_nmf, stat_results, omission_counts, prefix="textblob_")
    else:
        logger.error("Failed to load or process data. Exiting TextBlob analysis.")
    logger.info("--- Finished TextBlob Analysis Pipeline ---")