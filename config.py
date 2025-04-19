import os
import nltk
from nltk.corpus import stopwords

# --- Project Structure ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is in the project root

# --- File Paths ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Input data
ARTICLE_LINKS_CSV = os.path.join(DATA_DIR, 'article-links.csv') # Assuming you switch from docx

# Intermediate data
SCRAPED_ARTICLES_CSV = os.path.join(DATA_DIR, 'articles.csv')
PREPROCESSED_ARTICLES_CSV = os.path.join(DATA_DIR, 'preprocessed_articles.csv')

# Output files (will be created in OUTPUT_DIR)
# Analysis scripts will define their specific output files using OUTPUT_DIR

# Log files (will be created in LOG_DIR)
SCRAPER_LOG_FILE = os.path.join(LOG_DIR, 'scraper.log')
PREPROCESS_LOG_FILE = os.path.join(LOG_DIR, 'preprocess.log')
ANALYSIS_VADER_LOG_FILE = os.path.join(LOG_DIR, 'analysis_vader.log')
ANALYSIS_TEXTBLOB_LOG_FILE = os.path.join(LOG_DIR, 'analysis_textblob.log')
PIPELINE_LOG_FILE = os.path.join(LOG_DIR, 'pipeline.log')


# --- Scraping Parameters ---
SCRAPER_TIMEOUT = 15  # Increased timeout in seconds
SCRAPER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
SCRAPER_SLEEP_MIN = 1.0 # Minimum sleep time between requests
SCRAPER_SLEEP_MAX = 2.5 # Maximum sleep time between requests
MIN_ARTICLE_WORD_COUNT = 50 # Minimum words for an article to be considered valid

# --- Preprocessing Parameters ---
KEEP_NUMBERS = True # Set to False to remove numbers during preprocessing

# --- Analysis Parameters ---
TOP_N_WORDS = 30 # Number of top words for frequency analysis
TOP_N_ENTITIES = 20 # Number of top entities for NER
N_TOPICS = 10 # Number of topics for LDA/NMF
STAT_ALPHA = 0.05 # Significance level for statistical tests

# --- Stopwords ---
# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

ENGLISH_STOPWORDS = set(stopwords.words('english'))
# Add common, often high-frequency words specific to this context that might obscure analysis
# Refine this list based on initial results
CUSTOM_STOPWORDS = {
    'delhi', 'election', 'bjp', 'aap', 'congress', 'party', 'govt', 'government',
    'state', 'chief', 'minister', 'leader', 'vote', 'voter', 'seat', 'poll',
    'campaign', 'result', 'kejriwal', 'modi', 'rekha', 'gupta', 'atishi', 'verma',
    'said', 'also', 'would', 'could', 'like', 'one', 'two', 'year', 'people',
    'new', 'time', 'make', 'even', 'get', 'city', 'capital', 'assembly'
    # Consider adding outlet names if they appear frequently in text: 'express', 'hindu', etc.
}
COMBINED_STOPWORDS = ENGLISH_STOPWORDS.union(CUSTOM_STOPWORDS)


# --- Ideological & Political Terms ---
# Carefully curate and justify these lists in your research paper
IDEOLOGICAL_TERMS = {
    "pro_bjp_modi": [
        "masterstroke", "visionary leadership", "modi’s resolve", "modi magic",
        "strong governance", "development agenda", "nation building", "double engine", "double-engine",
        "governance efficiency", "economic growth", "reform" , "nationalism", "hindutva",
        "decisive mandate", "resounding victory", "sabka saath", "sabka vikas"
    ],
    "pro_aap_kejriwal": [
        "aam aadmi", "common man", "welfare", "governance model", "delhi model",
        "education reform", "healthcare reform", "mohalla clinic", "anti-corruption",
        "people's mandate", "disruptor", "alternative politics", "clean governance",
        "free electricity", "free water", "honest politics"
    ],
    "critical_bjp_modi": [
        "authoritarian", "centralized control", "erosion of democracy", "unilateral decision",
        "press freedom curbed", "dissent silenced", "government overreach", "censorship",
        "communal", "polarisation", "majoritarian", "institutional bias", "power grab",
        "job crisis", "economic slowdown", "threat to constitution"
    ],
    "critical_aap_kejriwal": [
        "anarchy", "populism", "freebie", "revdi", "u-turn", "confrontational politics",
        "liquor scam", "corruption", "mismanagement", "sheesh mahal", "false promise",
        "governance failure", "hollow promise", "doublespeak", "theatrics", "blame game"
    ],
    "neutral_governance": [
        "voter turnout", "urban planning", "fiscal allocation", "policy continuity",
        "electoral outcome", "citizen response", "bureaucratic reshuffle", "infrastructure",
        "civic amenities", "administration", "poll promises", "lieutenant governor", "lg",
        "municipal corporation", "mcd", "manifesto", "mandate"
    ],
    "emotive_political": [
        "political maneuvering", "mobilization tactics", "high-stakes", "battleground",
        "power grab", "betrayal", "scandal", "controversy", "decisive mandate", "rout",
        "setback", "spectacular victory", "existential crisis", "decimation", "thumping majority",
        "landslide", "debacle", "resounding verdict"
    ]
}

PARTY_TERMS = {
    'BJP': ['bjp', 'bharatiya janata party', 'modi', 'shah', 'yogi', 'adityanath', 'nadda', 'saffron party', 'rekha gupta', 'parvesh verma', 'vijender gupta'],
    'AAP': ['aap', 'aam aadmi party', 'kejriwal', 'atishi', 'sisodia', 'common man party', 'gopal rai', 'sanjay singh'],
    'Congress': ['congress', 'inc', 'rahul gandhi', 'gandhi', 'dikshit', 'sheila dikshit', 'sandeep dikshit', 'kharge']
    # Add other relevant parties/leaders if needed: 'tmc', 'sp', 'rjd', etc. if they feature in coverage
}

# --- spaCy Model ---
SPACY_MODEL = "en_core_web_sm" # Small model, efficient. Use 'md' or 'lg' for higher accuracy if needed.

# --- Omission Analysis Keywords (Example - Requires careful definition) ---
# Define key events/topics and associated keywords. Structure can be:
# OMISSION_TOPICS = {
#     "Liquor_Policy_Scam": ["liquor", "excise", "policy", "scam", "cbi", "ed", "arrest", "sisodia", "kejriwal"],
#     "Yamuna_Pollution": ["yamuna", "pollution", "river", "water quality", "foam", "toxic", "cleaning"],
#     "Air_Quality": ["air quality", "aqi", "pollution", "smog", "stubble burning", "pollution control"],
#     # Add more topics relevant to the election period
# }
OMISSION_TOPICS = {} # Start empty, populate based on actual election events