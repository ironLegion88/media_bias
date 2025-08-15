# AI-Powered Media Bias Analysis for Electoral Coverage

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

An end-to-end Natural Language Processing (NLP) pipeline designed to identify and quantify ideological bias in online news articles covering the 2025 Delhi Assembly Elections. This project moves beyond simple sentiment analysis to provide a multi-faceted view of media framing, source attribution, and thematic focus.

The core contribution is a novel methodology for analyzing party-specific sentiment by aggregating sentence-level scores, which avoids the neutralization effect common in full-article averaging and provides a more accurate, granular view of media slant.

---

## Key Features

- **Automated Content Scraping:** A robust scraper to fetch article text from a list of URLs, with error handling and logging.
- **Advanced NLP Preprocessing:** A configurable text cleaning pipeline to prepare data for analysis.
- **Multi-Dimensional Bias Analysis:** Measures media bias across several key dimensions (see below).
- **Comparative Sentiment Analysis:** Implements and compares both VADER and TextBlob for a comprehensive understanding of sentiment.
- **Unsupervised Topic Modeling:** Utilizes Scikit-Learn's LDA and NMF to uncover latent themes and frames in election coverage.
- **Named Entity Recognition (NER):** Employs spaCy to identify and analyze the sources (people, organizations) cited in articles.
- **Statistical Validation:** Uses SciPy for statistical tests (ANOVA, Chi-squared) to validate the significance of observed differences.
- **Automated Pipeline:** A master script (`run_pipeline.py`) executes the entire workflow from scraping to analysis and visualization.

---

## Analysis Dimensions

This project quantifies bias by focusing on five key dimensions derived from media analysis research:

1.  **Language and Wording:** Frequency analysis of ideologically charged terms.
2.  **Tone and Sentiment:** Comparative sentiment analysis of articles and specific political entities (parties, candidates).
3.  **Framing of Issues:** Unsupervised topic modeling (LDA/NMF) to identify how different outlets frame the same events.
4.  **Source Attribution:** NER to extract and analyze the types of sources (e.g., government, opposition, independent) each outlet cites.
5.  **Omission (Proxy):** A keyword-based proxy to track which outlets cover specific key events or controversies.

---

## Tech Stack

| Component           | Technologies & Libraries                                          |
| ------------------- | ----------------------------------------------------------------- |
| **Data Scraping**   | `Python`, `newspaper3k`, `requests`                               |
| **Data Processing** | `Pandas`, `NumPy`                                                 |
| **NLP & ML**        | `Scikit-Learn` (LDA, NMF, Vectorizers), `spaCy` (NER), `NLTK`, `VADER`, `TextBlob` |
| **Statistics**      | `SciPy`                                                           |
| **Visualization**   | `Matplotlib`, `Seaborn`, `WordCloud`                              |
| **Workflow**        | `subprocess`, `logging`, `os`                                     |

---

## Project Structure

```
media-bias-analysis/
├── data/                    # -> Houses input/output CSVs (links, scraped, preprocessed)
├── src/                     # -> All Python source code modules
│   ├── __init__.py
│   ├── article_scraper.py   # -> Scrapes articles from URLs
│   ├── preprocessing.py     # -> Cleans and preprocesses text data
│   ├── analysis_vader.py    # -> Full analysis pipeline using VADER
│   ├── analysis_textblob.py # -> Full analysis pipeline using TextBlob
├── analysis_results/        # -> All generated outputs (plots, JSONs, result CSVs)
├── logs/                    # -> Log files for each script, tracking progress and errors
├── .gitignore               # -> Specifies files/directories to ignore in version control
├── config.py                # -> Central configuration for paths, parameters, keywords, etc.
├── run_pipeline.py          # -> Master script to execute the entire pipeline
├── requirements.txt         # -> Project dependencies
└── README.md                # -> This file
```

---

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/media-bias-analysis.git
    cd media-bias-analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download necessary NLP models:**
    ```bash
    # Download spaCy model for NER
    python -m spacy download en_core_web_sm

    # The analysis scripts will automatically download required NLTK models on first run.
    ```

5.  **Prepare Input Data:**
    -   Add your target article URLs to `data/article-links.csv`. The file must contain `outlet`, `category`, and `link` columns.

---

## Usage

Execute the entire pipeline from scraping to analysis with a single command from the project root directory. The master script will run each module in the correct order.

```bash
python run_pipeline.py
```

-   The script will provide console output on its progress.
-   Detailed logs for each step are saved in the `/logs` directory.
-   All generated CSVs, JSON files, and plots are saved in the `/analysis_results` directory.

---

## Methodology Highlight: Party Sentiment Analysis

A key challenge in sentiment analysis is the **neutralization effect**, where an article with strong positive and strong negative statements about a party averages out to a neutral score. This project mitigates this by:

1.  **Sentence-Level Analysis:** Identifying only the sentences that explicitly mention a political party or its key members.
2.  **Per-Article Aggregation:** Calculating an average sentiment score for *each party* within a *single article* based on only those relevant sentences.
3.  **Categorization:** Classifying each article's stance towards a party as 'Positive', 'Neutral', or 'Negative' based on this calculated score.
4.  **Outlet-Level Aggregation:** Counting the number and proportion of articles from each outlet that fall into these three categories for each party, providing a clear and nuanced view of media slant.

---

## Limitations and Future Work

-   **Scope:** The analysis is currently limited to English-language media.
-   **Omission Proxy:** The omission analysis is a simplified proxy based on keywords and could be improved with more advanced event-detection models.
-   **NER Accuracy:** NER is performed on preprocessed text for simplicity, which may slightly reduce accuracy compared to running it on raw text.
-   **Future Work:**
    -   Develop an interactive dashboard using **Dash** or **Streamlit** to explore the results.
    -   Incorporate **Aspect-Based Sentiment Analysis (ABSA)** for even more granular insights.
    -   Fine-tune a transformer-based model (e.g., BERT) for domain-specific sentiment classification.
    -   Expand the analysis to include Hindi and other regional-language media.
