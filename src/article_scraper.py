import csv
import newspaper
import time
import random
import logging
import os
import requests

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
log_handler = logging.FileHandler(config.SCRAPER_LOG_FILE)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set logging level (e.g., INFO, DEBUG)
logger.addHandler(log_handler)
# Optional: Add console handler to see logs in terminal as well
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# logger.addHandler(console_handler)

# --- Newspaper3k Configuration ---
# To handle sites blocking scrapers (403 Forbidden) and timeouts
news_config = newspaper.Config()
news_config.browser_user_agent = config.SCRAPER_USER_AGENT
news_config.request_timeout = config.SCRAPER_TIMEOUT
news_config.fetch_images = False # Usually not needed, saves time
news_config.memoize_articles = False # Avoid caching issues during development/reruns

# --- Main Scraping Function ---
def scrape_articles():
    """
    Reads links from the input CSV, scrapes article text using newspaper3k,
    handles errors, logs progress, and saves data to the output CSV.
    Removed static bias mapping.
    """
    logger.info("--- Starting Article Scraping ---")
    data = []
    processed_links = set() # To avoid processing duplicates if any

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    try:
        with open(config.ARTICLE_LINKS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Verify header
            if not {'outlet', 'category', 'link'}.issubset(reader.fieldnames):
                logger.error(f"Input CSV {config.ARTICLE_LINKS_CSV} must contain 'outlet', 'category', 'link' columns.")
                return
            links_to_process = list(reader)
    except FileNotFoundError:
        logger.error(f"Input file not found: {config.ARTICLE_LINKS_CSV}")
        return
    except Exception as e:
        logger.error(f"Error reading input CSV {config.ARTICLE_LINKS_CSV}: {e}")
        return

    logger.info(f"Found {len(links_to_process)} links to process.")

    for i, row in enumerate(links_to_process):
        outlet = row.get('outlet', '').strip()
        category = row.get('category', '').strip()
        link = row.get('link', '').strip()

        # Basic validation
        if not outlet or not link or not link.startswith(('http://', 'https://')):
            logger.warning(f"Skipping invalid row {i+1}: Outlet='{outlet}', Link='{link}'")
            continue

        if link in processed_links:
            logger.warning(f"Skipping duplicate link: {link}")
            continue

        logger.info(f"Processing {i+1}/{len(links_to_process)}: {outlet} - {category}: {link}")
        processed_links.add(link)

        # Create a newspaper Article object with the custom config
        article = newspaper.Article(link, config=news_config)

        text = ""
        try:
            # Download and parse the article
            article.download()
            article.parse()
            text = article.text.strip()

            # Content Validation
            if text and len(text.split()) >= config.MIN_ARTICLE_WORD_COUNT:
                data.append({
                    "outlet": outlet,
                    "category": category,
                    "link": link,
                    "text": text
                    # No 'bias' column here anymore
                })
                logger.info(f"Successfully extracted {len(text.split())} words.")
            elif not text:
                 logger.warning(f"No text extracted from article: {link}")
            else:
                logger.warning(f"Extracted text too short ({len(text.split())} words) for: {link}")

        except newspaper.ArticleException as e:
            logger.error(f"Newspaper3k error extracting article {link}: {e}")
        except requests.exceptions.Timeout:
             logger.error(f"Timeout error extracting article {link} (Timeout={config.SCRAPER_TIMEOUT}s)")
        except requests.exceptions.HTTPError as e:
             logger.error(f"HTTP error extracting article {link}: {e.response.status_code} {e.response.reason}")
        except requests.exceptions.RequestException as e:
             logger.error(f"Network error extracting article {link}: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error extracting article {link}: {e}", exc_info=True) # Log traceback

        # Add a randomized delay to avoid overwhelming servers
        sleep_time = random.uniform(config.SCRAPER_SLEEP_MIN, config.SCRAPER_SLEEP_MAX)
        logger.debug(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    # Write extracted data to output CSV
    if data:
        try:
            with open(config.SCRAPED_ARTICLES_CSV, 'w', newline='', encoding='utf-8') as f:
                # Define fieldnames based on the keys in the first data dictionary
                fieldnames = list(data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Data collection complete. {len(data)} articles saved to {config.SCRAPED_ARTICLES_CSV}")
        except Exception as e:
             logger.error(f"Error writing output CSV {config.SCRAPED_ARTICLES_CSV}: {e}")
    else:
        logger.warning("No articles successfully scraped or passed validation. Output CSV not created.")

    logger.info("--- Finished Article Scraping ---")


if __name__ == "__main__":
    scrape_articles()