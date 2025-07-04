import streamlit as st
import asyncio
import aiohttp
from datetime import datetime, timedelta
import urllib.parse
import feedparser
from bs4 import BeautifulSoup
import gc
import re # For regex-based cleaning

# --- Extractive Summarization Imports ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Constants ---
TTL_HOURS = 24  # Cache TTL for fetched news data in hours
MIN_RSS_SUMMARY_LENGTH = 50  # Minimum length for RSS summary to be considered sufficient
MIN_SCRAPED_CONTENT_LENGTH = 250 # Minimum length for scraped content to be considered meaningful
MAX_ARTICLES_TO_PROCESS = 150 # Max articles to process to conserve resources

# ----------------------------
# SETUP
# ----------------------------

# Load extractive summarizer components - cached to load only once
@st.cache_resource
def load_extractive_summarizer_components():
    """Loads and caches the TextRank summarizer and tokenizer."""
    language = "english"
    stemmer = Stemmer(language)
    summarizer_extractive = TextRankSummarizer(stemmer=stemmer)
    summarizer_extractive.stop_words = get_stop_words(language)
    tokenizer = Tokenizer(language)
    return summarizer_extractive, tokenizer

EXT_SUMMARIZER, EXT_TOKENIZER = load_extractive_summarizer_components()

# --- RSS Feed URLs (UPDATED, PRIORITIZING DIRECT & STABLE FEEDS) ---
# Ensure these feeds are regularly checked for validity and content
RSS_FEED_URLS = [
    "https://www.eia.gov/rss/todayinenergy.xml",
    "https://www.eia.gov/rss/press_rss.xml",
    "https://www.energy.gov/fecm/oil-natural-gas.xml",
    "https://www.ogj.com/__rss/website-scheduled-content.xml?input=%7B%22sectionAlias%22%3A%22general-interest%22%7D",
    "https://www.ogj.com/__rss/website-scheduled-content.xml?input=%7B%22sectionAlias%22%3A%22exploration-development%22%7D",
    "https://www.ogj.com/__rss/website-scheduled-content.xml?input=%7B%22sectionAlias%22%3A%22drilling-production%22%7D",
    "https://www.ogj.com/__rss/website-scheduled-content.xml?input=%7B%22sectionAlias%22%3A%22refining-processing%22%7D",
    "https://www.ogj.com/__rss/website-scheduled-content.xml?input=%7B%22sectionAlias%22%3A%22pipelines-transportation%22%7D",
    "https://www.ogj.com/__rss/website-scheduled-content.xml?input=%7B%22sectionAlias%22%3A%22energy-transition%22%7D",
    "https://www.worldoil.com/rss?feed=news",
    "https://www.worldoil.com/rss?feed=topic:offshore",
    "https://www.worldoil.com/rss?feed=topic:exploration",
    "https://www.worldoil.com/rss?feed=topic:production",
    "https://www.worldoil.com/rss?feed=topic:drilling",
    "https://www.worldoil.com/rss?feed=topic:completion",
    "https://www.worldoil.com/rss?feed=topic:india",
    "https://www.worldoil.com/rss?feed=topic:middle+east",
    "https://www.worldoil.com/rss?feed=topic:saudi+arabia",
    "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
    "https://www.rigzone.com/news/rss/rigzone_exploration.aspx",
    "https://www.rigzone.com/news/rss/rigzone_production.aspx",
    "https://energy.economictimes.indiatimes.com/rss/topstories",
    "https://energy.economictimes.indiatimes.com/rss/oil-and-gas",
    "https://www.spglobal.com/commodityinsights/en/rss-feed/oil",
    "https://www.spglobal.com/commodityinsights/en/rss-feed/natural-gas",
    "https://www.spglobal.com/commodityinsights/en/rss-feed/petrochemicals",
    "https://www.spglobal.com/commodityinsights/en/rss-feed/lng",
]

# --- Keyword Definitions (All Lowercase for Consistency) ---
LOCAL_KEYWORDS = [
    "india", "ongc", "gail", "iocl", "reliance", "bharat petroleum", "mumbai",
    "gujarat", "delhi", "chennai", "kolkata", "bangalore", "mangalore",
    "ghats", "bay of bengal", "indian ocean", "indian", "modi",
    "oil india", "adani", "tata", "vedanta", "jiobp", "assam", "kochi",
    "vishakapatnam", "odisha", "maharashtra", "bengal", "ministry of petroleum",
    "domestic", "ahmedabad", "vadodara", "kerala", "andhra pradesh", "uttar pradesh",
    "sikkim", "himachal", "punjab", "haryana", "rajasthan", "madhya pradesh",
    "chhattisgarh", "jharkhand", "bihar", "west bengal", "goa", "karnataka",
    "tamil nadu", "telangana", "maharashtra", "gujarat", "odisha", "andaman",
    "lakshadweep", "puducherry", "chandigarh", "delhi ncr", "northeast india",
    "south india", "north india", "east india", "west india", "offshore india",
    "dgh", "regulator", "indian market"
]

GLOBAL_KEYWORDS = [
    "opec", "saudi", "brent", "shell", "exxon", "russia", "middle east",
    "usa", "europe", "china", "canada", "nigeria", "global", "international",
    "crude", "u.s.", "us", "norway", "france", "italy", "uk", "united kingdom",
    "united states of america", "mexico", "brazil", "australia", "tanzania",
    "korea", "iran", "iraq", "japan", "uae", "qatar", "venezuela", "america",
    "trump", "north sea", "africa", "latin", "kazakhstan", "afghanistan",
    "total energies", "indonesia", "vietnam", "pakistan", "cnpc", "petronas",
    "malaysia", "petrobras", "iea", "world", "norwegian sea", "israel",
    "gcc", "mena", "asean", "latin america", "gulf of mexico", "mediterranean",
    "black sea", "arctic", "persian gulf", "azerbaijan", "egypt", "algeria",
    "libya", "angola", "ecuador", "colombia", "argentina", "japan", "south korea",
    "thailand", "cambodia", "philippines", "brunei", "myanmar", "east timor",
    "pakistan", "bangladesh", "turkey", "poland", "chevron", "bp", "eni", "equinor",
    "saudi aramco", "conocophillips", "occidental petroleum", "pioneer natural resources",
    "gazprom", "jodi", "cop", "geopolitics", "sanctions", "trade", "exports", "imports",
    "supply chain", "global supply", "global demand", "world market", "california", 
    "los angeles", "la", "texas", "totalenergies"
]

UPSTREAM_KW = [
    "exploration", "drilling", "seismic", "rig", "deepwater",
    "offshore", "well", "reservoir", "discovery", "production",
    "e&p", "upstream", "oilfield", "gasfield", "shale", "fracking",
    "hydrocarbon", "reserves", "geology", "exploration license",
    "drilling permit", "finding", "appraisal", "development", "subsea",
    "crude", "quota", "oeuk", "offshore energies",
    "electrolyzer", "electrolysis", "green hydrogen production", "blue hydrogen production",
    "carbon capture", "geothermal", "wind farm", "solar farm",
    "takeover", "merger", "acquisition", "deal", # For corporate news about majors
    "oil well", "gas well", "upstream sector", "production rates",
    "field development", "unconventional resources", "drilling rig",
    "oil and gas leases", "petroleum exploration", "hydrocarbon exploration"
]

MIDSTREAM_KW = [
    "pipeline", "transport", "lng", "storage", "terminal", "distribution",
    "gasification", "shipping", "infrastructure", "transmission", "transportation", # Corrected typo
    "midstream", "compressor station", "pump station", "tanker", "carrier", "regasification",
    "gathering", "export terminal", "import terminal", "hub", "network", "capacity",
    "processing plant", "gas processing", "oil transport", "gas transport",
    "gas storage", "crude storage", "logistics", "distribution network",
    "transmission lines", "export facilities", "import facilities",
    "bunker fuel", "hydrogen pipeline", "hydrogen storage", "ammonia transport",
    "hydrogen carrier", "ccus" # Carbon Capture Utilization Storage (often transport/storage)
]

# Corrected typo: removed missing comma after 'pricing'
DOWNSTREAM_KW = [
    "refinery", "petrol", "diesel", "retail", "marketing", "fuel", "processing",
    "lubricants", "petrochemicals", "sales", "consumption", "price", "pricing",
    "downstream", "gasoline", "jet fuel", "kerosene", "asphalt", "bitumen",
    "nylon", "plastics", "chemical plant", "fuel station", "filling station",
    "demand", "margins", "end user", "power generation", "power plant",
    "industrial feedstock", "buy", "market",
    "hydrogen fuel cell", "hydrogen vehicles", "green hydrogen application",
    "industrial hydrogen use", "decarbonization", "energy efficiency", "heating",
    "cooking", "powering homes", "powering cars", "distillates", "refined products",
    "fuel oil", "gas oil", "naphtha", "propane", "butane", "resins", "polymers",
    "fertilizers", "specialty chemicals", "fuel economy", "retail prices",
    "wholesale prices", "customer", "supply chain management" # Broader context for downstream logistics
]

# Combined keywords for initial relevance check
ENERGY_CORE_KEYWORDS = list(set(
    UPSTREAM_KW + MIDSTREAM_KW + DOWNSTREAM_KW + ["oil", "gas", "petroleum", "energy sector",
    "fossil fuel", "natural gas", "oilpatch", "energy market", "oil prices", "gas prices",
    "barrel", "exploration & production", "petrochemical plant", "oil rig", "gas field",
    "oilfield", "energy transition", "carbon capture", "hydrogen", "lng market",
    "oil market", "gas market", "energy market"] # Adding some general ones here that might not be in U/M/D
))
# Ensure no duplicates and sort for consistency if needed, but not critical for `any()` check


IRRELEVANT_KEYWORDS = [
    "ad", "sponsored", "buy now", "shop now", "discount",
    "coupon", "offer", "best deals", "affiliate",
    "cooking", "pan", "stove", "grill", "bbq",
    "shopping", "supermarket", "fashion", "food", "recipe",
    "restaurant", "clothing", "returns", "apple", "customers",
    "tourism", "travel", "airline", "hotel",
    "entertainment", "movie", "music", "sports", "gaming",
    "health", "medical", "pharmaceutical", "vaccine",
    "education", "university", "school", "student", "teacher", "election",
    "crime", "police", "court", "justice", "space", "astronomy",
    "weather", "natural disaster", "storm", "flood", "earthquake",
    "real estate", "housing", "mortgage", "software", "app", "startup",
    "device", "chip", "cybersecurity", "automotive", "vehicles",
    "electric vehicles", "ev", "battery", "agriculture", "metals",
    "minerals", "finance", "banking", "loans", "company earnings", # Added more financial terms that are not sector-specific
    "quarterly results", "stock", "shares", "investment", "bonds"
]

JUNK_DOMAINS = [
    "amazon.com", "ebay.com", "walmart.com", "etsy.com", "aliexpress.com",
    "bestbuy.com", "target.com", "indiamart.com", "alibaba.com", "shopee.com",
    "flipkart.com", "facebook.com", "instagram.com", "linkedin.com",
    "twitter.com", "reddit.com", "pinterest.com", "wikipedia.org",
    "investopedia.com", "britannica.com", "youtube.com" # Directly block youtube.com
]

# Initialize session state variables - crucial for Streamlit's state management
if 'fetch_trigger' not in st.session_state:
    st.session_state.fetch_trigger = 0
if 'displayed_article_count' not in st.session_state:
    st.session_state.displayed_article_count = 10  # Initial articles to display
if 'all_fetched_articles' not in st.session_state:
    st.session_state.all_fetched_articles = []  # Stores all unique articles fetched across runs
if 'last_region_filter' not in st.session_state: # Initialize all filter states
    st.session_state.last_region_filter = "All"
if 'last_stream_filter' not in st.session_state: # Initialize all filter states
    st.session_state.last_stream_filter = "All"

# ----------------------------
# HELPER FUNCTIONS (for data processing)
# ----------------------------

def strip_html_tags(text):
    """Strips HTML tags and resolves entities from a string using BeautifulSoup."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    # Use HTML entities to text for better representation of characters like &amp;
    return soup.get_text(separator=' ', strip=True)

def clean_summary_for_markdown(text):
    """Escapes markdown-sensitive characters for Streamlit display."""
    # Ensure text is string and handle None
    if not isinstance(text, str):
        return ""
    # Common markdown special characters that need escaping
    # Only escape if they are not part of a markdown structure (e.g., in links)
    # For simplicity, escape all problematic ones.
    text = text.replace('\\', '\\\\') # Escape backslashes first
    text = text.replace('*', '\\*')
    text = text.replace('_', '\\_')
    text = text.replace('`', '\\`')
    text = text.replace('#', '\\#')
    text = text.replace('$', '\\$')
    text = text.replace('%', '\\%')
    text = text.replace('^', '\\^')
    text = text.replace('&', '\\&')
    text = text.replace('[', '\\[')
    text = text.replace(']', '\\]')
    text = text.replace('(', '\\(')
    text = text.replace(')', '\\)')
    text = text.replace('<', '\\<')
    text = text.replace('>', '\\>')
    text = text.replace('|', '\\|')
    text = text.replace('~', '\\~')
    text = text.replace('@', '\\@')
    text = text.replace('₹', '\\₹') # Indian Rupee symbol
    text = text.replace('£', '\\£') # Pound symbol
    text = text.replace('€', '\\€') # Euro symbol

    # Specific replacements for common patterns that might break markdown or look bad
    text = re.sub(r'\s{2,}', ' ', text).strip() # Replace multiple spaces with single space
    return text


def is_relevant_and_not_ad(title, description, url, source_name):
    """Determines if an article is relevant and not an advertisement."""
    title_lower = title.lower()
    desc_lower = strip_html_tags(description).lower()
    source_name_lower = source_name.lower()

    text_to_check_overall = f"{title_lower} {desc_lower} {source_name_lower}"

    # Check for core energy keywords in either title or description
    has_energy_keyword = any(keyword in title_lower for keyword in ENERGY_CORE_KEYWORDS) or \
                         any(keyword in desc_lower for keyword in ENERGY_CORE_KEYWORDS)

    if not has_energy_keyword:
        return False

    # Check for irrelevant/ad keywords in combined text
    if any(keyword in text_to_check_overall for keyword in IRRELEVANT_KEYWORDS):
        return False

    # Check for junk domains
    try:
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        if any(junk_domain in domain for junk_domain in JUNK_DOMAINS):
            return False
    except Exception as e:
        logger.warning(f"Error parsing URL {url} for junk domain check: {e}")
        return False # Treat as irrelevant if URL parsing fails

    return True

def classify_region(text):
    """Classifies the region of an article based on keywords."""
    text_lower = text.lower() # Renamed to avoid confusion with parameter 'text'
    if any(keyword in text_lower for keyword in LOCAL_KEYWORDS):
        return "Local"
    if any(keyword in text_lower for keyword in GLOBAL_KEYWORDS):
        return "Global"
    return "Unclassified Region"

def classify_stream(text):
    """Classifies the stream (Upstream, Midstream, Downstream) of an article based on keywords."""
    text_lower = text.lower() # Renamed to avoid confusion with parameter 'text'
    if any(keyword in text_lower for keyword in UPSTREAM_KW):
        return "Upstream"
    elif any(keyword in text_lower for keyword in MIDSTREAM_KW):
        return "Midstream"
    elif any(keyword in text_lower for keyword in DOWNSTREAM_KW):
        return "Downstream"
    return "Unclassified Stream"

def get_sort_key(article):
    """Helper function to get a sortable datetime object from an article."""
    try:
        if article["published_at"] == "N/A":
            return datetime.min # Push "N/A" dates to the very end
        return datetime.strptime(article["published_at"], "%b %d, %Y %H:%M %p UTC")
    except ValueError:
        logger.warning(f"Could not parse date for article: {article.get('title', 'No Title')}. Date string: {article.get('published_at', 'N/A')}. Sorting to end.")
        return datetime.min

# ----------------------------
# CORE ASYNC FETCHING & PROCESSING
# ----------------------------

async def _fetch_single_rss_feed(session, feed_url):
    """Fetches raw content of a single RSS feed."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/xml, text/xml, application/rss+xml, application/atom+xml'
        }
        async with session.get(feed_url, headers=headers, timeout=15) as response:
            response.raise_for_status() # Raises HTTPStatusError for bad responses (4xx/5xx)
            return await response.text()
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP Client Error fetching RSS feed {feed_url}: {e}")
        return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching RSS feed {feed_url}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching RSS feed {feed_url}: {e}")
        return None

async def _extract_main_article_content(session, url):
    """
    Asynchronously fetches and extracts the main article text from a given URL.
    Includes aggressive cleaning of non-article content.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        async with session.get(url, headers=headers, timeout=20) as response:
            response.raise_for_status()
            content = await response.text()

            soup = BeautifulSoup(content, 'html.parser')

            # Prioritized selectors for main article content
            selectors = [
                'article', 'main', 'div.entry-content', 'div.article-content',
                'div.td-post-content', 'div.post-content', 'div.gsc_oci_ext',
                'div[itemprop="articleBody"]', 'div[role="main"]', 'section.content',
                'div#article-body', 'div.story-body', 'div.body-content',
                'div.news-content', 'div.story-fulltext', 'div.article-text',
                'div[class*="content-wrapper"]', 'div[id*="article-"]',
                'div[class*="article-"]', 'div.l-article-content',
                'div.paywall-content', 'div[class*="body-copy"]', 'div.article-body',
            ]

            main_content_element = None
            for selector in selectors:
                main_content_element = soup.select_one(selector)
                if main_content_element:
                    break

            if main_content_element:
                # Aggressive removal of common non-article elements
                junk_selectors = [
                    'script', 'style', 'aside', 'nav', 'footer', 'header', 'form',
                    '.ad', '.ads', '[class*="advert"]', '[id*="ad-"]', '[class*="ad-"]',
                    '.newsletter', '.social-share', '.related-posts', '.comments',
                    '.share-bar', '.read-more', '.post-meta', '.author-info',
                    '.subscribe', '.sponsored', '.promo', '[id*="sticky"]',
                    'figure', 'img', 'video', 'iframe', 'svg', 'button', 'input',
                    '.skip-link', '.breadcrumbs', '.wp-block-group', '.wp-block-columns',
                    '.sidebar', '[class*="sidebar"]', '[id*="sidebar"]',
                    '.cookie-notice', '.gdpr-banner', '.modal', '.popup',
                    '.author-box', '.source-box', '.tags', '.category-links',
                    '.entry-utility', '.entry-meta', '.article-meta',
                    '.flex-video', '.caption', 'blockquote' # Blockquotes can be problematic too
                ]
                for tag_or_class in junk_selectors:
                    for tag in main_content_element.select(tag_or_class): # Use select for all matches
                        tag.decompose()

                # Extract text and clean whitespace
                text = main_content_element.get_text(separator='\n', strip=True)
                text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
                text = ' '.join(text.split())

                if len(text) < MIN_SCRAPED_CONTENT_LENGTH:
                    logger.debug(f"Scraped content too short ({len(text)} chars) for URL: {url}")
                    return "" # Return empty if too short

                return text
            return "" # No main content element found

    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP Client Error extracting content from {url}: {e}")
        return ""
    except asyncio.TimeoutError:
        logger.error(f"Timeout extracting content from {url}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error extracting content from {url}: {e}")
        return ""

async def _fetch_and_parse_all_rss(session, rss_feed_urls, status_placeholder, progress_bar_placeholder):
    """Fetches all RSS feeds and parses their entries."""
    status_placeholder.info(f"Initiating fetch from {len(rss_feed_urls)} RSS feeds...")
    logger.info(f"Initiating fetch from {len(rss_feed_urls)} RSS feeds.")

    rss_fetch_bar = progress_bar_placeholder.progress(0, text="Fetching RSS feeds...")
    feed_content_tasks = [_fetch_single_rss_feed(session, url) for url in rss_feed_urls]

    all_feed_raw_contents = await asyncio.gather(*feed_content_tasks)
    rss_fetch_bar.empty()
    status_placeholder.empty()
    logger.info("Finished fetching raw RSS feed contents.")

    articles_to_process = []
    seen_articles_hash = set()

    status_placeholder.write("Parsing RSS feed entries and applying initial filters...")
    parsing_and_filtering_bar = progress_bar_placeholder.progress(0, text="Parsing and filtering entries...")

    total_feed_entries_estimate = sum(
        len(feedparser.parse(content).entries)
        for content in all_feed_raw_contents if content and _is_valid_feed_content(content)
    ) or 1 # Avoid division by zero

    current_entry_count = 0
    for i, raw_content in enumerate(all_feed_raw_contents):
        feed_url = rss_feed_urls[i]
        if raw_content is None:
            logger.warning(f"Skipping empty or failed raw content for URL: {feed_url}")
            continue

        feed = feedparser.parse(raw_content) # feedparser handles errors gracefully internally
        source_name = feed.feed.title if hasattr(feed.feed, 'title') and feed.feed.title else feed_url

        if not feed.entries:
            logger.info(f"No entries found in feed: {feed_url}")
            continue

        for entry in feed.entries:
            title = getattr(entry, 'title', "").strip()
            url = getattr(entry, 'link', "").strip()
            desc = strip_html_tags(getattr(entry, 'summary', "")).strip()

            published_datetime = "N/A"
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    dt_obj = datetime(*entry.published_parsed[:6])
                    published_datetime = dt_obj.strftime("%b %d, %Y %H:%M %p UTC")
                except Exception as e:
                    logger.warning(f"Could not parse published_parsed for '{title}': {e}")

            if title and url and is_relevant_and_not_ad(title, desc, url, source_name):
                # Using a robust identifier for uniqueness
                article_identifier = (title.lower(), url.lower())
                if article_identifier not in seen_articles_hash:
                    seen_articles_hash.add(article_identifier)
                    articles_to_process.append({
                        "title": title,
                        "summary_rss": desc,
                        "url": url,
                        "published_at": published_datetime,
                        "source_name": source_name
                    })
            current_entry_count += 1
            if total_feed_entries_estimate > 0:
                parsing_and_filtering_bar.progress(
                    current_entry_count / total_feed_entries_estimate,
                    text=f"Parsing and filtering entries... {int(current_entry_count/total_feed_entries_estimate*100)}%"
                )

    parsing_and_filtering_bar.empty()
    status_placeholder.empty()
    logger.info(f"Identified {len(articles_to_process)} unique, relevant articles after initial parsing and filtering.")
    return articles_to_process

def _is_valid_feed_content(content):
    """Simple check to see if feedparser might parse content successfully."""
    return content and ("<rss" in content or "<feed" in content)

async def _scrape_and_prioritize_content(session, articles_data, status_placeholder, progress_bar_placeholder):
    """Determines content for summarization, scrapes full articles if RSS summary is insufficient."""
    contents_for_summarization = [None] * len(articles_data)
    scrape_needed_tasks = []
    articles_requiring_scrape_indices = []

    for i, article_data in enumerate(articles_data):
        rss_summary = article_data['summary_rss'].strip()

        if len(rss_summary) >= MIN_RSS_SUMMARY_LENGTH:
            contents_for_summarization[i] = rss_summary
        else:
            articles_requiring_scrape_indices.append(i)
            scrape_needed_tasks.append(
                asyncio.create_task(_extract_main_article_content(session, article_data['url']))
            )

    if scrape_needed_tasks:
        status_placeholder.write(f"Fetching {len(scrape_needed_tasks)} full article contents (where RSS was insufficient)...")
        article_fetch_bar = progress_bar_placeholder.progress(0, text="Fetching full articles...")

        # Await all scraping tasks concurrently
        scraped_results = await asyncio.gather(*scrape_needed_tasks)

        for i_task, scraped_text_raw in enumerate(scraped_results):
            original_idx = articles_requiring_scrape_indices[i_task]

            # Prioritize scraped content if it meets minimum length, else fallback to RSS summary or title
            if len(scraped_text_raw.strip()) >= MIN_SCRAPED_CONTENT_LENGTH:
                contents_for_summarization[original_idx] = scraped_text_raw.strip()
            else:
                contents_for_summarization[original_idx] = articles_data[original_idx]['summary_rss'].strip()
                if not contents_for_summarization[original_idx]:
                    contents_for_summarization[original_idx] = articles_data[original_idx]['title'].strip()
                    if not contents_for_summarization[original_idx]:
                         contents_for_summarization[original_idx] = "Content too short or unavailable for detailed summarization."
            article_fetch_bar.progress(
                (i_task + 1) / len(scrape_needed_tasks),
                text=f"Fetching full articles... {int((i_task+1)/len(scrape_needed_tasks)*100)}%"
            )
        article_fetch_bar.empty()
        status_placeholder.empty()
        logger.info(f"Finished scraping {len(scraped_results)} full articles.")

    return contents_for_summarization

async def _summarize_and_classify_articles(articles_data, contents_for_summarization, status_placeholder, progress_bar_placeholder):
    """Generates summaries and classifies articles."""
    status_placeholder.write("Generating summaries and classifying articles...")
    summarization_progress_bar = progress_bar_placeholder.progress(0, text="Generating summaries...")

    final_processed_articles = []
    for i, article_data in enumerate(articles_data):
        final_content_for_this_article = contents_for_summarization[i]

        final_summary_text = ""
        if final_content_for_this_article and final_content_for_this_article != "Content too short or unavailable for detailed summarization.":
            try:
                parser = PlaintextParser.from_string(final_content_for_this_article, EXT_TOKENIZER)
                sentences = [str(s) for s in EXT_SUMMARIZER(parser.document, sentences_count=3)]
                final_summary_text = " ".join(sentences)

                # Fallback if summarizer returns very little or nothing
                if not final_summary_text or len(final_summary_text.strip()) < 50:
                    final_summary_text = final_content_for_this_article[:500] + "..." if len(final_content_for_this_article) > 500 else final_content_for_this_article
            except Exception as e:
                logger.error(f"Error during summarization for '{article_data['title']}': {e}")
                final_summary_text = final_content_for_this_article[:500] + "..." if len(final_content_for_this_article) > 500 else final_content_for_this_article
        else:
            final_summary_text = final_content_for_this_article # Use placeholder message directly if content was unavailable

        # Clean summary for markdown display in Streamlit
        final_summary_text = clean_summary_for_markdown(final_summary_text)

        # Combine title and content for more robust classification
        text_for_classification = f"{article_data['title']} {final_content_for_this_article}"
        region = classify_region(text_for_classification)
        stream = classify_stream(text_for_classification)

        final_processed_articles.append({
            "title": article_data["title"],
            "summary": final_summary_text,
            "region": region,
            "stream": stream,
            "url": article_data["url"],
            "published_at": article_data["published_at"]
        })
        summarization_progress_bar.progress(
            (i + 1) / len(articles_data),
            text=f"Generating summaries and classifying... {int((i+1)/len(articles_data)*100)}%"
        )

    summarization_progress_bar.empty()
    status_placeholder.empty()
    logger.info(f"Finished generating summaries and classifications for {len(final_processed_articles)} articles.")
    return final_processed_articles

async def fetch_and_process_news(rss_feed_urls, status_placeholder, progress_bar_placeholder):
    """
    Orchestrates the entire news fetching, parsing, scraping, summarization,
    and classification pipeline.
    """
    async with aiohttp.ClientSession() as session:
        # Step 1: Fetch and parse all RSS feeds
        articles_to_process = await _fetch_and_parse_all_rss(
            session, rss_feed_urls, status_placeholder, progress_bar_placeholder
        )

        if not articles_to_process:
            status_placeholder.info("No relevant articles found after initial RSS parsing and filtering.")
            progress_bar_placeholder.empty()
            return []

        # Limit processing to conserve resources if too many articles
        if len(articles_to_process) > MAX_ARTICLES_TO_PROCESS:
            status_placeholder.warning(f"Limiting processing to first {MAX_ARTICLES_TO_PROCESS} articles to conserve resources.")
            logger.warning(f"Limiting processing to first {MAX_ARTICLES_TO_PROCESS} articles.")
            articles_to_process = articles_to_process[:MAX_ARTICLES_TO_PROCESS]

        # Step 2: Prioritize content for summarization (RSS summary or scraped full text)
        contents_for_summarization = await _scrape_and_prioritize_content(
            session, articles_to_process, status_placeholder, progress_bar_placeholder
        )

        # Step 3: Summarize and classify
        final_processed_articles = await _summarize_and_classify_articles(
            articles_to_process, contents_for_summarization, status_placeholder, progress_bar_placeholder
        )

    gc.collect() # Trigger garbage collection to free up memory
    return final_processed_articles

# ----------------------------
# CACHED NEWS DATA FETCH
# ----------------------------

@st.cache_data(ttl=timedelta(hours=TTL_HOURS), show_spinner=False)
def get_processed_news_cached():
    """
    Cached function to fetch and process news.
    Streamlit will re-run this only if cache is cleared or TTL expires.
    """
    logger.info(f"Attempting to fetch and process news (Cache TTL: {TTL_HOURS} hours)...")

    # Placeholders for status messages and progress bars during the fetch process
    status_placeholder = st.empty()
    progress_bar_placeholder = st.empty()

    articles = asyncio.run(fetch_and_process_news(RSS_FEED_URLS, status_placeholder, progress_bar_placeholder))

    # Ensure placeholders are empty once processing is complete
    status_placeholder.empty()
    progress_bar_placeholder.empty()

    logger.info(f"Finished processing news. Found {len(articles)} final articles to cache.")
    return articles

# ----------------------------
# STREAMLIT UI
# ----------------------------

def display_articles(articles):
    """Displays filtered and paginated articles."""
    if not articles:
        st.warning("No articles match your filter criteria or could be fetched. This could be due to:")
        st.warning("- **Website blocking:** Many news websites actively block automated access to their RSS feeds (e.g., `403 Forbidden` errors).")
        st.warning("- **RSS feed issues:** The feeds may be malformed, outdated, or contain no recent articles.")
        st.warning("- **Strict filters:** Your relevance and reputable source filters might be too strict for the available content.")
        st.warning("You may try adjusting your filters or refreshing the news.")
        return

    st.success(f"Showing {len(articles)} unique article(s) matching your filters.")
    for article in articles:
        with st.expander(article["title"]):
            st.markdown(f"**Published**: `{article['published_at']}`     **Stream**: `{article['stream']}`     **Region**: `{article['region']}`")
            st.markdown(f"🔗 [Read Full Article]({article['url']})", unsafe_allow_html=True)
            st.info(article["summary"]) # Summary is already cleaned by clean_summary_for_markdown

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Energy News Dashboard", layout="wide")
    st.title("🛢️ Fuelling Insights")
    st.subheader("Real-Time Oil & Gas News — Filtered, Classified, Summarised")

    st.sidebar.header("🔍 Filters")
    region_filter = st.sidebar.selectbox("Region", ["All", "Local", "Global"])
    stream_filter = st.sidebar.selectbox("Stream", ["All", "Upstream", "Midstream", "Downstream"])

    # Reset displayed article count if filters change
    # Using st.session_state directly for filter values to detect changes
    if st.session_state.get('current_region_filter', 'All') != region_filter or \
       st.session_state.get('current_stream_filter', 'All') != stream_filter:
        st.session_state.displayed_article_count = 10
        st.session_state.current_region_filter = region_filter
        st.session_state.current_stream_filter = stream_filter
        # st.rerun() # Rerunning here might cause flicker, let the main loop re-evaluate

    if st.sidebar.button("Refresh News", key="refresh_button"):
        st.cache_data.clear()  # Clear the cache to force a re-fetch
        st.session_state.displayed_article_count = 10
        st.rerun()

    last_updated_time = datetime.now().strftime('%b %d, %Y %H:%M:%S')
    st.sidebar.caption(f"🕒 Last data processed at: {last_updated_time}")

    # Fetch articles (cached)
    all_fetched_articles = get_processed_news_cached()
    st.session_state.all_fetched_articles = all_fetched_articles

    # Apply UI filters
    filtered_articles = [
        item for item in st.session_state.all_fetched_articles
        if (region_filter == "All" or item["region"] == region_filter) and
           (stream_filter == "All" or item["stream"] == stream_filter)
    ]

    st.info(f"Articles matching filters: {len(filtered_articles)}")

    # Sort the filtered articles by date
    filtered_articles.sort(key=get_sort_key, reverse=True)

    # Determine articles to display based on session state for pagination
    articles_to_display = filtered_articles[:st.session_state.displayed_article_count]

    display_articles(articles_to_display)

    remaining_articles_count = len(filtered_articles) - st.session_state.displayed_article_count

    if remaining_articles_count > 0:
        load_count = min(10, remaining_articles_count) # Load 10 or remaining, whichever is smaller
        if st.button(f"Load More ({load_count} articles)", key="load_more_button"):
            st.session_state.displayed_article_count += load_count
            st.rerun()
    elif len(filtered_articles) > 0: # Only show this if there were articles initially
        st.info("No more articles to load from the RSS feeds. (RSS feeds typically provide only the most recent articles and do not offer historical pagination.)")


if __name__ == "__main__":
    main()