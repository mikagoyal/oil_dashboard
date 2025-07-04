import streamlit as st
import asyncio
import aiohttp
from datetime import datetime, timedelta
import urllib.parse
import feedparser
from bs4 import BeautifulSoup
import gc
import re # For regex-based cleaning

import nltk
nltk.data.path.append("./nltk_data")

# --- Extractive Summarization Imports ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- Firebase Admin SDK Imports ---
# Make sure to install: pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore, exceptions, auth # Import auth service

# --- Configure Logging ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Constants ---
TTL_HOURS = 24  # Cache TTL for fetched news data in hours
MIN_RSS_SUMMARY_LENGTH = 50  # Minimum length for RSS summary to be considered sufficient
MIN_SCRAPED_CONTENT_LENGTH = 250 # Minimum length for scraped content to be considered meaningful
MAX_ARTICLES_TO_PROCESS = 150 # Max articles to process to conserve resources
DESIRED_SUMMARY_SENTENCE_COUNT = 3 

# --- SVG Icon Definitions ---
light_icon_svg = """
<svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="#a100ff"><path d="M160-120v-660q0-24 18-42t42-18h269q24 0 42 18t18 42v288h65q20.63 0 35.31 14.69Q664-462.63 664-442v219q0 21.68 15.5 36.34Q695-172 717-172t37.5-14.66Q770-201.32 770-223v-295q-11 6-23 9t-24 3q-39.48 0-66.74-27.26Q629-560.52 629-600q0-31.61 18-56.81Q665-682 695-690l-95-95 36-35 153 153q14 14 22.5 30.5T820-600v377q0 43.26-29.82 73.13-29.81 29.87-73 29.87Q674-120 644-149.87q-30-29.87-30-73.13v-219h-65v322H160Zm60-432h269v-228H220v228Zm503-4q18 0 31-13t13-31q0-18-13-31t-31-13q-18 0-31 13t-13 31q0 18 13 31t31 13ZM220-180h269v-312H220v312Zm269 0H220h269Z"/></svg>
"""

dark_icon_svg = """
<svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="#ffffff"><path d="M160-120v-660q0-24 18-42t42-18h269q24 0 42 18t18 42v288h65q20.63 0 35.31 14.69Q664-462.63 664-442v219q0 21.68 15.5 36.34Q695-172 717-172t37.5-14.66Q770-201.32 770-223v-295q-11 6-23 9t-24 3q-39.48 0-66.74-27.26Q629-560.52 629-600q0-31.61 18-56.81Q665-682 695-690l-95-95 36-35 153 153q14 14 22.5 30.5T820-600v377q0 43.26-29.82 73.13-29.81 29.87-73 29.87Q674-120 644-149.87q-30-29.87-30-73.13v-219h-65v322H160Zm60-432h269v-228H220v228Zm503-4q18 0 31-13t13-31q0-18-13-31t-31-13q-18 0-31 13t-13 31q0 18 13 31t31 13ZM220-180h269v-312H220v312Zm269 0H220h269Z"/></svg>
"""

filters_dark = """
<svg xmlns="http://www.w3.org/2000/svg" height="30px" viewBox="0 -960 960 960" width="30px" fill="#ffffff"><path d="M80-200v-66.67h400V-200H80Zm0-200v-66.67h600V-400H80Zm0-200v-66.67h800V-600H80Z"/></svg>
"""
filters_light = """
<svg xmlns="http://www.w3.org/2000/svg" height="30px" viewBox="0 -960 960 960" width="30px" fill="#a100ff"><path d="M80-200v-66.67h400V-200H80Zm0-200v-66.67h600V-400H80Zm0-200v-66.67h800V-600H80Z"/></svg>
"""

# --- Initialize Firebase Admin SDK and Firestore ---
@st.cache_resource
def initialize_firebase_services():
    """
    Initializes Firebase Admin SDK and Firestore client.
    Does not handle user authentication directly, but provides the DB client.
    """
    try:
        cred_dict = {
            "type": st.secrets["firestore_credentials"]["type"],
            "project_id": st.secrets["firestore_credentials"]["project_id"],
            "private_key_id": st.secrets["firestore_credentials"]["private_key_id"],
            "private_key": st.secrets["firestore_credentials"]["private_key"],
            "client_email": st.secrets["firestore_credentials"]["client_email"],
            "client_id": st.secrets["firestore_credentials"]["client_id"],
            "auth_uri": st.secrets["firestore_credentials"]["auth_uri"],
            "token_uri": st.secrets["firestore_credentials"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firestore_credentials"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firestore_credentials"]["client_x509_cert_url"],
            "universe_domain": st.secrets["firestore_credentials"]["universe_domain"],
        }
        
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase app initialized successfully.")
        else:
            logger.info("Firebase app already initialized.")
        
        firestore_client = firestore.client()
        return firestore_client

    except Exception as e:
        st.error(f"Error initializing Firebase/Firestore. Please check your .streamlit/secrets.toml file and Google Cloud project setup: {e}")
        logger.error(f"Firebase initialization failed: {e}")
        return None

db = initialize_firebase_services()

# --- Updated CUSTOM CSS ---
st.markdown(
    """
    <style>
    /* Hide the sidebar collapse button */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Hide the logo spacer */
    [data-testid="stLogoSpacer"] {
        display: none !important;
    }

    /* Adjust sidebar header padding */
    [data-testid="stSidebarHeader"] {
        padding-top: 1.5rem !important;
        padding-bottom: 0px !important;
        height: auto !important;
        min-height: auto !important;
        margin: 0px !important;
    }

    /* Reset padding/margin for sidebar header children */
    [data-testid="stSidebarHeader"] > div {
        margin: 0px !important;
        padding: 0px !important;
        min-height: auto !important;
        height: auto !important;
    }

    /* --- Fix for Sidebar Input Width and Eye Icon Positioning (Revised) --- */

/* Ensure the main sidebar section allows its children to take full width */
section[data-testid="stSidebar"] > div {
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
}

/* Target the outermost container of the text input (e.g., for email and password) in the sidebar */
section[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] {
    width: 100% !important;
    max-width: none !important; /* Crucial: Overrides any fixed max-width */
    box-sizing: border-box !important;
    margin: 0 !important; /* Remove any default margins */
    padding: 0 !important; /* Remove any default padding */
}

/* Target the inner container that holds both the input field and the eye button */
/* This is critical for controlling the layout of the input and its icon */
section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
    display: flex !important; /* Make it a flex container */
    flex-direction: row !important; /* Arrange items in a row */
    align-items: center !important; /* Vertically align items */
    justify-content: space-between !important; /* Distribute space between input and button */
    padding: 0 !important; /* Remove all padding from this container */
    margin: 0 !important; /* Remove all margin from this container */
    overflow: hidden !important; /* Hide any overflow if elements are slightly larger */
}

/* Target the actual input field (for both email and password) */
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="password"] {
    flex-grow: 1 !important; /* Allow the input field to grow and take all available space */
    width: auto !important; /* Allow width to be determined by flex-grow */
    max-width: none !important; /* Ensure no max-width limits it */
    box-sizing: border-box !important;
    padding-right: 0.5rem !important; /* Keep a small padding between text and icon */
    min-width: 0 !important; /* Allow it to shrink if needed, but flex-grow dominates */
    margin: 0 !important; /* Remove any margins */
}

/* Target the password visibility toggle button specifically */
section[data-testid="stSidebar"] button[aria-label="Show password text"],
section[data-testid="stSidebar"] button[aria-label="Hide password text"] { /* Add rule for "Hide" state */
    flex-shrink: 0 !important; /* Prevent the button from shrinking */
    padding: 0.2rem !important; /* Adjust padding around the icon itself */
    margin-left: 0 !important; /* Remove any left margin */
    margin-right: 0 !important; /* Ensure no extra space on its right */
    width: auto !important; /* Allow button to take only necessary width */
    min-width: unset !important; /* Remove any minimum width */
    height: auto !important; /* Adjust height based on content */
    background-color: transparent !important; /* Make button background transparent */
    border: none !important; /* Remove button border */
    color: var(--text-color) !important; /* Ensure icon color is correct */
}

/* Target the SVG icon inside the button if needed, ensure it's positioned correctly */
section[data-testid="stSidebar"] button[aria-label="Show password text"] svg,
section[data-testid="stSidebar"] button[aria-label="Hide password text"] svg {
    display: block !important; /* Ensure SVG behaves as a block */
    width: 20px !important; /* Set explicit width if needed */
    height: 20px !important; /* Set explicit height if needed */
    vertical-align: middle !important; /* Align vertically */
}


/* --- Fix for Login/Sign Up Buttons --- */

/* Target the columns container for login/signup buttons */
/* The key is to make the columns themselves flex items and manage their spacing */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    gap: 0.5rem !important; /* Add a small gap between buttons */
    justify-content: space-between !important; /* Distribute space evenly */
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
}

/* Target the individual columns holding the buttons */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 !important; /* Make each column take equal space */
    min-width: 0 !important; /* Allow columns to shrink */
    padding: 0 !important; /* Remove column padding */
    margin: 0 !important; /* Remove column margin */
    display: flex !important; /* Make column a flex container */
    justify-content: center !important; /* Center button within its column */
    align-items: center !important; /* Vertically center button */
}

/* Target the actual buttons within these columns */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"] button {
    width: 100% !important; /* Make buttons fill their column */
    max-width: none !important;
    box-sizing: border-box !important;
    /* Re-apply your desired button styling here if it was lost */
    background-color: var(--secondary-background-color) !important;
    border: 1px solid #CCC !important;
    color: var(--text-color) !important;
    padding: 0.5rem 1rem !important;
    border-radius: 4px !important;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[data-testid="column"] button:hover {
    background-color: var(--primary-color) !important;
    color: #FFFFFF !important;
    border-color: var(--primary-color) !important;
}

    /* Main content margin */
    .main {
        margin-left: 21rem;
    }

    /* Bookmark button styling with increased specificity */
    /* Target buttons whose data-testid starts with "stButton-bookmark_btn" */
    div[data-testid^="stButton-bookmark_btn"] > button {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--primary-color) !important; /* Ensure purple border */
        color: var(--text-color) !important;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 120px;
    }
    div[data-testid^="stButton-bookmark_btn"] > button:hover {
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border-color: var(--primary-color) !important;
    }

    /* Tab styling to remove boxes */
    div[data-testid="stTabs"] button {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        color: var(--text-color) !important;
        font-weight: normal !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--secondary-background-color) !important;
        border-bottom: 2px solid var(--primary-color) !important;
    }
    div[data-testid="stTabs"] button:hover {
        background-color: var(--secondary-background-color) !important;
    }

    /* Checkbox styling for better visibility */
    .stCheckbox > label {
        color: var(--text-color) !important;
        margin-bottom: 0.3rem;
    }

    /* Search input styling */
    .stTextInput > div > div > input {
        width: 100 !important;
        padding: 0.5rem 2rem 0.5rem 0.5rem !important; /* Extra padding on right for clear button */
        box-sizing: border-box !important;
        border: 1px solid var(--primary-color) !important;
        border-radius: 4px !important;
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important; /* Changed to text-color for better visibility */
    }

    /* Search input placeholder styling for dark mode */
    .stTextInput > div > div > input::placeholder {
        color: var(--primary-color) !important; /* Changed to primary-color for consistency */
        opacity: 1 !important; /* Ensure full opacity */
    }

    /* Clear button styling (inside search) */
    .stButton > button[data-testid="stButton-clear_search"] { /* Target clear button specifically */
        width: 100% !important;
        padding: 0.4rem !important;
        box-sizing: border-box !important;
        background-color: transparent !important;
        border: none !important; /* No border for clear button to blend with input */
        color: var(--text-color) !important;
        font-size: 1.2rem !important;
        line-height: 1 !important;
        text-align: center !important;
    }
    .stButton > button[data-testid="stButton-clear_search"]:hover {
        color: var(--primary-color) !important;
        background-color: transparent !important;
    }

    /* General button styling (for other buttons like Load More, Search Articles) */
    /* Exclude bookmark buttons and clear search button from this general styling */
    [data-testid="stButton"] > button:not([data-testid^="stButton-bookmark_btn"]):not([data-testid="stButton-clear_search"]) {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid #CCC !important; /* Default border */
        color: var(--text-color) !important;
        padding: 0.5rem 1rem !important;
        border-radius: 4px !important;
        cursor: pointer;
    }
    [data-testid="stButton"] > button:not([data-testid^="stButton-bookmark_btn"]):not([data-testid="stButton-clear_search"]):hover {
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border-color: var(--primary-color) !important;
    }

    /* Sources button styling */
button[data-testid="stButton-sources_button"] {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid var(--primary-color) !important;
    color: var(--text-color) !important;
    padding: 0.5rem 1rem !important;
    border-radius: 4px !important;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
button[data-testid="stButton-sources_button"]:hover {
    background-color: var(--primary-color) !important;
    color: #FFFFFF !important;
    border-color: var(--primary-color) !important;
}

    /* Ensure container aligns items */
    [data-testid="column"] {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Adjust container spacing */
    .st-emotion-cache-1jic1uk {
        margin-bottom: 0.5rem !important;
    }

    /* Increase specificity for Source button */
div[data-testid="stButton-sources_button"] > button {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid var(--primary-color) !important;
    color: var(--text-color) !important;
    padding: 0.5rem 1rem !important;
    border-radius: 4px !important;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
div[data-testid="stButton-sources_button"] > button:hover {
    background-color: var(--primary-color) !important;
    color: #FFFFFF !important;
    border-color: var(--primary-color) !important;
}

/* Ensure modal fits within the layout */
div[data-testid="modalContainer-sources_modal"] {
    z-index: 1000 !important;
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--primary-color) !important;
    border-radius: 4px !important;
}
    </style>
    """,
    unsafe_allow_html=True
)


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
    "https://oilprice.com/rss/main",
    "https://www.oedigital.com/energy/oil?format=feed&limitstart=&type=rss",
    "https://oilandgas-investments.com/feed/",
    "https://www.industrialinfo.com/rss/news/mostRecent.jsp?cat=INDUSTRY;REFINING",
    "https://www.oilfutures.co.uk/feeds/posts/default",

]


# --- Keyword Definitions (All Lowercase for Consistency) ---

# REGIONAL KEYWORDS - NEW, MORE GRANULAR CLASSIFICATION
INDIA_KEYWORDS = [
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

APAC_KEYWORDS = [
    "china", "japan", "korea", "australia", "indonesia", "malaysia", "singapore",
    "thailand", "vietnam", "philippines", "new zealand", "papua new guinea",
    "timor-leste", "brunei", "myanmar", "cambodia", "laos", "mongolia", "sri lanka",
    "bangladesh", "pakistan", "fiji", "pacific islands", "asean", "south asia",
    "east asia", "southeast asia", "oceania", "sinopec", "petronas", "pertamina", "pttep",
    "asia pacific", "asia", "koreas", "apec", "soco", "cnooc", "japex"
]

MIDDLE_EAST_KEYWORDS = [
    "saudi", "uae", "qatar", "iran", "iraq", "kuwait", "oman", "bahrain", "yemen",
    "syria", "lebanon", "israel", "jordan", "persian gulf", "middle east",
    "aramco", "adnco", "qp", "knpc", "opec", "gcc", "mena", "adnoc", "israel"
]

EUROPE_KEYWORDS = [
    "europe", "uk", "united kingdom", "norway", "germany", "france", "italy",
    "netherlands", "russia", "eu", "european union", "north sea", "mediterranean",
    "equinor", "bp", "shell", "totalenergies", "eni", "gazprom", "poland", "spain",
    "ireland", "scotland", "denmark", "sweden", "finland", "baltic", "black sea",
    "caspian", "norwegian sea", "baku", "azerbaijan", "kazakhstan", "turkey",
    "ukraine", "eec", "eia europe", "european market"
]

AFRICA_KEYWORDS = [
    "africa", "nigeria", "angola", "algeria", "libya", "egypt", "south africa",
    "ghana", "mozambique", "tanzania", "uganda", "kenya", "equatorial guinea",
    "gabon", "congo", "senegal", "ivory coast", "morocco", "tunisia", "nnpc",
    "african", "sub-saharan", "north africa"
]

NORTH_AMERICA_KEYWORDS = [
    "usa", "canada", "mexico", "united states", "north america", "gulf of mexico",
    "alaska", "texas", "alberta", "pioneer natural resources", "exxon", "chevron",
    "conocophillips", "occidental petroleum", "america", "us", "u.s.", "american",
    "permian", "eagle ford", "bakken", "marcellus", "canadian", "pemex", "us market",
    "california", "north dakota", "houston", "devon energy", "haynesville",
    "pennsylvania", "gulf of america"
]

SOUTH_AMERICA_KEYWORDS = [
    "brazil", "venezuela", "colombia", "argentina", "ecuador", "guyana", "suriname",
    "bolivia", "chile", "peru", "petrobras", "ecopetrol", "pdvsa", "latin america",
    "south american"
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

DOWNSTREAM_KW = [
    "refinery", "petrol", "diesel", "retail", "marketing", "fuel", "processing",
    "lubricants", "petrochemicals", "sales", "consumption", "priceOW3Nprice", "pricing",
    "downstream", "gasoline", "jet fuel", "kerosene", "asphalt", "bitumen",
    "nylon", "plastics", "chemical plant", "fuel station", "filling station",
    "demand", "margins", "end user", "power generation", "power plant",
    "industrial feedstock", "buy", "market", "sale",
    "hydrogen fuel cell", "hydrogen vehicles", "green hydrogen application",
    "industrial hydrogen use", "decarbonization", "energy efficiency", "heating",
    "cooking", "powering homes", "powering cars", "distillates", "refined products",
    "fuel oil", "gas oil", "naphtha", "propane", "butane", "resins", "polymers",
    "fertilizers", "specialty chemicals", "fuel economy", "retail prices",
    "wholesale prices", "customer", "supply chain management" # Broader context for downstream logistics
]

# Combined keywords for initial relevance check
ENERGY_CORE_KEYWORDS = list(set(
    UPSTREAM_KW + MIDSTREAM_KW + DOWNSTREAM_KW + [
    "oil", "gas", "petroleum", "energy sector", "fossil fuel", "natural gas",
    "oilpatch", "energy market", "oil prices", "gas prices", "barrel",
    "exploration & production", "petrochemical plant", "oil rig", "gas field",
    "oilfield", "energy transition", "carbon capture", "hydrogen", "lng market",
    "oil market", "gas market", "energy market", "crude oil", "natural gas liquids",
    "nGL", "hydrocarbons", "petrochemical complex", "energy policy", "regulations"
]))

IRRELEVANT_KEYWORDS = [
    "ad", "sponsored", "buy now", "shop now", "discount",
    "coupon", "best deals", "burner",
    "cooking", "pan", "stove", "grill", "bbq",
    "shopping", "supermarket", "fashion", "food", "recipe",
    "restaurant", "clothing", "apple", "customers",
    "tourism", "travel", "airline", "hotel",
    "entertainment", "movie", "music", "sports", "gaming",
    "health", "medical", "pharmaceutical", "vaccine",
    "education", "university", "school", "student", "teacher", "election",
    "crime", "police", "court", "justice", "space", "astronomy",
    "weather", "natural disaster", "storm", "flood", "earthquake",
    "real estate", "housing", "mortgage", "software", "app", "startup",
    "device", "chip", "cybersecurity", "agriculture", "metals",
    "minerals", "finance", "banking", "loans", "company earnings",
    "quarterly results", "bonds", "blockchain", "football", "basketball", 
    "cricket", "olympics", "world cup", "championship",
    "celebrity", "gossip", "tv show", "film", "art", "museum", "gallery", "fashion week",
    "ecommerce", "online store", "gaming console",
    "health care", "doctor", "hospital", "therapy", "mental health", "fitness", "diet",
    "schooling", "curriculum", "campus", "phd", "master's",
    "verdict", "trial", "sentencing", "police report", "investigation",
    "galaxy", "universe", "planet", "telescope", "astronaut", "nasa", "spacex",
    "housing market", "property", "rent", "landlord", "tenant", "real estate agent",
    "data science", "cloud computing", "fintech", "biotech", "robotics", "gadgets", "apps",
    "supply chain disruption", "inflation", "interest rates", "central bank", "recession",
    "gdp", "market cap", "forex", "commodities trading", "hedge fund",
    "private equity", "venture capital", "mergers and acquisitions", "ipo", "dividends",
    "portfolio", "risk management", "financial planning", "credit", "audit",
    "audit report", "accounting", "balance sheet", "income statement", "cash flow", "earnings call"
]


JUNK_DOMAINS = [
    "amazon.com", "ebay.com", "walmart.com", "etsy.com", "aliexpress.com",
    "bestbuy.com", "target.com", "indiamart.com", "alibaba.com", "shopee.com",
    "flipkart.com", "myntra.com", "snapdeal.com", "jabong.com", "zomato.com",
    "swiggy.com", "uber.com", "ola.com", "airbnb.com", "booking.com", "expedia.com",
    "tripadvisor.com", "yelp.com", "glassdoor.com", "indeed.com", "naukri.com",
    "monster.com", "timesjobs.com", "upwork.com", "fiverr.com", "freelancer.com",
    "coursera.org", "edx.org", "udemy.com", "skillshare.com", "khanacademy.org",
    "youtube.com", "dailymotion.com", "vimeo.com", "tiktok.com", "pinterest.com",
    "facebook.com", "instagram.com", "linkedin.com", "twitter.com", "reddit.com",
    "wikipedia.org", "investopedia.com", "britannica.com", "dictionary.com",
    "thesaurus.com", "imdb.com", "rottentomatoes.com", "metacritic.com",
    "espn.com", "cricbuzz.com", "fandom.com", "ign.com", "gamespot.com",
    "steamcommunity.com", "github.com", "stackoverflow.com", "medium.com",
    "dev.to", "geeksforgeeks.org", "programiz.com", "w3schools.com", "tutorialspoint.com",
    "stackoverflow.com", "quora.com", "answers.com", "cnet.com", "zdnet.com",
    "techcrunch.com", "theverge.com", "engadget.com", "arstechnica.com",
    "macrumors.com", "gsmarena.com", "xda-developers.com", "androidcentral.com",
    "windowscentral.com", "lifehacker.com", "howtogeek.com", "makeuseof.com",
    "healthline.com", "webmd.com", "mayoclinic.org", "nih.gov", "cdc.gov",
    "who.int", "un.org", "gov.uk", "usa.gov", "canada.ca", "ec.europa.eu",
    "indiatimes.com", "timesofindia.indiatimes.com", "ndtv.com", "zeenews.india.com",
    "hindustantimes.com", "deccanherald.com", "thehindu.com", "moneycontrol.com",
    "businesstoday.in", "livemint.com", "economictimes.indiatimes.com",
    "cnbc.com", "bloomberg.com", "reuters.com", "apnews.com", "afp.com",
    "bbc.com", "cnn.com", "foxnews.com", "nytimes.com", "wsj.com",
    "theguardian.com", "washingtonpost.com", "ft.com", "economist.com",
    "forbes.com", "businessinsider.com", "techradar.com",
    "investing.com", "fxstreet.com", "dailyfx.com", "babypips.com",
    "google.com/search", "google.com/finance", "google.com/news", "news.google.com",
    "bing.com", "duckduckgo.com", "yahoo.com", "mail.yahoo.com", "gmail.com",
    "outlook.live.com", "protonmail.com", "icloud.com", "apple.com", "microsoft.com",
    "oracle.com", "ibm.com", "salesforce.com", "sap.com", "adobe.com",
    "cisco.com", "dell.com", "hp.com", "lenovo.com", "samsung.com", "lg.com",
    "sony.com", "nintendo.com", "playstation.com", "xbox.com",
    "netflix.com", "hulu.com", "disneyplus.com", "primevideo.com", "spotify.com",
    "applemusic.com", "pandora.com", "soundcloud.com", "bandcamp.com",
    "patreon.com", "kickstarter.com", "indiegogo.com", "gofundme.com",
    "change.org", "avaaz.org", "greenpeace.org", "wwf.org", "amnesty.org",
    "doctorswithoutborders.org", "redcross.org", "unicef.org", "unesco.org",
    "who.int", "wto.org", "imf.org", "worldbank.org", "federalreserve.gov",
    "ecb.europa.eu", "bankofengland.co.uk", "bankofjapan.or.jp",
    "youtube.com"
]

# Initialize session state variables
if 'fetch_trigger' not in st.session_state:
    st.session_state.fetch_trigger = 0
if 'displayed_article_count' not in st.session_state:
    st.session_state.displayed_article_count = 10
if 'all_fetched_articles' not in st.session_state:
    st.session_state.all_fetched_articles = []
if 'current_region_filters' not in st.session_state:
    st.session_state.current_region_filters = []
if 'current_stream_filters' not in st.session_state:
    st.session_state.current_stream_filters = []
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "dark"
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []
if 'show_search_bar' not in st.session_state:
    st.session_state.show_search_bar = False
if 'logged_in' not in st.session_state: # New: Track login status
    st.session_state.logged_in = False
if 'user_email' not in st.session_state: # New: Store user's email
    st.session_state.user_email = None


# ----------------------------
# AUTHENTICATION FUNCTIONS
# ----------------------------

def _signup_user(email, password):
    """Attempts to create a new user in Firebase Authentication."""
    if db is None:
        st.error("Database not initialized. Cannot sign up.")
        return False
    try:
        user = auth.create_user(email=email, password=password)
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.toast(f"Account created and logged in as {email}!", icon="✅")
        logger.info(f"User created: {user.uid} with email: {email}")
        st.session_state.bookmarks = load_bookmarks_from_firestore() # Load new user's bookmarks (should be empty)
        return True
    except exceptions.FirebaseError as e:
        if "EMAIL_ALREADY_EXISTS" in str(e):
            st.error("This email is already registered. Please try logging in instead.")
        else:
            st.error(f"Error signing up: {e}")
        logger.error(f"Signup error for {email}: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during signup: {e}")
        logger.error(f"Unexpected signup error for {email}: {e}")
        return False

def _login_user(email, password):
    """
    Simulates user login by checking if the email exists in Firebase Auth.
    WARNING: This does NOT securely verify the password in Streamlit's backend.
    It's for demonstration of data separation, not robust authentication.
    """
    if db is None:
        return "Database not initialized. Cannot log in."
    try:
        # Check if user exists by email
        user = auth.get_user_by_email(email)
        
        # In a real client-side app, you'd use Firebase client SDK to signInWithEmailAndPassword
        # Here, we are just checking existence. For this demo, we assume if email exists,
        # and a password is provided (even if not verified by Admin SDK), the user intends to log in.
        # A more secure approach would involve custom tokens or passwordless links.

        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.toast(f"Logged in as {email}!", icon="✅")
        logger.info(f"User logged in (simulated): {email}")
        st.session_state.bookmarks = load_bookmarks_from_firestore() # Load user's bookmarks
        return None  # No error
    except exceptions.FirebaseError as e:
        if "email not found" in str(e).lower():
            return "No account found with that email. Please sign up."
        else:
            return f"Error logging in: {e}"
    except Exception as e:
        return f"An unexpected error occurred during login: {e}"

def _logout_user():
    """Logs out the current user."""
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.bookmarks = [] # Clear bookmarks on logout
    st.toast("Logged out successfully!", icon="👋")
    logger.info("User logged out.")
    st.rerun() # Rerun to update UI


# --- HELPER FUNCTIONS (for data processing) ---

def strip_html_tags(text):
    """Strips HTML tags and resolves entities from a string using BeautifulSoup."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    # Use HTML entities to text for better representation of characters like &
    return soup.get_text(separator=' ', strip=True)

def clean_summary_for_markdown(text):
    """Escapes markdown-sensitive characters for Streamlit display."""
    # Ensure text is string and handle None
    if not isinstance(text, str):
        return ""
    # Common markdown special characters that need escaping
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
    text = text.replace('â¹', '\\â¹') # Indian Rupee symbol
    text = text.replace('Â£', '\\Â£') # Pound symbol
    text = text.replace('â¬', '\\â¬') # Euro symbol

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
    """Classifies the region of an article based on keywords with a specific hierarchy, prioritizing country names."""
    text_lower = text.lower()

    # Prioritize specific country names first
    country_keywords = {
        "norway": "Europe",
        "india": "India",
        "saudi": "Middle East",
        "uae": "Middle East",
        "qatar": "Middle East",
        "iran": "Middle East",
        "iraq": "Middle East",
        "kuwait": "Middle East",
        "oman": "Middle East",
        "usa": "North America",
        "u.s.": "North America",
        "us": "North America",
        "canada": "North America",
        "mexico": "North America",
        "brazil": "South America",
        "nigeria": "Africa",
        "angola": "Africa",
        "egypt": "Africa",
        "china": "APAC",
        "japan": "APAC",
        "australia": "APAC"
        # Add more country-to-region mappings as needed
    }
    
    # Check for country-specific keywords first
    for country, region in country_keywords.items():
        if country in text_lower:
            return region

    # Fallback to regional keyword checks if no country is matched
    if any(keyword in text_lower for keyword in INDIA_KEYWORDS):
        return "India"
    if any(keyword in text_lower for keyword in MIDDLE_EAST_KEYWORDS):
        return "Middle East"
    if any(keyword in text_lower for keyword in NORTH_AMERICA_KEYWORDS):
        return "North America"
    if any(keyword in text_lower for keyword in SOUTH_AMERICA_KEYWORDS):
        return "South America"
    if any(keyword in text_lower for keyword in EUROPE_KEYWORDS):
        return "Europe"
    if any(keyword in text_lower for keyword in AFRICA_KEYWORDS):
        return "Africa"
    if any(keyword in text_lower for keyword in APAC_KEYWORDS):
        return "APAC"
    
    # Fallback to Global if no specific region matched
    return "Unclassified Region"

def classify_stream(text):
    """Classifies the stream (Upstream, Midstream, Downstream) of an article based on keywords."""
    text_lower = text.lower()
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

# --- Firestore Bookmark Functions ---
def load_bookmarks_from_firestore():
    """Loads bookmarks from Firestore for the current user's email."""
    if db is None or not st.session_state.logged_in or st.session_state.user_email is None:
        logger.info("Not logged in or DB not initialized. Cannot load bookmarks.")
        return []
    try:
        # Use user-specific collection based on email
        bookmarks_ref = db.collection('users').document(st.session_state.user_email).collection('bookmarks')
        docs = bookmarks_ref.stream()
        bookmarks = []
        for doc in docs:
            bookmarks.append(doc.to_dict())
        logger.info(f"Loaded {len(bookmarks)} bookmarks from Firestore for user: {st.session_state.user_email}.")
        return bookmarks
    except exceptions.FirebaseError as e:
        logger.error(f"Error loading bookmarks from Firestore for user {st.session_state.user_email}: {e}")
        st.error(f"Could not load bookmarks from database: {e}. Check Firestore permissions for your user ID.")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading bookmarks for user {st.session_state.user_email}: {e}")
        st.error(f"An unexpected error occurred while loading bookmarks: {e}")
        return []

def add_bookmark_to_firestore(article):
    """Adds an article to Firestore bookmarks for the current user's email."""
    if db is None or not st.session_state.logged_in or st.session_state.user_email is None:
        st.error("Please log in to bookmark articles.")
        return
    try:
        # URL-encode the link to make it a valid Firestore document ID
        encoded_link = urllib.parse.quote_plus(article['link'])
        # Use user-specific collection based on email
        doc_ref = db.collection('users').document(st.session_state.user_email).collection('bookmarks').document(encoded_link)
        doc_ref.set(article)
        st.toast(f"Bookmarked: {article['title']}", icon="🔖")
        logger.info(f"Added bookmark to Firestore for user {st.session_state.user_email}: {article['title']} with ID: {encoded_link}")
        # Force a reload of bookmarks from Firestore to update UI
        st.session_state.bookmarks = load_bookmarks_from_firestore()
    except exceptions.FirebaseError as e:
        logger.error(f"Error adding bookmark to Firestore for user {st.session_state.user_email}: {e}")
        st.error(f"Could not add bookmark to database: {e}. Check Firestore permissions for your user ID.")
    except Exception as e:
        st.error(f"An unexpected error occurred while adding bookmark: {e}")
        logger.error(f"Unexpected error adding bookmark for user {st.session_state.user_email}: {e}")

def remove_bookmark_from_firestore(article_link):
    """Removes an article from Firestore bookmarks for the current user's email."""
    if db is None or not st.session_state.logged_in or st.session_state.user_email is None:
        st.error("Please log in to manage bookmarks.")
        return
    try:
        # URL-encode the link to match the document ID format
        encoded_link = urllib.parse.quote_plus(article_link)
        # Use user-specific collection based on email
        doc_ref = db.collection('users').document(st.session_state.user_email).collection('bookmarks').document(encoded_link)
        doc_ref.delete()
        st.toast("Bookmark removed.", icon="🗑️")
        logger.info(f"Removed bookmark from Firestore for user {st.session_state.user_email}: {article_link} with ID: {encoded_link}")
        # Force a reload of bookmarks from Firestore to update UI
        st.session_state.bookmarks = load_bookmarks_from_firestore()
    except exceptions.FirebaseError as e:
        logger.error(f"Error removing bookmark from Firestore for user {st.session_state.user_email}: {e}")
        st.error(f"Could not remove bookmark from database: {e}. Check Firestore permissions for your user ID.")
    except Exception as e:
        st.error(f"An unexpected error occurred while removing bookmark: {e}")
        logger.error(f"Unexpected error removing bookmark for user {st.session_state.user_email}: {e}")

# Initialize bookmarks in session state on startup by loading from Firestore
# This will only load if a user is already marked as logged in from a previous session state.
if 'bookmarks' not in st.session_state and st.session_state.logged_in:
    st.session_state.bookmarks = load_bookmarks_from_firestore()


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
                        "source_name": source_name,
                        "link": url  # Add link field for consistency with bookmarks
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
            contents_for_summarization[i] = None # Explicitly set to None for articles needing scrape
            scrape_needed_tasks.append(
                asyncio.create_task(_extract_main_article_content(session, article_data['url']))
            )
            articles_requiring_scrape_indices.append(i)

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
                sentences = [str(s) for s in EXT_SUMMARIZER(parser.document, sentences_count=DESIRED_SUMMARY_SENTENCE_COUNT)]
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
            "title": article_data["title"], # Corrected from 'article' to 'article_data'
            "summary": final_summary_text,
            "region": region,
            "stream": stream,
            "url": article_data["url"],
            "published_at": article_data["published_at"],
            "link": article_data["url"],  # Add link field for consistency with bookmarks
            "source_name": article_data["source_name"]
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

def display_articles(articles, prefix="article"):
    """Displays filtered and paginated articles with bookmark buttons."""
    if not articles:
        st.warning("No articles match your filter criteria or could be fetched. This could be due to:")
        st.warning("- **Website blocking:** Many news websites actively block automated access to their RSS feeds (e.g., `403 Forbidden` errors).")
        st.warning("- **RSS feed issues:** The feeds may be malformed, outdated, or contain no recent articles.")
        st.warning("- **Strict filters:** Your relevance and reputable source filters might be too strict for the available content.")
        st.warning("You may try adjusting your filters or refreshing the news.")
        return

    for idx, article in enumerate(articles):
        with st.expander(f"{article['title']} (Source: {article['source_name']})", expanded=False):
            # Create columns for metadata and bookmark button
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"**Published**: `{article['published_at']}`     **Stream**: `{article['stream']}`     **Region**: `{article['region']}`")
                st.markdown(f"🌐 [Read Full Article]({article['url']})", unsafe_allow_html=True)
            with col2:
                # Bookmark button logic
                # Only show bookmark button if logged in
                if st.session_state.logged_in:
                    is_bookmarked = any(b['link'] == article['link'] for b in st.session_state.bookmarks)
                    bookmark_label = "Remove Bookmark" if is_bookmarked else "Bookmark"
                    
                    # Use a specific key prefix for bookmark buttons to apply custom CSS
                    if st.button(
                        bookmark_label,
                        key=f"bookmark_btn_{prefix}_{idx}",
                        use_container_width=True
                    ):
                        if is_bookmarked:
                            remove_bookmark_from_firestore(article['link'])
                        else:
                            add_bookmark_to_firestore(article)
                        st.rerun()
                else:
                    st.button("Log in to Bookmark", key=f"login_to_bookmark_{prefix}_{idx}", use_container_width=True, disabled=True)

            st.info(article["summary"])

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Energy News Dashboard", layout="wide")

    selected_icon_svg_content = dark_icon_svg if st.session_state.current_theme == "dark" else light_icon_svg

    # Display title at the top
    st.markdown(
        f"""
        <h1>
            <span class='title-icon'>{selected_icon_svg_content}</span> Fuelling Insights
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Display subheader
    st.subheader("Real-Time Oil & Gas News — Filtered, Classified, Summarised")

    # In the main() function, within the st.markdown for theme CSS (both dark and light modes)

    # --- Theme CSS ---
    if st.session_state.current_theme == "dark":
        st.markdown(
            """
            <style>
            :root {
                --primary-color: #A100FF;
                --background-color: #000000;
                --secondary-background-color: #1C1E23;
                --text-color: #FFFFFF;
                --font: sans-serif;
            }
            .stApp {
                background-color: var(--background-color) !important;
                color: var(--text-color) !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, div, li, ul, a {
                color: var(--text-color) !important;
            }
            section[data-testid="stSidebar"] {
                background-color: var(--secondary-background-color) !important;
                color: #FFFFFF !important;
            }
            /* Ensure h3 in sidebar matches primary color in dark mode */
            section[data-testid="stSidebar"] h3 {
                color: #FFFFFF !important; /* This is #A100FF */
            }
            .stExpander {
                background-color: var(--secondary-background-color) !important;
                border-color: #CCC !important;
                color: var(--text-color) !important;
            }
            .stExpander > div > div {
                background-color: var(--secondary-background-color) !important;
                color: var(--text-color) !important;
            }

            /* Increase font size for news content */
            .stExpander [data-testid="stMarkdownContainer"] p,
            .stExpander [data-testid="stMarkdownContainer"] a,
            .stExpander [data-testid="stInfoContainer"] p {
                font-size: 16px !important;
                line-height: 1.6 !important;
            }
            .stExpander [data-testid="stExpanderHeader"] {
                font-size: 20px !important;
            }

            /* Ensure the eye icon in the password field is visible in dark mode */
            section[data-testid="stSidebar"] button[aria-label="Show password text"] svg,
            section[data-testid="stSidebar"] button[aria-label="Hide password text"] svg {
                fill: #000000 !important; /* Set to your text color (white in dark mode) */
            }
            code {
                color: var(--text-color) !important;
                background-color: #35373b !important;
            }
            button {
                background-color: var(--secondary-background-color) !important;
                color: var(--text-color) !important;
                border: 1px solid #CCC !important;
            }
            button:hover {
                background-color: #1c0d29 !important;
                border-color: #AAA !important;
            }
            a.btn-link {
                color: var(--text-color) !important;
            }
            .stCheckbox > label {
                color: var(--text-color) !important;
            }
            div[data-testid="stAlertContainer"] {
                background-color: #A100FF !important;
                border-left: 5px solid #A100FF !important;
                color: #FFFFFF !important;
            }
            div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentWarning"],
            div[data-testid="stAlertContentSuccess"], div[data-testid="stAlertContentError"] {
                background-color: #A100FF !important;
                color: #FFFFFF !important;
            }
            div[data-testid="stAlertContentInfo"] p, div[data-testid="stAlertContentWarning"] p,
            div[data-testid="stAlertContentSuccess"] p, div[data-testid="stAlertContentError"] p {
                color: #FFFFFF !important;
            }

            /* Fix popover container background and border */
            div[data-baseweb="popover"] {
                background-color: var(--secondary-background-color) !important;
                border: 1px solid var(--primary-color) !important;
                color: var(--text-color) !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
                border-radius: 10px !important;
            }

            /* Darken the full select popover container (that light grey box) */
            div[data-baseweb="popover"] > div {
                background-color: var(--secondary-background-color) !important;
                color: var(--text-color) !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            :root {
                --primary-color: #A100FF;
                --background-color: #FFFFFF;
                --secondary-background-color: #E6E6FA;
                --text-color: #000000;
                --font: sans-serif;
            }
            .stApp {
                background-color: var(--background-color) !important;
                color: var(--text-color) !important;
            }
            h1, h2, h3, h4, h5, h6, p, span, div, li, ul, a {
                color: var(--text-color) !important;
            }
            section[data-testid="stSidebar"] {
                background-color: var(--secondary-background-color) !important;
                color: #000000 !important;
            }
            /* Ensure h3 in sidebar matches primary color in dark mode */
            section[data-testid="stSidebar"] h3 {
                color: #000000 !important; /* This is #A100FF */
            }
            /* Ensure the eye icon in the password field is visible in dark mode */
            section[data-testid="stSidebar"] button[aria-label="Show password text"] svg,
            section[data-testid="stSidebar"] button[aria-label="Hide password text"] svg {
                fill: #000000 !important; 
            }
            .stExpander {
                background-color: var(--secondary-background-color) !important;
                border-color: #CCC !important;
                color: var(--text-color) !important;
            }
            .stExpander > div > div {
                background-color: var(--secondary-background-color) !important;
                color: var(--text-color) !important;
            }

            /* Increase font size for news content */
            .stExpander [data-testid="stMarkdownContainer"] p,
            .stExpander [data-testid="stMarkdownContainer"] a,
            .stExpander [data-testid="stInfoContainer"] p {
                font-size: 16px !important;
                line-height: 1.6 !important;
            }
            .stExpander [data-testid="stExpanderHeader"] {
                font-size: 20px !important;
            }
            code {
                color: var(--text-color) !important;
                background-color: #dbcbf5 !important;
            }
            button {
                background-color: var(--secondary-background-color) !important;
                color: var(--text-color) !important;
                border: 1px solid #CCC !important;
            }
            button:hover {
                background-color: #DDE0E3 !important;
                border-color: #AAA !important;
            }
            a.btn-link {
                color: var(--text-color) !important;
            }
            .stCheckbox > label {
                color: var(--text-color) !important;
            }
            div[data-testid="stAlertContainer"] {
                background-color: #A100FF !important;
                border-left: 5px solid #A100FF !important;
                color: #ffffff !important;
            }
            div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentWarning"],
            div[data-testid="stAlertContentSuccess"], div[data-testid="stAlertContentError"] {
                background-color: #A100FF !important;
                color: #ffffff !important;
            }
            div[data-testid="stAlertContentInfo"] p, div[data-testid="stAlertContentWarning"] p,
            div[data-testid="stAlertContentSuccess"] p, div[data-testid="stAlertContentError"] p {
                color: #ffffff !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )

    # Initialize session state for search
    if 'show_search_bar' not in st.session_state:
        st.session_state.show_search_bar = True
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""


    # Search field with integrated clear button
    search_container = st.empty()
    with search_container.container():
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.session_state.search_query = st.text_input(
                "Search articles by keywords",
                st.session_state.search_query,
                key="search_input",
                placeholder="Enter keywords...",
                label_visibility="collapsed"
            ).lower()
        with col2:
            if st.session_state.search_query:
                if st.button("✖", key="clear_search", help="Clear search"):
                    st.session_state.search_query = ""
                    st.rerun()

    # Create tabs
    tabs = st.tabs(["All News", "Bookmarks", "Events"])

    with tabs[0]:  # All News Tab
        settings_light = """
        <svg xmlns="http://www.w3.org/2000/svg" height="40px" viewBox="0 -960 960 960" width="40px" fill="#a100ff"><path d="m382-80-18.67-126.67q-17-6.33-34.83-16.66-17.83-10.34-32.17-21.67L178-192.33 79.33-365l106.34-78.67q-1.67-8.33-2-18.16-.34-9.84-.34-18.17 0-8.33.34-18.17.33-9.83 2-18.16L79.33-595 178-767.67 296.33-715q14.34-11.33 32.34-21.67 18-10.33 34.66-16L382-880h196l18.67 126.67q17 6.33 35.16 16.33 18.17 10 31.84 22L782-767.67 880.67-595l-106.34 77.33q1.67 9 2 18.84.34 9.83.34 18.83 0 9-.34 18.5Q776-452 774-443l106.33 78-98.66 172.67-118-52.67q-14.34 11.33-32 22-17.67 10.67-35 16.33L578-80H382Zm55.33-66.67h85l14-110q32.34-8 60.84-24.5T649-321l103.67 44.33 39.66-70.66L701-415q4.33-16 6.67-32.17Q710-463.33 710-480q0-16.67-2-32.83-2-16.17-7-32.17l91.33-67.67-39.66-70.66L649-638.67q-22.67-25-50.83-41.83-28.17-16.83-61.84-22.83l-13.66-110h-85l-14 110q-33 7.33-61.5 23.83T311-639l-103.67-44.33-39.66 70.66L259-545.33Q254.67-529 252.33-513 250-497 250-480q0 16.67 2.33 32.67 2.34 16 6.67 32.33l-91.33 67.67 39.66 70.66L311-321.33q23.33 23.66 51.83 40.16 28.5 16.5 60.84 24.5l13.66 110Zm43.34-200q55.33 0 94.33-39T614-480q0-55.33-39-94.33t-94.33-39q-55.67 0-94.5 39-38.84 39-38.84 94.33t38.84 94.33q38.83 39 94.5 39ZM480-480Z"/></svg>
        """
        settings_dark = """
        <svg xmlns="http://www.w3.org/2000/svg" height="40px" viewBox="0 -960 960 960" width="40px" fill="#ffffff"><path d="m382-80-18.67-126.67q-17-6.33-34.83-16.66-17.83-10.34-32.17-21.67L178-192.33 79.33-365l106.34-78.67q-1.67-8.33-2-18.16-.34-9.84-.34-18.17 0-8.33.34-18.17.33-9.83 2-18.16L79.33-595 178-767.67 296.33-715q14.34-11.33 32.34-21.67 18-10.33 34.66-16L382-880h196l18.67 126.67q17 6.33 35.16 16.33 18.17 10 31.84 22L782-767.67 880.67-595l-106.34 77.33q1.67 9 2 18.84.34 9.83.34 18.83 0 9-.34 18.5Q776-452 774-443l106.33 78-98.66 172.67-118-52.67q-14.34 11.33-32 22-17.67 10.67-35 16.33L578-80H382Zm55.33-66.67h85l14-110q32.34-8 60.84-24.5T649-321l103.67 44.33 39.66-70.66L701-415q4.33-16 6.67-32.17Q710-463.33 710-480q0-16.67-2-32.83-2-16.17-7-32.17l91.33-67.67-39.66-70.66L649-638.67q-22.67-25-50.83-41.83-28.17-16.83-61.84-22.83l-13.66-110h-85l-14 110q-33 7.33-61.5 23.83T311-639l-103.67-44.33-39.66 70.66L259-545.33Q254.67-529 252.33-513 250-497 250-480q0 16.67 2.33 32.67 2.34 16 6.67 32.33l-91.33 67.67 39.66 70.66L311-321.33q23.33 23.66 51.83 40.16 28.5 16.5 60.84 24.5l13.66 110Zm43.34-200q55.33 0 94.33-39T614-480q0-55.33-39-94.33t-94.33-39q-55.67 0-94.5 39-38.84 39-38.84 94.33t38.84 94.33q38.83 39 94.5 39ZM480-480Z"/></svg>
        """
        selected_settings_icon = settings_dark if st.session_state.current_theme == "dark" else settings_light

        st.sidebar.markdown(
            f"""
            <h2>
                <span class='sidebar-icon'>{selected_settings_icon}</span> Settings
            </h2>
            """,
            unsafe_allow_html=True
        )

        if st.sidebar.button("Toggle Dark/Light Mode"):
            st.session_state.current_theme = "light" if st.session_state.current_theme == "dark" else "dark"
            st.rerun()

        selected_filters_icon = filters_dark if st.session_state.current_theme == "dark" else filters_light

        # Authentication Section in Sidebar
        st.sidebar.subheader("User Account")
        if not st.session_state.logged_in:
            with st.sidebar.form("login_form"):
                st.markdown("#### Login / Sign Up")
                email = st.text_input("Email", key="login_email").strip()
                password = st.text_input("Password", type="password", key="login_password").strip()
                
                # Submit buttons in a single row
                col_login_signup = st.columns(2)
                with col_login_signup[0]:
                    if st.form_submit_button("Log In"):
                        if not email or not password:
                            st.sidebar.markdown(
                                """
                                <div style="width: 100%; padding: 10px; background-color: #a100ff; color: white; text-align: center; border-radius: 5px; margin-bottom: 10px;">
                                    Please enter both email and password.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            error_message = _login_user(email, password)
                            if error_message:
                                st.sidebar.markdown(
                                    f"""
                                    <div style="width: 100%; padding: 10px; background-color: #a100ff; color: white; text-align: center; border-radius: 5px; margin-bottom: 10px;">
                                        {error_message}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                with col_login_signup[1]:
                    if st.form_submit_button("Sign Up"):
                        if not email or not password:
                            st.sidebar.markdown(
                                """
                                <div style="width: 100%; padding: 10px; background-color: #a100ff; color: white; text-align: center; border-radius: 5px; margin-bottom: 10px;">
                                    Please enter both email and password.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            error_message = _signup_user(email, password)
                            if error_message:
                                st.sidebar.markdown(
                                    f"""
                                    <div style="width: 100%; padding: 10px; background-color: #a100ff; color: white; text-align: center; border-radius: 5px; margin-bottom: 10px;">
                                        {error_message}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
        else:
            st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")
            if st.sidebar.button("Log Out"):
                _logout_user()

        st.sidebar.markdown(
            f"""
            <h3>
                <span class='sidebar-icon'>{selected_filters_icon}</span> Filters
            </h3>
            """,
            unsafe_allow_html=True
        )

        # Region Checkboxes
        st.sidebar.subheader("Regions")
        region_options = [
            "India", "APAC", "Middle East", "Europe", "Africa",
            "North America", "South America", "Unclassified Region"
        ]
        region_filters = {}
        for region in region_options:
            region_filters[region] = st.sidebar.checkbox(
                region,
                value=region in st.session_state.current_region_filters,
                key=f"region_{region.lower().replace(' ', '_')}"
            )
        st.session_state.current_region_filters = [
            region for region, selected in region_filters.items() if selected
        ]

        # Stream Checkboxes
        st.sidebar.subheader("Streams")
        stream_options = ["Upstream", "Midstream", "Downstream", "Unclassified Stream"]
        stream_filters = {}
        for stream in stream_options:
            stream_filters[stream] = st.sidebar.checkbox(
                stream,
                value=stream in st.session_state.current_stream_filters,
                key=f"stream_{stream.lower().replace(' ', '_')}"
            )
        st.session_state.current_stream_filters = [
            stream for stream, selected in stream_filters.items() if selected
        ]

        # Refresh Button
        if st.sidebar.button("Refresh News"):
            st.cache_data.clear()
            st.session_state.fetch_trigger += 1
            st.rerun()

        # Fetch articles
        articles = get_processed_news_cached()

        # Apply search filter
        filtered_articles = articles
        if st.session_state.search_query:
            filtered_articles = [
                article for article in articles
                if (st.session_state.search_query in article["title"].lower() or
                    st.session_state.search_query in article["summary"].lower())
            ]

        # Apply region and stream filters
        if st.session_state.current_region_filters:
            filtered_articles = [
                article for article in filtered_articles
                if article["region"] in st.session_state.current_region_filters
            ]
        if st.session_state.current_stream_filters:
            filtered_articles = [
                article for article in filtered_articles
                if article["stream"] in st.session_state.current_stream_filters
            ]

        # Sort articles
        filtered_articles = sorted(
            filtered_articles,
            key=get_sort_key,
            reverse=True
        )

        # Source selection UI with popover
        if 'selected_sources' not in st.session_state:
            st.session_state.selected_sources = []

        col1, col2 = st.columns([0.9, 0.1])

        with col1:
            st.markdown("### Latest News")

        with col2:
            with st.popover("Source"):
                unique_sources = sorted(set(article['source_name'] for article in articles if article['source_name']))

                if not unique_sources:
                    st.info("No sources available. Please try refreshing the news.")
                else:
                    st.markdown("**Select Sources**")
                    # Track checkbox selections
                    if "selected_sources" not in st.session_state:
                        st.session_state.selected_sources = set()

                    updated_sources = set()
                    for src in unique_sources:
                        checked = src in st.session_state.selected_sources
                        if st.checkbox(src, value=checked, key=f"chk_{src}"):
                            updated_sources.add(src)

                    if st.button("Apply", key="apply_sources_popover"):
                        st.session_state.selected_sources = updated_sources
                        st.rerun()

        # Apply source filter
        filtered_articles_for_display = filtered_articles
        if st.session_state.selected_sources:
            filtered_articles_for_display = [
                article for article in filtered_articles
                if article['source_name'] in st.session_state.selected_sources
            ]

        # Display articles
        display_articles(filtered_articles_for_display[:st.session_state.displayed_article_count], prefix="news")

        # Load More Button
        if len(filtered_articles_for_display) > st.session_state.displayed_article_count:
            if st.button("Load More", key="load_more"):
                st.session_state.displayed_article_count += 10
                st.rerun()

        with tabs[1]:  # Bookmarks Tab
            st.markdown("### Bookmarked Articles")
            if not st.session_state.logged_in:
                st.info("Please log in to view and manage your private bookmarks.")
            else:
                st.info(f"""
                    **Important Note on Bookmarks:**
                    Your bookmarks are now saved privately in a **Google Cloud Firestore** database, tied to your email: `{st.session_state.user_email}`. This provides persistent storage across app reloads and different devices for *your* bookmarks.
                    
                    Currently loaded bookmarks for your session: **{len(st.session_state.bookmarks)}**
                    
                    Please remember that articles are fetched from RSS feeds, which typically only contain recent news. Older bookmarked articles might no longer be available in the live feed or on the original publisher's website if they remove or archive content.
                    """)
                
                # Add a manual reload button for debugging
                if st.button("Manually Reload Bookmarks from Database"):
                    st.session_state.bookmarks = load_bookmarks_from_firestore()
                    st.rerun()

                if not st.session_state.bookmarks:
                    st.info("No bookmarks yet. Use the bookmark button on articles to save them here.")
                else:
                    display_articles(st.session_state.bookmarks, prefix="bookmark")

        # -------------------- EVENTS TAB --------------------
        with tabs[2]:
            # Use the same theme toggle logic as other tabs
            calendar_icon_light = """
            <svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="#a100ff"><path d="M596.82-220Q556-220 528-248.18q-28-28.19-28-69Q500-358 528.18-386q28.19-28 69-28Q638-414 666-385.82q28 28.19 28 69Q694-276 665.82-248q-28.19 28-69 28ZM180-80q-24 0-42-18t-18-42v-620q0-24 18-42t42-18h65v-60h65v60h340v-60h65v60h65q24 0 42 18t18 42v620q0 24-18 42t-42 18H180Zm0-60h600v-430H180v430Zm0-490h600v-130H180v130Zm0 0v-130 130Z"/></svg>
            """
            calendar_icon_dark = """
            <svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="#ffffff"><path d="M596.82-220Q556-220 528-248.18q-28-28.19-28-69Q500-358 528.18-386q28.19-28 69-28Q638-414 666-385.82q28 28.19 28 69Q694-276 665.82-248q-28.19 28-69 28ZM180-80q-24 0-42-18t-18-42v-620q0-24 18-42t42-18h65v-60h65v60h340v-60h65v60h65q24 0 42 18t18 42v620q0 24-18 42t-42 18H180Zm0-60h600v-430H180v430Zm0-490h600v-130H180v130Zm0 0v-130 130Z"/></svg>
            """
            selected_calendar = calendar_icon_dark if st.session_state.current_theme == "dark" else calendar_icon_light
            st.markdown(f"<h1><span class='title-icon'>{selected_calendar}</span> Upcoming Energy Sector Events</h1>", unsafe_allow_html=True)

            @st.cache_data(ttl=86400)
            def fetch_et_energy_events():
                import requests
                from bs4 import BeautifulSoup
                from urllib.parse import urljoin
                import logging

                logger = logging.getLogger(__name__)
                base_url = "https://energy.economictimes.indiatimes.com"
                url = f"{base_url}/events"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Referer": base_url
                }
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    logger.info(f"Successfully fetched {url} with status {response.status_code}")
                    soup = BeautifulSoup(response.text, "html.parser")

                    events = []
                    event_items = soup.select("li.award-stroy-panel__item")
                    logger.info(f"Found {len(event_items)} event items on the page")
                    for event_item in event_items:
                        content_div = event_item.select_one("div.award-stroy-panel__content")
                        if content_div:
                            title_tag = content_div.select_one("h2")
                            points_list = content_div.select_one("ul.award-stroy-panel__points")
                            date_li = points_list.select_one("li:has(span.calender-icon)") if points_list else None
                            location_li = points_list.select_one("li:has(span.location-icon)") if points_list else None
                            link_tag = event_item.select_one("div.mid-current-promo__btn-group a.btn")

                            if title_tag and link_tag:
                                title = title_tag.get_text(strip=True)
                                event_url = link_tag.get("href")
                                date = date_li.select_one("p").get_text(strip=True) if date_li else "Date not available"
                                location = location_li.select_one("p").get_text(strip=True) if location_li else "Location not specified"

                                events.append({
                                    "title": title,
                                    "date": date,
                                    "location": location,
                                    "url": event_url
                                })
                    if not events:
                        logger.warning("No events found with current selectors.")
                    return events
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to fetch events from {url}: {e}")
                    return []

            events = fetch_et_energy_events()
            if events:
                for event in events:
                    with st.expander(f" {event['title']}", expanded=False):
                        st.caption(f"🗓 {event['date']} | 📍 {event['location']}")
                        st.markdown(f"[🔗 View Event]({event['url']})", unsafe_allow_html=True)
            else:
                st.info("No events found or failed to fetch. Check logs for details.")

            # Apply consistent styling from other tabs
            if st.session_state.current_theme == "dark":
                st.markdown("""
                    <style>
                    .stExpander { background-color: #1C1E23 !important; border-color: #CCC !important; color: #FFFFFF !important; }
                    .stExpander > div > div { background-color: #1C1E23 !important; color: #FFFFFF !important; }
                    h3 { color: #A100FF !important; }
                    </style>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <style>
                    .stExpander { background-color: #E6E6FA !important; border-color: #CCC !important; color: #000000 !important; }
                    .stExpander > div > div { background-color: #E6E6FA !important; color: #000000 !important; }
                    h3 { color: #A100FF !important; }
                    </style>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
