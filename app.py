import gradio as gr
import os
import base64
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from google.ai import generativelanguage as genai
from google.generativeai.protos import Part
from datetime import datetime, timedelta
from PIL import Image
from PIL.Image import Resampling
import io
import time
import openai
from requests.adapters import HTTPAdapter, Retry
import json
import hashlib
from datetime import datetime
import subprocess
import sys
from pathlib import Path
import tempfile
import logging
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import argostranslate.package
import argostranslate.translate
# Correct import (one line)
from langdetect import detect, DetectorFactory
import logging
import google.generativeai as genai
import os
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from flask import Flask, request, jsonify
import threading
import random
import string
import smtplib
import queue
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Detect environment and set base URL
SPACE_ID = os.getenv("SPACE_ID")
SPACE_HOST = os.getenv("SPACE_HOST")  # Hugging Face also provides this

if SPACE_ID:
    BASE_URL = f"https://{SPACE_ID.replace('/', '-')}.hf.space"
    IS_HUGGINGFACE = True
elif SPACE_HOST:
    BASE_URL = f"https://{SPACE_HOST}"
    IS_HUGGINGFACE = True
else:
    BASE_URL = "http://127.0.0.1:7860"
    IS_HUGGINGFACE = False

logging.info(f"🌐 Base URL: {BASE_URL}")
flask_app = Flask(__name__)
from flask_cors import CORS

CORS(flask_app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins for Hugging Face
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ DYNAMIC API URL HELPER
def get_api_base_url():
    """Get the correct API base URL for current environment"""
    if IS_HUGGINGFACE:
        # On Hugging Face, Flask runs on same domain but port 5000
        return BASE_URL.replace(':7860', ':5000')
    else:
        return "http://localhost:5000"

API_BASE_URL = get_api_base_url()
logging.info(f"🌐 API Base URL: {API_BASE_URL}")

MONGODB_URI = os.getenv("MONGODB_URI")

import traceback

# Now you can safely set the seed
DetectorFactory.seed = 0  # Makes detection consistent across runs

# ✅ ENHANCED LANGUAGE SUPPORT
SUPPORTED_LANGUAGES = {
    # Indian Languages (22)
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Kannada": "kn",
    "Odia": "or",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Assamese": "as",
    "Maithili": "mai",
    "Sanskrit": "sa",
    "Konkani": "kok",
    "Nepali": "ne",
    "Sindhi": "sd",
    "Dogri": "doi",
    "Kashmiri": "ks",
    "Manipuri": "mni",
    "Bodo": "brx",
    "Santali": "sat",

    # Foreign Languages (30+)
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Greek": "el",
    "Czech": "cs",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Ukrainian": "uk",
    "Indonesian": "id",
    "Malay": "ms",
    "Thai": "th",
    "Vietnamese": "vi",
    "Hebrew": "he",
    "Persian": "fa",
    "Swahili": "sw",
}

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError

    MONGODB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ pymongo not found. Install with: pip install pymongo")
    MONGODB_AVAILABLE = False

# ================================
# FIREBASE ADMIN SDK SETUP
# ================================
FIREBASE_AVAILABLE = False

try:
    # Option 1: Try service account file path from environment
    firebase_service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")

    if os.path.exists(firebase_service_account_path):
        cred = credentials.Certificate(firebase_service_account_path)

        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            logging.info("✅ Firebase already initialized")
            FIREBASE_AVAILABLE = True
        except ValueError:
            # Not initialized yet, so initialize it
            firebase_admin.initialize_app(cred)
            FIREBASE_AVAILABLE = True
            logging.info("✅ Firebase Admin SDK initialized successfully from file")
    else:
        # Option 2: Try JSON string from environment variable
        firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if firebase_creds_json:
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)

            try:
                firebase_admin.get_app()
                FIREBASE_AVAILABLE = True
            except ValueError:
                firebase_admin.initialize_app(cred)
                FIREBASE_AVAILABLE = True
                logging.info("✅ Firebase Admin SDK initialized from environment variable")
        else:
            logging.warning("⚠️ Firebase service account not configured. Google Sign-In will be unavailable.")
            FIREBASE_AVAILABLE = False

except Exception as e:
    FIREBASE_AVAILABLE = False
    logging.error(f"❌ Firebase initialization failed: {e}")
    logging.error(traceback.format_exc())

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ ADD THIS HELPER FUNCTION HERE
def get_api_base_url():
    """Get the correct API base URL for current environment"""
    if IS_HUGGINGFACE:
        # On Hugging Face, Flask runs on same domain but port 5000
        return BASE_URL.replace(':7860', ':5000')
    else:
        return "http://localhost:5000"

API_BASE_URL = get_api_base_url()
logging.info(f"🌐 API Base URL: {API_BASE_URL}")

# --- Configuration & Environment Setup ---
load_dotenv()

STABILITY_API_HOST = os.getenv("STABILITY_API_HOST", "https://api.stability.ai")
HF_API_KEY = os.getenv("HF_API_KEY")


GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # ✅ Keep only ONE

# Track which APIs are currently rate-limited
api_rate_limits = {
    "gemini": {"limited": False, "reset_time": None, "current_key_index": 0},
    "groq": {"limited": False, "reset_time": None},
    "openai": {"limited": False, "reset_time": None}
}

# Other API keys
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
# Map service API keys (optional - works without them)
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
HERE_API_KEY = os.getenv("HERE_API_KEY", "")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
# --- MongoDB Setup ---
db = None
users_collection = None

if MONGODB_AVAILABLE:
    try:
        MONGODB_URI = os.getenv("MONGODB_URI")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["manjula_ai"]
        users_collection = db["users"]
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)  # ✅ ADD THIS LINE
        users_collection.create_index("firebase_uid", unique=True, sparse=True)  # ✅ ADD THIS LINE
        logging.info("✅ MongoDB connected successfully")
        print("Number of users in the database:", users_collection.count_documents({}))
    except Exception as e:
        logging.error(f"❌ MongoDB connection failed: {e}")
        MONGODB_AVAILABLE = False

# --- External Library Imports ---
try:
    from googleapiclient.discovery import build
except ImportError:
    logging.warning("'googleapiclient' library not found. Image search will fail.")
    build = None

REPLICATE_AVAILABLE = False
try:
    import replicate

    if REPLICATE_API_TOKEN:
        REPLICATE_AVAILABLE = True
        logging.info("✅ Replicate available")
    else:
        logging.info("ℹ️ No Replicate token - will use FREE services")
except ImportError:
    logging.info("ℹ️ Replicate library not found - will use FREE services")
    replicate = None

try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    logging.warning("'speech_recognition' library not found. Install with: pip install SpeechRecognition")
    SPEECH_RECOGNITION_AVAILABLE = False

# ============================================================================
# VIDEO GENERATION SETUP - Add this block here
# ============================================================================
# Try to import imageio and numpy (required for local video generation)
IMAGEIO_AVAILABLE = False
imageio = None
np = None

try:
    import imageio
    import numpy as np

    IMAGEIO_AVAILABLE = True
    logging.info("✅ imageio and numpy available for local video generation")
except ImportError:
    logging.warning("⚠️ imageio or numpy not found. Install with: pip install imageio imageio-ffmpeg numpy")
    IMAGEIO_AVAILABLE = False

# Video generation API keys
OPENAI_SORA_API_KEY = os.getenv("OPENAI_SORA_API_KEY")
RUNWAYML_GEN2_API_KEY = os.getenv("RUNWAYML_GEN2_API_KEY")
# ============================================================================

# Rate limit timers
api_reset_times = {
    "text_qa": None,
    "image_gen": None,
    "image_qa": None,
    "image_search": None,
    "video_gen": None,
    "public_ip": None,
    "file_qa": None,
}

# Guest session history (cleared on logout/new guest)
guest_session_history = {
    "chat": [],
    "file_qa": [],
    "ip_history": [],
    "image_gen": [],
    "video_gen": [],
    "image_search": [],
    "image_qa": [],
    "public_ip": []
}

current_session_id = 0
session_isolation_lock = {}
otp_storage = {}

# User session management
current_user = {"username": "Guest", "logged_in": False, "is_guest": True}
guest_chat_count = 0
GUEST_CHAT_LIMIT = 10

# ================================
# INSTANT FIREBASE LOGIN DETECTION
# ================================
firebase_login_queue = queue.Queue()  # ✅ Thread-safe queue for instant updates


def notify_gradio_login(user_data):
    """Signal that Firebase login occurred with user data"""
    firebase_login_queue.put(user_data)
    logging.info(f"🔔 Gradio notified of Firebase login: {user_data.get('username')}")


# ================================
#  LANGUAGE TRANSLATION MODULE
# ================================
# --- Cache for installed languages ---
INSTALLED_LANGS = {}


def _ensure_language_pack(source_code: str, target_code: str):
    """Download language pack if not already installed."""
    pack_key = f"{source_code}->{target_code}"

    if pack_key in INSTALLED_LANGS:
        return INSTALLED_LANGS[pack_key]

    # Update package index
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Find package: source → target
    package_to_install = next(
        (pkg for pkg in available_packages
         if pkg.from_code == source_code and pkg.to_code == target_code),
        None
    )

    if package_to_install is None:
        # Try to find via English as intermediate
        logging.warning(f"Direct {source_code}->{target_code} not found. Trying via English...")

        # Install source -> en
        pkg_to_en = next(
            (pkg for pkg in available_packages
             if pkg.from_code == source_code and pkg.to_code == "en"),
            None
        )

        # Install en -> target
        pkg_from_en = next(
            (pkg for pkg in available_packages
             if pkg.from_code == "en" and pkg.to_code == target_code),
            None
        )

        if pkg_to_en:
            download_path = pkg_to_en.download()
            argostranslate.package.install_from_path(download_path)
            logging.info(f"Installed intermediate pack: {source_code}->en")

        if pkg_from_en:
            download_path = pkg_from_en.download()
            argostranslate.package.install_from_path(download_path)
            logging.info(f"Installed intermediate pack: en->{target_code}")

        INSTALLED_LANGS[pack_key] = True
        return True

    # Direct package found - install it
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    INSTALLED_LANGS[pack_key] = True
    logging.info(f"Installed Argos language pack: {source_code}->{target_code}")
    return True


def translate_text(text: str, target_lang: str = "es", source_lang: str = "auto") -> str:
    """
    FIXED: Multi-service translation with proper error handling
    Gemini → LibreTranslate → MyMemory → ArgosTranslate
    """
    if not text.strip():
        return ""

    # Detect source language if auto
    if source_lang == "auto":
        try:
            detected = detect(text)
            valid_codes = set(SUPPORTED_LANGUAGES.values())
            source_lang = detected if detected in valid_codes else "en"
            logging.info(f"Detected source language: {source_lang}")
        except Exception as e:
            logging.warning(f"Language detection failed: {e}. Assuming English.")
            source_lang = "en"

    # Don't translate if source and target are the same
    if source_lang == target_lang:
        return text

    # Validate target language
    valid_codes = set(SUPPORTED_LANGUAGES.values())
    if target_lang not in valid_codes:
        return f"[Translation failed: Unsupported target language '{target_lang}']"

    # ============================================================
    # METHOD 1: Gemini 2.0 Flash (PRIMARY if quota available)
    # ============================================================
    if GEMINI_CLIENT:
        try:
            logging.info("🌐 Trying Gemini 2.0 Flash translation (FREE)...")
            gemini_result = translate_text_gemini(text, target_lang, source_lang)
            if gemini_result and len(gemini_result) > 0:
                return gemini_result
        except Exception as e:
            logging.warning(f"Gemini quota exceeded, trying backup services: {e}")

    # ============================================================
    # METHOD 2: LibreTranslate (FREE, no API key)
    # ============================================================
    try:
        logging.info("🌐 Trying LibreTranslate (FREE)...")
        url = "https://libretranslate.com/translate"
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            result = response.json()
            translated = result.get("translatedText", "").strip()
            if translated and translated.lower() != text.lower():
                logging.info(f"✅ LibreTranslate success: {source_lang}->{target_lang}")
                return translated
    except Exception as e:
        logging.warning(f"LibreTranslate failed: {e}")

    # ============================================================
    # METHOD 3: MyMemory Translation API (FREE, no API key)
    # ============================================================
    try:
        logging.info("🌐 Trying MyMemory Translation API (FREE)...")
        url = f"https://api.mymemory.translated.net/get?q={quote(text)}&langpair={source_lang}|{target_lang}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            result = response.json()
            translated = result.get("responseData", {}).get("translatedText", "").strip()
            if translated and translated.lower() != text.lower():
                logging.info(f"✅ MyMemory success: {source_lang}->{target_lang}")
                return translated
    except Exception as e:
        logging.warning(f"MyMemory Translation failed: {e}")

    # ============================================================
    # METHOD 4: ArgosTranslate (OFFLINE fallback) - FIXED
    # ============================================================
    try:
        logging.info("🌐 Trying ArgosTranslate (OFFLINE)...")

        # Safely try to install language pack
        try:
            _ensure_language_pack(source_lang, target_lang)
        except Exception as pack_error:
            logging.warning(f"ArgosTranslate pack install failed: {pack_error}")
            raise Exception("Skipping ArgosTranslate")

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed_languages if lang.code == source_lang), None)
        to_lang = next((lang for lang in installed_languages if lang.code == target_lang), None)

        # Try direct translation
        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            if translation:
                translated = translation.translate(text)
                logging.info(f"✅ ArgosTranslate direct: {source_lang}->{target_lang}")
                return translated.strip()

        # Try via English if direct not available
        en_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
        if en_lang and from_lang and to_lang:
            trans_to_en = from_lang.get_translation(en_lang)
            if trans_to_en:
                intermediate_text = trans_to_en.translate(text)
                trans_from_en = en_lang.get_translation(to_lang)
                if trans_from_en:
                    translated = trans_from_en.translate(intermediate_text)
                    logging.info(f"✅ ArgosTranslate via English: {source_lang}->en->{target_lang}")
                    return translated.strip()

    except Exception as e:
        logging.warning(f"ArgosTranslate skipped: {e}")

    return f"[Translation temporarily unavailable. Gemini quota exceeded. Try again in 60 seconds.]"


def speak_translation(text: str, lang: str):
    """Generate MP3 audio of the translated text with language validation."""
    try:
        # gTTS supported languages (comprehensive list)
        gtts_langs = [
            'af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en',
            'eo', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'hr', 'hu', 'hy', 'id', 'is',
            'it', 'iw', 'ja', 'jw', 'km', 'kn', 'ko', 'la', 'lv', 'mk', 'ml', 'mr',
            'my', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sq',
            'sr', 'su', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi',
            'zh-CN', 'zh-TW', 'zh'
        ]

        # Fallback to English if language not supported
        if lang not in gtts_langs:
            logging.warning(f"⚠️ Language '{lang}' not supported by gTTS. Using English for audio.")
            lang = 'en'

        tts = gTTS(text=text, lang=lang, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        logging.info(f"✅ Audio generated for language: {lang}")
        return tmp.name
    except Exception as e:
        logging.error(f"TTS failed for language '{lang}': {e}")
        return None

def translate_text_gemini(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """
    Translate text using Gemini 2.0 Flash (PRIMARY - FREE, 100+ languages)
    """
    if not GEMINI_CLIENT:
        return None

    try:
        # Detect source if auto
        if source_lang == "auto":
            try:
                source_lang = detect(text)
            except:
                source_lang = "en"

        # Get full language names for better results
        source_name = next((name for name, code in SUPPORTED_LANGUAGES.items() if code == source_lang), source_lang)
        target_name = next((name for name, code in SUPPORTED_LANGUAGES.items() if code == target_lang), target_lang)

        # Construct translation prompt
        prompt = f"""Translate the following text from {source_name} to {target_name}.

Rules:
- Provide ONLY the translated text, nothing else
- Maintain the original meaning and tone
- Keep formatting if present
- Do not add explanations or notes

Text to translate:
{text}

Translation:"""

        model = GEMINI_CLIENT.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)

        if response and response.text:
            translated = response.text.strip()
            # Remove any common prefixes that might appear
            prefixes_to_remove = [
                "Translation:", "Here is the translation:",
                "Translated text:", "The translation is:"
            ]
            for prefix in prefixes_to_remove:
                if translated.lower().startswith(prefix.lower()):
                    translated = translated[len(prefix):].strip()

            logging.info(f"✅ Gemini translation success: {source_lang}->{target_lang}")
            return translated

        return None

    except Exception as e:
        logging.error(f"Gemini translation failed: {e}")
        return None


# --- Save translation history (only for logged-in users) ---
def save_translation_history(userid, original, translated, target_lang, src_lang="en"):
    """Save translation to MongoDB (logged-in users only)."""
    if not MONGODB_AVAILABLE or not current_user.get("logged_in"):
        return
    try:
        col = db["translation_history"]
        col.insert_one({
            "userid": userid,
            "original": original,
            "translated": translated,
            "source_lang": src_lang,
            "target_lang": target_lang,
            "timestamp": datetime.now()
        })
        logging.info(f"✅ Translation history saved for user: {userid}")
    except Exception as e:
        logging.error(f"Failed to save translation history: {e}")


def perform_translation(text, target_lang):
    """Main translation function with auto-detection"""
    if not text.strip():
        return "", None, "Please enter text to translate."

    # Auto-detect source language
    try:
        source_lang = detect(text)
        logging.info(f"Detected source language: {source_lang}")
    except Exception as e:
        logging.warning(f"Language detection failed: {e}. Assuming English.")
        source_lang = "en"

    # Check if source and target are the same
    if source_lang == target_lang:
        return text, None, f"**Original:** {text}\n\n⚠️ Source and target languages are the same ({source_lang})"

    # Translate
    translated = translate_text(text, target_lang, source_lang)

    # Generate audio
    audio_path = speak_translation(translated, target_lang)

    # Save to history (if logged in)
    if current_user.get("logged_in"):
        save_translation_history(
            current_user["username"],
            text,
            translated,
            target_lang,
            source_lang
        )

    return (
        translated,
        audio_path,
        f"**Original ({source_lang}):** {text}\n\n**Translated ({target_lang}):** {translated}"
    )


# --- User Authentication Functions ---
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


# ================================
# FIREBASE AUTHENTICATION FUNCTIONS
# ================================

def verify_firebase_token(id_token):
    """Verify Firebase ID token with clock skew tolerance"""

    try:
        firebase_admin.get_app()
    except ValueError:
        print("❌ Firebase not initialized")
        return None

    try:
        print("Calling Firebase Admin SDK verify_id_token()...")

        # ✅ FIX: Add clock_skew_seconds parameter to handle time differences
        decoded_token = firebase_auth.verify_id_token(
            id_token,
            check_revoked=False,
            clock_skew_seconds=10  # ✅ Allows 10 seconds tolerance for clock drift
        )

        print(f"✅ Token decoded successfully")
        print(f"   UID: {decoded_token.get('uid')}")
        print(f"   Email: {decoded_token.get('email')}")
        print(f"   Name: {decoded_token.get('name')}")

        return {
            "uid": decoded_token['uid'],
            "email": decoded_token.get('email'),
            "name": decoded_token.get('name'),
            "picture": decoded_token.get('picture'),
            "email_verified": decoded_token.get('email_verified', False)
        }

    except Exception as e:
        print(f"❌ Verification error: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return None


def register_or_login_firebase_user(user_info):
    """Register or login user from Firebase authentication - FIXED"""
    global current_session_id, guest_chat_count

    if not MONGODB_AVAILABLE:
        current_user["username"] = user_info["email"].split("@")[0]
        current_user["logged_in"] = True
        current_user["is_guest"] = False
        current_user["email"] = user_info["email"]
        current_user["full_name"] = user_info.get("name", "")
        return True, "Logged in successfully (no database)"

    try:
        username = user_info["email"].split("@")[0].lower()
        email = user_info["email"]
        full_name = user_info.get("name", "")

        existing_user = users_collection.find_one({"email": email})

        if existing_user:
            # ✅ LOGIN EXISTING USER
            clear_guest_history()
            guest_chat_count = 0
            current_session_id += 1

            users_collection.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.now()}}
            )

            current_user["username"] = existing_user["username"]
            current_user["logged_in"] = True
            current_user["is_guest"] = False
            current_user["email"] = email
            current_user["full_name"] = existing_user.get("full_name", full_name)

            notify_gradio_login({
                "username": username,
                "email": email,
                "full_name": full_name or username,
                "action": "login"
            })

            logging.info(f"✅ Firebase user logged in: {username}")
            return True, f"Welcome back, {full_name or username}!"

        else:
            # ✅ REGISTER NEW USER AND LOG THEM IN IMMEDIATELY
            new_user = {
                "username": username,
                "email": email,
                "full_name": full_name,
                "password": None,
                "auth_provider": "google",
                "firebase_uid": user_info["uid"],
                "profile_picture": user_info.get("picture"),
                "email_verified": user_info.get("email_verified", False),
                "created_at": datetime.now(),
                "last_login": datetime.now(),
                "usage_count": {
                    "chat": 0, "file_qa": 0, "image_gen": 0,
                    "video_gen": 0, "image_search": 0, "image_qa": 0, "maps": 0
                },
                "history": [],
                "ip_history": []
            }

            users_collection.insert_one(new_user)

            # ✅ CRITICAL FIX: LOG IN THE NEW USER IMMEDIATELY!
            clear_guest_history()
            guest_chat_count = 0
            current_session_id += 1

            current_user["username"] = username
            current_user["logged_in"] = True
            current_user["is_guest"] = False
            current_user["email"] = email
            current_user["full_name"] = full_name

            notify_gradio_login({
                "username": username,
                "email": email,
                "full_name": full_name or username,
                "action": "register"
            })

            logging.info(f"✅ New Firebase user registered AND logged in: {username}")
            return True, f"Welcome to All Mind, {full_name or username}!"

    except Exception as e:
        logging.error(f"Firebase registration/login failed: {e}")
        return False, f"Authentication failed: {str(e)}"

def request_otp(username, password, email, fullname):
    """Send OTP to email for registration"""
    if not username or not password or not email:
        return (
            "**Error:** Username, password, and email are required!",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    if len(username) < 3:
        return (
            "**Error:** Username must be at least 3 characters long!",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    if len(password) < 6:
        return (
            "**Error:** Password must be at least 6 characters long!",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # Check if user already exists
    if MONGODB_AVAILABLE:
        try:
            existing_user = users_collection.find_one({"username": username.lower().strip()})
            if existing_user:
                return (
                    f"**Error:** Username '{username}' is already taken!",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            existing_email = users_collection.find_one({"email": email.lower().strip()})
            if existing_email:
                return (
                    f"**Error:** Email '{email}' is already registered!",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        except Exception as e:
            logging.error(f"Database check error: {e}")

    # Generate 6-digit OTP
    otp = ''.join(random.choices(string.digits, k=6))

    # Store OTP temporarily (expires in 10 minutes)
    otp_storage[email] = {
        "otp": otp,
        "username": username,
        "password": password,
        "fullname": fullname,
        "expires": datetime.now() + timedelta(minutes=10)
    }

    # Send OTP via email
    try:
        if not SMTP_EMAIL or not SMTP_PASSWORD:
            return (
                "**Error:** Email service not configured. Please contact administrator.",
                gr.update(visible=False),
                gr.update(visible=False)
            )

        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = email
        msg['Subject'] = "Your Manjula AI Registration OTP"

        body = f"""
Hello {fullname or username}!

Your OTP for registering with Manjula AI is: {otp}

This OTP will expire in 10 minutes.

If you didn't request this, please ignore this email.

Best regards,
Manjula AI Team
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_EMAIL, email, text)
        server.quit()

        logging.info(f"✅ OTP sent to {email}")

        return (
            f"✅ **OTP sent successfully to {email}!**\n\nPlease check your email and enter the 6-digit code below.",
            gr.update(visible=True),  # otp_input
            gr.update(visible=True)  # verify_otp_btn
        )

    except Exception as e:
        logging.error(f"Failed to send OTP email: {e}")
        return (
            f"**Error:** Failed to send OTP. Please check your email address or try again later.\n\nError: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False)
        )


def verify_otp_and_register(email, otp):
    """Verify OTP and complete registration"""
    global current_session_id, guest_chat_count

    if not email or not otp:
        return (
            "**Error:** Please enter the OTP code!",
            gr.update(visible=True),
            gr.update(visible=True)
        )

    # Check if OTP exists and is valid
    if email not in otp_storage:
        return (
            "**Error:** No OTP found for this email. Please request a new OTP.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    stored_data = otp_storage[email]

    # Check if OTP expired
    if datetime.now() > stored_data["expires"]:
        del otp_storage[email]
        return (
            "**Error:** OTP has expired. Please request a new one.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # Verify OTP
    if otp.strip() != stored_data["otp"]:
        return (
            "**Error:** Invalid OTP! Please check and try again.",
            gr.update(visible=True),
            gr.update(visible=True)
        )

    # OTP verified - create user account
    username = stored_data["username"]
    password = stored_data["password"]
    fullname = stored_data["fullname"]

    # Remove OTP from storage
    del otp_storage[email]

    if not MONGODB_AVAILABLE:
        # Fallback without database
        clear_guest_history()
        guest_chat_count = 0
        current_session_id += 1

        current_user["username"] = username
        current_user["logged_in"] = True
        current_user["is_guest"] = False
        current_user["email"] = email
        current_user["full_name"] = fullname

        return (
            f"✅ **Registration successful!**\n\nWelcome, {fullname or username}! You can now use all features.",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # Create user in database
    try:
        new_user = {
            "username": username.lower().strip(),
            "email": email.lower().strip(),
            "full_name": fullname or "",
            "password": hash_password(password),
            "auth_provider": "email",
            "created_at": datetime.now(),
            "last_login": datetime.now(),
            "usage_count": {
                "chat": 0,
                "file_qa": 0,
                "image_gen": 0,
                "video_gen": 0,
                "image_search": 0,
                "image_qa": 0
            },
            "history": [],
            "ip_history": []
        }

        users_collection.insert_one(new_user)

        logging.info(f"✅ New user registered: {username}")

        return (
            f"✅ **Registration successful!**\n\nWelcome to Manjula AI, {fullname or username}!\n\n**Please login to continue.**",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    except DuplicateKeyError:
        return (
            f"**Error:** Username or email already exists! Please try a different one.",
            gr.update(visible=False),
            gr.update(visible=False)
        )
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return (
            f"**Error:** Registration failed. Please try again.\n\nError: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False)
        )


def clear_guest_history():
    """Clear all guest session history"""
    global guest_session_history
    guest_session_history = {
        "chat": [],
        "file_qa": [],
        "ip_history": [],
        "image_gen": [],
        "video_gen": [],
        "image_search": [],
        "image_qa": [],
        "public_ip": []
    }
    logging.info("🧹 Guest history cleared completely")


def start_as_guest():
    """Start using app as guest with limited features"""
    global guest_chat_count, current_session_id

    # ---- reset guest state -------------------------------------------------
    guest_chat_count = 0
    current_user["username"] = "Guest"
    current_user["logged_in"] = False
    current_user["is_guest"] = True

    current_session_id += 1
    clear_guest_history()

    # ---- 17 return values (order must match demo.load outputs) ------------
    return (
        # 1. guest_status (Markdown)
        "Welcome, Guest! You can try the Chat feature with 10 free messages.\n\n"
        "**Register to unlock:**\n"
        "- File Q&A\n"
        "- Image Generation\n"
        "- Video Generation\n"
        "- Image Search\n"
        "- Image Q&A\n"
        "- Usage Statistics\n"
        "- Unlimited Chat",

        # 2. auth_section (Group) – hide login/register
        gr.update(visible=False),

        # 3. main_app (Group) – show the main UI
        gr.update(visible=True),

        # 4. user_info (Markdown)
        f"Guest Mode | {guest_chat_count}/{GUEST_CHAT_LIMIT} chats used",

        # 5-9. Feature tabs – all hidden for guests
        gr.update(visible=False),  # file_qa_tab
        gr.update(visible=False),  # image_gen_tab
        gr.update(visible=False),  # image_qa_tab
        gr.update(visible=False),  # image_search_tab
        gr.update(visible=False),  # video_gen_tab

        # 10. translation_tab
        gr.update(visible=False),

        # 11. maps_tab
        gr.update(visible=False),

        # 12. stats_btn (Button) – hidden for guests
        gr.update(visible=False),

        # 13. history_chatbot (Chatbot) – empty
        [],

        # 14. guest_chat_warning (Markdown)
        gr.update(
            visible=True,
            value=(
                f"Guest Mode: You have {GUEST_CHAT_LIMIT}/{GUEST_CHAT_LIMIT} "
                "free chats remaining. Register to get unlimited access!"
            )
        ),

        # 15. chatbot (Chatbot) – empty conversation
        [],

        # 16. session_id (State)
        current_session_id,

        # 17. mic_chat (Audio) – no file yet
        None
    )


def check_auth_and_load():
    """Check authentication status and load appropriate UI"""
    global guest_chat_count, current_session_id

    # Check if user is already logged in (from Firebase or previous session)
    if current_user.get("logged_in") and not current_user.get("is_guest"):
        username = current_user.get("username", "User")
        logging.info(f"✅ User already logged in on page load: {username}")

        # Return LOGGED IN state (16 values - same as login_user function)
        return (
            f"✅ **Welcome back, {username}!** All features unlocked.",
            gr.update(visible=False),  # auth_section
            gr.update(visible=True),  # main_app
            f"**Logged in as:** {username}",
            gr.update(visible=True),  # file_qa_tab
            gr.update(visible=True),  # image_gen_tab
            gr.update(visible=True),  # image_qa_tab
            gr.update(visible=True),  # image_search_tab
            gr.update(visible=True),  # video_gen_tab
            gr.update(visible=True),  # translation_tab
            gr.update(visible=True),  # maps_tab
            gr.update(visible=True),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )
    else:
        # Not logged in - start as guest (existing function)
        logging.info("🆕 Starting fresh guest session")
        return start_as_guest()


# ================================
# FIREBASE WEB CONFIG (for frontend)  ← ADD THIS
# ================================
# ================================
# FIREBASE WEB CONFIG (for frontend
# ================================
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY", ""),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": os.getenv("FIREBASE_APP_ID", ""),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "")
}

firebase_config_json = json.dumps(firebase_config)

# ✅ CORRECTED VERSION - Replace your existing register_html and login_html with this:

print("=" * 80)
print("🔍 FIREBASE CONFIG DEBUG")
print("=" * 80)
print("Firebase config JSON:", firebase_config_json)
print("apiKey present:", bool(firebase_config.get("apiKey")))
print("authDomain present:", bool(firebase_config.get("authDomain")))
print("projectId present:", bool(firebase_config.get("projectId")))
print("=" * 80)

register_html = f"""
    <div style="padding: 20px; text-align: center;">
        <a href="{API_BASE_URL}/firebase-auth?action=register" target="_blank" style="
            background-color: #4285f4;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
        ">
            <svg width="18" height="18" viewBox="0 0 48 48">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            </svg>
            Continue with Google
        </a>
        <p style="margin-top: 15px; font-size: 14px; color: #666;">
            Opens in new window → Returns here automatically
        </p>
    </div>
    """

login_html = f"""
    <div style="padding: 20px; text-align: center;">
        <a href="{API_BASE_URL}/firebase-auth?action=login" target="_blank" style="
            background-color: #4285f4;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
        ">
            <svg width="18" height="18" viewBox="0 0 48 48">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            </svg>
            Continue with Google
        </a>
        <p style="margin-top: 15px; font-size: 14px; color: #666;">
            Opens in new window → Returns here automatically
        </p>
    </div>
    """
def login_user(username, password):
    """Login user - FIXED with complete session isolation & 17 outputs"""
    global current_session_id, guest_chat_count

    # === 1. MONGODB NOT AVAILABLE (Fallback) ===
    if not MONGODB_AVAILABLE:
        current_session_id += 1
        clear_guest_history()
        guest_chat_count = 0,
        current_user["username"] = username
        current_user["logged_in"] = True
        current_user["is_guest"] = False

        return (
            f"**Welcome back, {username}!**\n\nYou have full access to all features!",
            gr.update(visible=False),  # auth_section
            gr.update(visible=True),  # main_app
            f"**Logged in as:** {username}",
            gr.update(visible=True),  # file_qa_tab
            gr.update(visible=True),  # image_gen_tab
            gr.update(visible=True),  # image_qa_tab
            gr.update(visible=True),  # image_search_tab
            gr.update(visible=True),  # video_gen_tab
            gr.update(visible=True),  # translation_tab
            gr.update(visible=True),  # maps_tab
            gr.update(visible=True),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )

    # === 2. EMPTY CREDENTIALS ===
    if not username or not password:
        return (
            "**Username and password are required!**",
            gr.update(visible=True),  # auth_section
            gr.update(visible=False),  # main_app
            "**Not logged in**",
            gr.update(visible=False),  # file_qa_tab
            gr.update(visible=False),  # image_gen_tab
            gr.update(visible=False),  # image_qa_tab
            gr.update(visible=False),  # image_search_tab
            gr.update(visible=False),  # video_gen_tab
            gr.update(visible=False),  # translation_tab
            gr.update(visible=False),  # maps_tab
            gr.update(visible=False),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )

    # === 3. VALID LOGIN ===
    try:
        user = users_collection.find_one({"username": username.lower().strip()})

        if user and user["password"] == hash_password(password):
            clear_guest_history()
            guest_chat_count = 0
            current_session_id += 1

            current_user["username"] = username.lower().strip()
            current_user["logged_in"] = True
            current_user["is_guest"] = False

            try:
                client_ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
            except Exception:
                client_ip = "Unknown"

            users_collection.update_one(
                {"username": username.lower().strip()},
                {
                    "$addToSet": {"ip_history": client_ip},
                    "$set": {"last_login": datetime.now()}
                }
            )
            current_user["last_ip"] = client_ip

            logging.info(f"User logged in: {username} | Session ID: {current_session_id} | Guest data WIPED")

            return (
                f"**Welcome back, {username}!**\n\nYou have full access to all features!",
                gr.update(visible=False),  # auth_section
                gr.update(visible=True),  # main_app
                f"**Logged in as:** {username}",
                gr.update(visible=True),  # file_qa_tab
                gr.update(visible=True),  # image_gen_tab
                gr.update(visible=True),  # image_qa_tab
                gr.update(visible=True),  # image_search_tab
                gr.update(visible=True),  # video_gen_tab
                gr.update(visible=True),  # translation_tab
                gr.update(visible=True),  # maps_tab
                gr.update(visible=True),  # stats_btn
                [],  # history_chatbot
                gr.update(visible=False),  # guest_chat_warning
                [],  # chatbot
                current_session_id,  # session_id
                None  # mic_chat
            )
        else:
            # === 4. INVALID CREDENTIALS ===
            return (
                "**Invalid username or password!**",
                gr.update(visible=True),  # auth_section
                gr.update(visible=False),  # main_app
                "**Not logged in**",
                gr.update(visible=False),  # file_qa_tab
                gr.update(visible=False),  # image_gen_tab
                gr.update(visible=False),  # image_qa_tab
                gr.update(visible=False),  # image_search_tab
                gr.update(visible=False),  # video_gen_tab
                gr.update(visible=False),  # translation_tab
                gr.update(visible=False),  # maps_tab
                gr.update(visible=False),  # stats_btn
                [],  # history_chatbot
                gr.update(visible=True),  # guest_chat_warning
                [],  # chatbot
                current_session_id,  # session_id
                None  # mic_chat
            )

    # === 5. EXCEPTION ===
    except Exception as e:
        logging.error(f"Login error: {e}")
        return (
            f"**Login failed:** {str(e)}",
            gr.update(visible=True),  # auth_section
            gr.update(visible=False),  # main_app
            "**Not logged in**",
            gr.update(visible=False),  # file_qa_tab
            gr.update(visible=False),  # image_gen_tab
            gr.update(visible=False),  # image_qa_tab
            gr.update(visible=False),  # image_search_tab
            gr.update(visible=False),  # video_gen_tab
            gr.update(visible=False),  # translation_tab
            gr.update(visible=False),  # maps_tab
            gr.update(visible=False),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=True),  # guest_chat_warning
            [],  # chatbot
            current_session_id,  # session_id
            None  # mic_chat
        )

def logout_user():
    """FIXED: Logout with all 15 return values"""
    global current_session_id, guest_chat_count

    username = current_user.get("username", "Unknown")

    current_user["username"] = None
    current_user["logged_in"] = False
    current_user["is_guest"] = False

    guest_chat_count = 0
    clear_guest_history()
    current_session_id += 1

    logging.info(f"✅ User logged out: {username} | Session ID: {current_session_id} | ALL SESSION DATA WIPED")

    return (
        "✅ Logged out successfully! Please login or register to continue.",
        gr.update(visible=True),  # auth_section
        gr.update(visible=False),  # main_app
        "👤 **Not logged in**",  # user_info
        gr.update(visible=False),  # file_qa_tab
        gr.update(visible=False),  # image_gen_tab
        gr.update(visible=False),  # image_qa_tab
        gr.update(visible=False),  # image_search_tab
        gr.update(visible=False),  # video_gen_tab
        gr.update(visible=False),  # translation_tab
        gr.update(visible=False),  # maps_tab
        gr.update(visible=False),  # stats_btn
        [],  # history_chatbot
        gr.update(visible=False),  # guest_chat_warning
        [],  # chatbot
        current_session_id,  # session_id
        None  # mic_chat ✅ THIS WAS MISSING!
    )


def format_history_for_chatbot():
    """Fetch ONLY current user's history - COMPLETELY ISOLATED"""
    chatbot_history = []

    if current_user.get("is_guest", False):
        logging.info("📋 Loading GUEST session history (temporary)")

        entries = guest_session_history.get("chat", [])

        if entries:
            chatbot_history.append((
                f"--- **💬 Guest Session History** ({len(entries)} messages) ---",
                "**⚠️ TEMPORARY DATA - Will be deleted on logout**"
            ))

            for entry in entries:
                timestamp = entry.get("timestamp", "")
                full_query = entry.get("full_query", "No Query")
                full_response = entry.get("full_response", "No Response")

                chatbot_history.append((
                    f"**{timestamp}** | Q: {full_query}",
                    f"A: {full_response}"
                ))
        else:
            chatbot_history.append((
                None,
                "📭 **No guest history yet.**\n\nStart chatting to see your activity here!\n\n🔒 **Note:** Guest history is temporary. Register to save permanently."
            ))

        return chatbot_history

    if not current_user.get("logged_in", False) or not current_user.get("username"):
        logging.warning("⚠️ Attempted to load history without login")
        return [(None, "📭 **Please login to view your history.**")]

    if not MONGODB_AVAILABLE:
        return [(None, "❌ **Database not available.**")]

    try:
        username = current_user["username"]
        logging.info(f"📋 Loading history for user: {username}")

        user_doc = users_collection.find_one(
            {"username": username},
            {"history": 1, "_id": 0}
        )

        if not user_doc or not user_doc.get("history"):
            logging.info(f"📭 User {username} has no history")
            return [(
                None,
                f"📭 **No history for: {username}**\n\nStart using features to see your activity here!"
            )]

        total_items = len(user_doc["history"])
        chatbot_history.append((
            f"--- **📚 {username.upper()}'s Complete History** ({total_items} items) ---",
            "**All YOUR interactions. This data belongs ONLY to you.**"
        ))

        feature_icons = {
            "chat": "💬", "file_qa": "📄", "image_gen": "🎨",
            "video_gen": "🎥", "image_search": "🖼️", "image_qa": "🔍", "public_ip": "🌐"
        }

        feature_groups = {}
        for entry in user_doc["history"]:
            feature = entry.get("feature", "chat")
            feature_groups.setdefault(feature, []).append(entry)

        for feature, entries in feature_groups.items():
            icon = feature_icons.get(feature, "📋")
            feature_name = feature.replace("_", " ").title()

            chatbot_history.append((
                f"--- **{icon} {feature_name}** ({len(entries)} items) ---",
                ""
            ))

            for entry in entries:
                timestamp = entry.get("timestamp")
                if hasattr(timestamp, "strftime"):
                    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                text = entry.get("text", "No Query")[:200]
                reply = entry.get("reply", "No Response")[:300]

                chatbot_history.append((
                    f"**{timestamp}** | Q: {text}",
                    f"A: {reply}"
                ))

        logging.info(f"✅ Loaded {total_items} history items for {username}")
        return chatbot_history

    except Exception as e:
        logging.error(f"❌ Error loading history for {current_user.get('username', 'unknown')}: {e}")
        return [(None, f"❌ **Error loading history:** {e}")]


def check_feature_access(feature_name):
    """Check if user can access a feature"""
    if current_user["logged_in"]:
        return None

    if current_user["is_guest"] and feature_name != "chat":
        return (
            f"🔒 **Feature Locked: {feature_name.replace('_', ' ').title()}**\n\n"
            f"This feature is only available to registered users.\n\n"
            f"**Register now to unlock:**\n"
            f"- 📄 File Q&A - Upload and analyze any document\n"
            f"- 🎨 Image Generation - Create AI art from text\n"
            f"- 🎥 Video Generation - Generate AI videos\n"
            f"- 🖼️ Image Search - Search Google Images\n"
            f"- 🔍 Image Q&A - Ask questions about images\n"
            f"- 📊 Usage Statistics - Track your activity\n"
            f"- ♾️ Unlimited Chat - No message limits\n\n"
            f"👉 **Click 'Logout' and then 'Register' to create a free account!**"
        )

    return None


def check_guest_chat_limit():
    """Check if guest has exceeded chat limit"""
    global guest_chat_count

    if current_user["is_guest"]:
        if guest_chat_count >= GUEST_CHAT_LIMIT:
            return (
                f"🚫 **Guest Chat Limit Reached ({GUEST_CHAT_LIMIT}/{GUEST_CHAT_LIMIT})**\n\n"
                f"You've used all your free guest messages!\n\n"
                f"**Register now to get:**\n"
                f"- ♾️ **Unlimited Chat** - Chat as much as you want\n"
                f"- 📄 **File Q&A** - Upload and analyze documents\n"
                f"- 🎨 **Image Generation** - Create AI art\n"
                f"- 🎥 **Video Generation** - Generate AI videos\n"
                f"- 🖼️ **Image Search** - Search Google Images\n"
                f"- 🔍 **Image Q&A** - Ask questions about images\n"
                f"- 📊 **Usage Statistics** - Track all your activity\n\n"
                f"Registration is **100% FREE** and takes less than 1 minute!\n\n"
                f"👉 **Click 'Logout' and then 'Register' to unlock all features!**"
            )

    return None


def increment_usage(feature):
    """Increment usage count for a feature"""
    if MONGODB_AVAILABLE and current_user["logged_in"]:
        try:
            users_collection.update_one(
                {"username": current_user["username"]},
                {"$inc": {f"usage_count.{feature}": 1}}
            )
        except Exception as e:
            logging.error(f"Failed to increment usage: {e}")


def get_user_stats():
    """Get user statistics"""
    if not MONGODB_AVAILABLE or not current_user["logged_in"]:
        return "❌ Please login to view statistics."

    try:
        user = users_collection.find_one({"username": current_user["username"]})
        if user:
            stats = f"""
## 📊 Your Usage Statistics

**Username:** {user['username']}
**Email:** {user.get('email', 'N/A')}
**Full Name:** {user.get('full_name', 'N/A')}
**Member Since:** {user['created_at'].strftime('%Y-%m-%d %H:%M:%S')}
**Last Login:** {user['last_login'].strftime('%Y-%m-%d %H:%M:%S') if user.get('last_login') else 'N/A'}

### Feature Usage:
- 💬 Chat: {user['usage_count']['chat']} times
- 📄 File Q&A: {user['usage_count']['file_qa']} times
- 🎨 Image Generation: {user['usage_count']['image_gen']} times
- 🎥 Video Generation: {user['usage_count']['video_gen']} times
- 🖼️ Image Search: {user['usage_count']['image_search']} times
- 🔍 Image Q&A: {user['usage_count']['image_qa']} times
- 🗺️ Maps & Location: {user['usage_count'].get('maps', 0)} times

**Total Usage:** {sum(user['usage_count'].values())} actions
"""
            return stats
    except Exception as e:
        logging.error(f"Failed to get user stats: {e}")
        return f"❌ Failed to load statistics: {str(e)}"


# ================================
# API KEY ROTATION & TRACKING
# ================================
import random
from datetime import datetime, timedelta

# Track API key usage and cooldowns
api_key_status = {
    0: {"limited": False, "reset_time": None, "uses": 0},
    1: {"limited": False, "reset_time": None, "uses": 0},
    2: {"limited": False, "reset_time": None, "uses": 0}
}
current_gemini_key_index = 0


def get_next_available_gemini_key():
    """Get next available Gemini API key with smart rotation"""
    global current_gemini_key_index

    now = datetime.now()

    # First, reset any expired cooldowns
    for idx in api_key_status:
        if api_key_status[idx]["limited"] and api_key_status[idx]["reset_time"]:
            if now >= api_key_status[idx]["reset_time"]:
                api_key_status[idx]["limited"] = False
                api_key_status[idx]["reset_time"] = None
                api_key_status[idx]["uses"] = 0
                logging.info(f"✅ Gemini API key {idx + 1} cooldown expired - now available")

    # Try to find an available key
    for attempt in range(len(GEMINI_API_KEYS)):
        test_index = (current_gemini_key_index + attempt) % len(GEMINI_API_KEYS)

        if GEMINI_API_KEYS[test_index] and not api_key_status[test_index]["limited"]:
            current_gemini_key_index = test_index
            api_key_status[test_index]["uses"] += 1

            logging.info(f"🔑 Using Gemini API key #{test_index + 1} (uses: {api_key_status[test_index]['uses']})")
            return GEMINI_API_KEYS[test_index], test_index

    # All keys are rate-limited
    logging.error("❌ ALL Gemini API keys are rate-limited!")
    return None, None


def mark_gemini_key_limited(key_index, retry_seconds=60):
    """Mark a specific Gemini key as rate-limited"""
    if key_index is not None:
        api_key_status[key_index]["limited"] = True
        api_key_status[key_index]["reset_time"] = datetime.now() + timedelta(seconds=retry_seconds)
        logging.warning(f"⚠️ Gemini API key #{key_index + 1} rate-limited. Reset in {retry_seconds}s")


# ================================
# INITIALIZE GEMINI CLIENT
# ================================
GEMINI_CLIENT = None

# Initialize with first available key
first_key = next((key for key in GEMINI_API_KEYS if key), None)
if first_key:
    try:
        genai.configure(api_key=first_key)
        GEMINI_CLIENT = genai
        logging.info("🟢 Gemini Client successfully initialized.")
    except Exception as e:
        logging.error(f"❌ Failed to initialize Gemini Client: {e}")
        GEMINI_CLIENT = None
else:
    logging.error("❌ No GEMINI_API_KEY set in environment variables.")


# ================================
# MULTI-API HELPER FUNCTIONS (Keep existing ones)
# ================================
def mark_api_limited(api_name, retry_seconds=60):
    """Mark an API as rate-limited"""
    api_rate_limits[api_name]["limited"] = True
    api_rate_limits[api_name]["reset_time"] = datetime.now() + timedelta(seconds=retry_seconds)
    logging.warning(f"⚠️ {api_name.upper()} rate-limited. Reset in {retry_seconds}s")


def is_api_available(api_name):
    """Check if API is available (not rate-limited)"""
    if not api_rate_limits[api_name]["limited"]:
        return True

    if datetime.now() >= api_rate_limits[api_name]["reset_time"]:
        api_rate_limits[api_name]["limited"] = False
        api_rate_limits[api_name]["reset_time"] = None
        logging.info(f"✅ {api_name.upper()} rate limit expired - now available")
        return True

    return False


def check_rate_limit(task_key):
    reset_time = api_reset_times.get(task_key)
    now = datetime.now()
    if reset_time and now < reset_time:
        remaining = reset_time - now
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"⚠️ You have reached the {task_key.replace('_', ' ')} limit. Try again in {hours}h {minutes}m {seconds}s."
    return None


def get_timer_text(task_key, label=None):
    reset_time = api_reset_times.get(task_key)
    now = datetime.now()
    label = label or task_key.replace('_', ' ').capitalize()
    if reset_time and now < reset_time:
        remaining = reset_time - now
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"⏳ {label} limit reached. Refreshes in {hours}h {minutes}m {seconds}s."
    return f"✅ {label} available now."


def set_rate_limit(task_key):
    api_reset_times[task_key] = datetime.now() + timedelta(hours=12)


def create_session_with_retries(total_retries=5, backoff_factor=0.5):
    status_forcelist = [429, 500, 502, 503, 504]
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods={"POST"},
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def add_to_guest_history(category, user_query, response, metadata=None):
    """Add guest interaction to session-only history (not saved to DB)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_query_str = str(user_query).strip() if user_query is not None else ""
    response_str = str(response).strip() if response is not None else ""

    entry = {
        "timestamp": timestamp,
        "query": user_query_str[:100],
        "response": response_str[:200],
        "full_query": user_query_str,
        "full_response": response_str,
        "metadata": metadata or {}
    }
    guest_session_history[category].insert(0, entry)

    if len(guest_session_history[category]) > 50:
        guest_session_history[category] = guest_session_history[category][:50]


def save_interaction_to_db(feature_name, user_query, ai_response, metadata=None):
    """Save interaction to MongoDB for logged-in users only"""

    if current_user["is_guest"]:
        guest_session_history.setdefault("chat", [])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        guest_session_history["chat"].append({
            "timestamp": timestamp,
            "full_query": user_query,
            "full_response": ai_response,
            "feature": feature_name,
            "metadata": metadata or {}
        })
        return

    if not MONGODB_AVAILABLE:
        return

    if not current_user["logged_in"] or not current_user["username"]:
        return

    try:
        history_entry = {
            "timestamp": datetime.now(),
            "feature": feature_name,
            "text": str(user_query)[:500] if user_query else "No Query",
            "reply": str(ai_response)[:1000] if ai_response else "No Response",
            "metadata": metadata or {}
        }

        users_collection.update_one(
            {"username": current_user["username"]},
            {"$push": {"history": history_entry}},
        )

        logging.info(f"✅ Saved {feature_name} interaction for user: {current_user['username']}")

    except Exception as e:
        logging.error(f"Failed to save interaction to MongoDB: {e}")


def show_history_modal():
    """Open history modal and populate the chatbot"""
    return gr.update(visible=True), format_history_for_chatbot()


def close_history_modal():
    """Close history modal"""
    return gr.update(visible=False)


def clear_all_history_action():
    """Clear history from UI only - DATABASE REMAINS UNTOUCHED"""
    if current_user["is_guest"]:
        clear_guest_history()
        return [], "✅ Guest session history cleared from UI!"

    elif current_user["logged_in"] and MONGODB_AVAILABLE:
        return [], "✅ History cleared from UI! (Database unchanged - click 'Refresh' to reload)"

    return [], "❌ Unable to clear history."


def transcribe_audio(audio_filepath):
    """Transcribe audio file to text using Google Speech Recognition"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return "❌ Speech recognition library not installed. Please run: pip install SpeechRecognition"

    if audio_filepath is None:
        return ""

    try:
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_filepath) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "❌ Could not understand audio. Please speak clearly."
    except sr.RequestError as e:
        return f"❌ Speech recognition service error: {e}"
    except Exception as e:
        return f"❌ Error processing audio: {e}"


# ================================
#  ANY-TO-ANY TRANSLATION FUNCTIONS
# ================================

def _detect_language(text: str) -> str:
    """Auto-detect source language (fallback = en)"""
    try:
        return detect(text)
    except Exception:
        return "en"


def translate_any_to_any(text: str, target_lang: str):
    """Detect source → translate → speak → return markdown + audio path"""
    if current_user["is_guest"]:
        return ("**Register** to use translation.", None)

    if not text.strip():
        return ("Please enter text to translate.", None)

    src_lang = _detect_language(text)
    logging.info(f"Detected source: {src_lang} → target: {target_lang}")

    translated, audio_path, full_msg = perform_translation(text, target_lang)

    if current_user.get("logged_in"):
        save_translation_history(
            current_user["username"],
            text,
            translated,
            target_lang,
            src_lang
        )

    return full_msg, audio_path


def transcribe_and_translate_any(audio_filepath, target_lang):
    """Voice → transcribe → detect → translate → speak"""
    if current_user["is_guest"]:
        return ("**Register** to use voice translation.", None)

    if not audio_filepath:
        return ("No audio recorded.", None)

    text = transcribe_audio(audio_filepath)
    if text.startswith("Could not") or text.startswith("SpeechRecognition"):
        return (text, None)

    return translate_any_to_any(text, target_lang)


def check_guest_feature_access(feature_name):
    """Check if guest user can access a specific feature"""
    if current_user["is_guest"]:
        return f"🚫 **Guest users can only use Text Chat.** Please login or register to access {feature_name}."
    return None


def query_model(prompt, history, session_id_state):
    """Chat function with GROQ FIRST, then Gemini rotation, then OpenAI"""
    global current_session_id, current_gemini_key_index

    # Session validation
    if session_id_state != current_session_id:
        logging.warning(f"⚠️ Session mismatch detected! Clearing stale history. "
                        f"Expected: {current_session_id}, Got: {session_id_state}")
        history = []
        session_id_state = current_session_id

    logging.info(f"📥 query_model | User: {current_user.get('username', 'Unknown')} | "
                 f"Guest: {current_user.get('is_guest')} | History len: {len(history)} | "
                 f"Session: {session_id_state} | Current: {current_session_id}")

    if not prompt or not prompt.strip():
        return history, "", session_id_state

    limit_check = check_guest_chat_limit()
    if limit_check:
        history.append((prompt, limit_check))
        return history, "", session_id_state

    limit_msg = check_rate_limit("text_qa")
    if limit_msg:
        history.append((prompt, limit_msg))
        return history, "", session_id_state

    global guest_chat_count
    if current_user["is_guest"]:
        guest_chat_count += 1
    else:
        increment_usage("chat")

    # Build message history
    llm_messages = []
    for user_msg, assistant_msg in history:
        if user_msg:
            llm_messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            llm_messages.append({"role": "model", "content": assistant_msg})

    llm_messages.append({"role": "user", "content": prompt})

    answer = None
    llm_name = "N/A"

    # ============================================================================
    # 🚀 PRIORITY 1: Try Groq FIRST (FREE, Fast, High Limits)
    # ============================================================================
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        llm_name = "Groq"
        try:
            groq_messages = []
            for msg in llm_messages:
                role = "assistant" if msg["role"] == "model" else msg["role"]
                groq_messages.append({"role": role, "content": msg["content"]})

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": groq_messages
                },
                timeout=30
            )

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                logging.info("✅ Groq (Primary) success!")
            else:
                logging.warning(f"⚠️ Groq failed with status {response.status_code}, trying Gemini...")

        except Exception as e:
            logging.warning(f"⚠️ Groq error: {e}, trying Gemini...")

    # ============================================================================
    # 🔥 PRIORITY 2: Try Gemini with ALL keys (if Groq failed)
    # ============================================================================
    if (answer is None or not answer) and GEMINI_CLIENT:
        for attempt in range(len(GEMINI_API_KEYS)):
            api_key, key_index = get_next_available_gemini_key()

            if not api_key:
                logging.warning(f"⚠️ All Gemini keys exhausted, trying OpenAI...")
                break

            try:
                # Reconfigure with new key
                genai.configure(api_key=api_key)

                # Format messages for Gemini
                gemini_formatted_messages = []
                for msg in llm_messages:
                    role = "model" if msg["role"] == "assistant" else msg["role"]
                    gemini_formatted_messages.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}],
                    })

                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(gemini_formatted_messages)
                answer = response.text
                llm_name = f"Gemini (Key #{key_index + 1})"

                logging.info(f"✅ Gemini success with key #{key_index + 1}")
                break  # Success! Exit loop

            except Exception as e:
                err_str = str(e).lower()

                if "quota" in err_str or "429" in err_str:
                    logging.warning(f"⚠️ Gemini key #{key_index + 1} quota exceeded, trying next key...")
                    mark_gemini_key_limited(key_index, 60)  # 1 minute cooldown
                    continue  # Try next key
                else:
                    # Other error - log and try next key
                    logging.error(f"❌ Gemini key #{key_index + 1} error: {e}")
                    mark_gemini_key_limited(key_index, 30)
                    continue

    # ============================================================================
    # 🔄 PRIORITY 3: Fallback to OpenAI (last resort)
    # ============================================================================
    if (answer is None or not answer) and OPENAI_KEY:
        llm_name = "OpenAI"
        try:
            client = openai.OpenAI(api_key=OPENAI_KEY)
            openai_formatted_messages = []
            for msg in llm_messages:
                role = "assistant" if msg["role"] == "model" else msg["role"]
                openai_formatted_messages.append({"role": role, "content": msg["content"]})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_formatted_messages
            )
            answer = response.choices[0].message.content.strip()
            logging.info("✅ OpenAI (Fallback) success!")
        except Exception as e:
            logging.error(f"❌ OpenAI API Error: {e}")

    # ============================================================================
    # Final fallback
    # ============================================================================
    if answer is None or not answer:
        answer = (
            "❌ **All AI services temporarily unavailable**\n\n"
            "**What we tried:**\n"
            "1. ⚡ Groq (Primary) - Failed or rate-limited\n"
            "2. 🔥 Gemini (3 keys) - All quota exceeded\n"
            "3. 🤖 OpenAI - Unavailable\n\n"
            "**Solutions:**\n"
            "- Wait 1-2 minutes for Groq to reset\n"
            "- Try again in 60 seconds for Gemini keys to refresh\n"
            "- Check your API keys in .env file"
        )

    history.append((prompt, answer))

    # Save to history
    if current_user["is_guest"]:
        add_to_guest_history("chat", prompt, answer, {"model": llm_name})
        logging.info(f"💬 Guest chat saved to SESSION ONLY (not in DB) - Model: {llm_name}")
    else:
        save_interaction_to_db("chat", prompt, answer, {"model": llm_name})
        logging.info(f"💬 User '{current_user['username']}' chat saved to DB - Model: {llm_name}")

    return history, "", session_id_state

def process_audio_and_chat(audio_filepath, history, session_id_state):
    """Process audio input, transcribe it, and get AI response - WITH SESSION VALIDATION"""
    global current_session_id

    # ✅ CRITICAL: Validate session BEFORE processing
    if session_id_state != current_session_id:
        logging.warning(f"⚠️ Voice input detected stale session. FORCING CLEAR. "
                        f"Expected: {current_session_id}, Got: {session_id_state}")
        history = []
        session_id_state = current_session_id

    # ✅ ADDITIONAL CHECK: If history exists but session changed, clear it
    if history and len(history) > 0:
        if session_id_state != current_session_id:
            logging.warning(f"⚠️ Clearing {len(history)} stale messages from previous user")
            history = []
            session_id_state = current_session_id

    if audio_filepath is None:
        return history, "", update_guest_warning(), session_id_state

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return history + [(None, transcribed_text)], "", update_guest_warning(), session_id_state

    result_history, result_input, result_session = query_model(transcribed_text, history, session_id_state)
    warning = update_guest_warning()

    logging.info(f"🎤 Voice chat processed | User: {current_user.get('username')} | "
                 f"Session: {result_session} | History items: {len(result_history)}")

    return result_history, result_input, warning, result_session


def start_new_chat():
    """Start a fresh chat session - clears UI only, not database"""
    global current_session_id
    current_session_id += 1

    logging.info(f"🆕 New chat started | User: {current_user.get('username', 'Guest')} | "
                 f"New Session ID: {current_session_id}")

    return [], current_session_id


def update_guest_warning():
    """Update guest warning message"""
    global guest_chat_count
    if current_user["is_guest"]:
        remaining = GUEST_CHAT_LIMIT - guest_chat_count
        if remaining > 0:
            return gr.update(
                value=f"⚠️ **Guest Mode:** You have {remaining}/{GUEST_CHAT_LIMIT} free chats remaining. Register to get unlimited access!",
                visible=True)
        else:
            return gr.update(value=f"🚫 **Guest limit reached!** Please register to continue chatting.",
                             visible=True)
    return gr.update(visible=False)


def query_and_update_warning(prompt, history, session_id_state):
    """Query model and update warning - with session validation"""
    result_history, result_input, result_session = query_model(prompt, history, session_id_state)
    warning = update_guest_warning()
    return result_history, result_input, warning, result_session


def on_user_input_detect_translation(user_input, user_is_registered):
    if user_is_registered and user_input.lower().startswith("translate"):
        return gr.update(visible=True), gr.update(value="")
    else:
        return gr.update(visible=False), user_input


# ============================================================================
# ULTRA-ROBUST MAPS with AUTO-FALLBACK SYSTEM
# ============================================================================
# ============================================================================
# FIXED: Maps with NO rate limiting for GPS location requests
# ============================================================================

# ============================================================================
# FIXED: Maps with NO rate limiting for GPS location requests
# ============================================================================

def generate_ultra_robust_map(mode, location, origin, destination, nearby_location, nearby_type):
    """
    ✅ FULLY WORKING: Maps with LIVE GPS navigation (like Google Maps)
    ✅ FIXED: No rate limiting on GPS location requests - users can fetch as many times as needed
    """
    guest_check = check_guest_feature_access("Maps & Location")
    if guest_check:
        return "", guest_check

    # ✅ CRITICAL FIX: ONLY rate limit MAP GENERATION, NOT GPS location requests
    # Check if this is a GPS location request (nearby_location starts with specific coordinates pattern)
    is_gps_request = nearby_location and ',' in nearby_location and nearby_location.count(',') >= 1

    # Only apply rate limiting for actual map generation, NOT for GPS coordinate fetching
    if not is_gps_request:  # ✅ Skip rate limit for GPS requests
        from datetime import datetime, timedelta
        current_time = datetime.now()
        rate_limit_key = f"maps_{current_user.get('username', 'guest')}"

        if not hasattr(generate_ultra_robust_map, 'rate_limits'):
            generate_ultra_robust_map.rate_limits = {}

        if rate_limit_key in generate_ultra_robust_map.rate_limits:
            last_request = generate_ultra_robust_map.rate_limits[rate_limit_key]
            time_diff = (current_time - last_request).total_seconds()
            if time_diff < 2:
                remaining = 2 - time_diff
                return "", f"⚠️ Please wait {remaining:.1f} seconds before next map request."

        generate_ultra_robust_map.rate_limits[rate_limit_key] = current_time

    # Input validation (keep existing functions)
    def validate_location_input(text, field_name="location"):
        if not text or not text.strip():
            return None, f"❌ Please enter a {field_name}."
        text = text.strip()
        if len(text) > 200:
            return None, f"❌ {field_name.capitalize()} too long (max 200 characters)."
        if len(text) < 2:
            return None, f"❌ {field_name.capitalize()} too short (min 2 characters)."

        dangerous_patterns = [
            '<script', '</script>', 'javascript:', 'onerror=', 'onload=',
            '<iframe', '</iframe>', '<object', '<embed', 'data:text/html'
        ]
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in text_lower:
                return None, f"❌ Invalid {field_name}: contains blocked characters."

        return text, None

    try:
        # ========================================================================
        # MODE 1: SEARCH LOCATION (same as before)
        # ========================================================================
        if mode == "Search Location":
            location, error = validate_location_input(location, "location")
            if error:
                return "", error

            js_safe_location = location.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n',
                                                                                                              '\\n')

            map_html = f'''
<div style="width: 100%; height: 650px; position: relative;">
    <iframe style="width:100%;height:100%;border:none;background-color:#e5e3df" 
            srcdoc="<!DOCTYPE html>
<html>
<head>
    <meta charset=&quot;utf-8&quot;>
    <link rel=&quot;stylesheet&quot; href=&quot;https://unpkg.com/leaflet@1.9.4/dist/leaflet.css&quot;/>
    <style>
        body{{margin:0;padding:0;background:#e5e3df}}
        #map{{width:100%;height:600px;background:#e5e3df}}
        #status{{position:absolute;top:10px;left:50%;transform:translateX(-50%);background:#2196F3;color:white;padding:12px 24px;border-radius:8px;z-index:1000;font-size:15px;box-shadow:0 4px 12px rgba(0,0,0,0.4);font-weight:500}}
        .success{{background:#4CAF50!important}}
        .error{{background:#f44336!important}}
    </style>
</head>
<body>
    <div id=&quot;status&quot;>🔍 Searching...</div>
    <div id=&quot;map&quot;></div>
    <script src=&quot;https://unpkg.com/leaflet@1.9.4/dist/leaflet.js&quot;></script>
    <script>
        const LOC = '{js_safe_location}';
        const s = document.getElementById('status');

        function show(t, c) {{
            s.textContent = t;
            s.className = c;
            s.style.display = 'block';
            if (c === 'success') {{
                setTimeout(() => s.style.display = 'none', 4000);
            }}
        }}

        const map = L.map('map').setView([20, 0], 2);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }}).addTo(map);

        (async () => {{
            try {{
                show('🔍 Searching: ' + LOC, '');

                const url = 'https://nominatim.openstreetmap.org/search?format=json&q=' + 
                           encodeURIComponent(LOC) + '&limit=1';

                const response = await fetch(url, {{
                    headers: {{ 'User-Agent': 'AllMindApp/1.0' }}
                }});

                if (!response.ok) {{
                    throw new Error('HTTP ' + response.status);
                }}

                const data = await response.json();

                if (data && data.length > 0) {{
                    const lat = parseFloat(data[0].lat);
                    const lon = parseFloat(data[0].lon);

                    map.setView([lat, lon], 13);

                    L.marker([lat, lon])
                        .addTo(map)
                        .bindPopup('<b>' + data[0].display_name + '</b>')
                        .openPopup();

                    show('✅ Found: ' + data[0].display_name.substring(0, 60), 'success');
                }} else {{
                    show('❌ Location not found', 'error');
                }}
            }} catch (e) {{
                console.error('❌ Error:', e);
                show('❌ Error: ' + e.message, 'error');
            }}
        }})();
    </script>
</body>
</html>">
    </iframe>
</div>
'''

            info = f"📍 **Location:** {location}\n\n🗺️ **Service:** OpenStreetMap (FREE)"

            if current_user["is_guest"]:
                add_to_guest_history("maps", location, "Location viewed", {"service": "OpenStreetMap"})
            else:
                save_interaction_to_db("maps", location, "Location viewed", {"service": "OpenStreetMap"})
                increment_usage("maps")

            return map_html, info

        # ========================================================================
        # MODE 2: GET DIRECTIONS
        # ========================================================================
        elif mode == "Get Directions":
            origin, error1 = validate_location_input(origin, "origin")
            if error1:
                return "", error1

            destination, error2 = validate_location_input(destination, "destination")
            if error2:
                return "", error2

            js_safe_origin = origin.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
            js_safe_dest = destination.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

            # [Keep the existing Get Directions map_html code - no changes needed]
            # ... [rest of Get Directions code remains the same]

        # ========================================================================
        # MODE 3: FIND NEARBY - ✅ FIXED FOR UNLIMITED GPS REQUESTS
        # ========================================================================
        else:  # Find Nearby
            # ✅ Accept empty location to use GPS
            if not nearby_type or not nearby_type.strip():
                return "", "❌ Please enter what you want to search for (e.g., 'pizza', 'gym', 'school')"

            nearby_type = nearby_type.strip().lower()

            # Security check only
            dangerous_patterns = ['<script', 'javascript:', 'onerror=']
            for pattern in dangerous_patterns:
                if pattern in nearby_type.lower():
                    return "", "❌ Invalid search term"

            # ✅ Check if location is provided or should use GPS
            use_gps = not nearby_location or not nearby_location.strip()

            if use_gps:
                location_text = "your current location"
                js_safe_loc = ""  # Empty means use GPS
            else:
                nearby_location, error1 = validate_location_input(nearby_location, "location")
                if error1:
                    return "", error1
                location_text = nearby_location
                js_safe_loc = nearby_location.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

            js_safe_type = nearby_type.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

            # ✅ CRITICAL FIX: Increased timeout to 60 seconds for GPS requests
            map_html = f'''
<div style="width: 100%; height: 650px; position: relative;">
    <iframe style="width:100%;height:100%;border:none;background-color:#e5e3df" 
            srcdoc="<!DOCTYPE html>
<html>
<head>
    <meta charset=&quot;utf-8&quot;>
    <link rel=&quot;stylesheet&quot; href=&quot;https://unpkg.com/leaflet@1.9.4/dist/leaflet.css&quot;/>
    <style>
        body{{margin:0;padding:0;background:#e5e3df}}
        #map{{width:100%;height:600px}}
        #status{{position:absolute;top:10px;left:50%;transform:translateX(-50%);background:#2196F3;color:white;padding:12px 24px;border-radius:8px;z-index:1000;font-size:15px;box-shadow:0 4px 12px rgba(0,0,0,0.4);white-space:nowrap}}
        .success{{background:#4CAF50!important}}
        .error{{background:#f44336!important}}
        .info{{background:#FF9800!important}}
    </style>
</head>
<body>
    <div id=&quot;status&quot;>🔍 Initializing...</div>
    <div id=&quot;map&quot;></div>
    <script src=&quot;https://unpkg.com/leaflet@1.9.4/dist/leaflet.js&quot;></script>
    <script>
        const LOC = '{js_safe_loc}';
        const TYPE = '{js_safe_type}';
        const USE_GPS = {str(use_gps).lower()};
        const s = document.getElementById('status');

        function show(t, c) {{
            s.textContent = t;
            s.className = c;
            s.style.display = 'block';
        }}

        const map = L.map('map').setView([20, 0], 2);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19
        }}).addTo(map);

        // GPS location marker (blue dot)
        const gpsIcon = L.divIcon({{
            className: '',
            html: `<div style=&quot;width:20px;height:20px;background:#4285f4;border:4px solid white;border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,0.4);&quot;></div>`,
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        }});

        async function searchNearby(lat, lon, locationName) {{
            try {{
                map.setView([lat, lon], 14);

                // Add blue marker for user's location
                L.marker([lat, lon], {{ icon: gpsIcon }})
                    .addTo(map)
                    .bindPopup('<b>📍 ' + locationName + '</b>')
                    .openPopup();

                show('🔍 Finding nearby ' + TYPE + '...', '');

                // ✅ SMART SEARCH: Try multiple methods
                let found = false;

                // Method 1: Try Overpass API with flexible query
                try {{
                    const overpassQuery = `
                        [out:json][timeout:25];
                        (
                          node["name"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                          node["amenity"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                          node["shop"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                          node["leisure"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                          way["name"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                          way["amenity"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                          way["shop"~"${{TYPE}}",i](around:3000,${{lat}},${{lon}});
                        );
                        out center 20;
                    `;

                    const poiResp = await fetch(
                        'https://overpass-api.de/api/interpreter?data=' + encodeURIComponent(overpassQuery),
                        {{ headers: {{ 'User-Agent': 'AllMindApp/1.0' }} }}
                    );
                    const poiData = await poiResp.json();

                    if (poiData.elements && poiData.elements.length > 0) {{
                        poiData.elements.forEach(p => {{
                            const pLat = p.lat || (p.center && p.center.lat);
                            const pLon = p.lon || (p.center && p.center.lon);
                            if (pLat && pLon) {{
                                const name = p.tags.name || p.tags.amenity || p.tags.shop || 'Unnamed';
                                L.marker([pLat, pLon]).addTo(map).bindPopup('<b>' + name + '</b>');
                            }}
                        }});
                        show('✅ Found ' + poiData.elements.length + ' results for: ' + TYPE, 'success');
                        found = true;
                    }}
                }} catch (e) {{
                    console.log('Overpass search error:', e);
                }}

                // Method 2: Fallback to Nominatim search
                if (!found) {{
                    const searchQuery = TYPE + ' near ' + locationName;
                    const nomResp = await fetch(
                        'https://nominatim.openstreetmap.org/search?format=json&q=' + 
                        encodeURIComponent(searchQuery) + '&limit=10',
                        {{ headers: {{ 'User-Agent': 'AllMindApp/1.0' }} }}
                    );
                    const nomData = await nomResp.json();

                    if (nomData && nomData.length > 0) {{
                        nomData.forEach(place => {{
                            const pLat = parseFloat(place.lat);
                            const pLon = parseFloat(place.lon);
                            L.marker([pLat, pLon]).addTo(map)
                                .bindPopup('<b>' + place.display_name + '</b>');
                        }});
                        show('✅ Found ' + nomData.length + ' results for: ' + TYPE, 'success');
                        found = true;
                    }}
                }}

                if (!found) {{
                    show('❌ No results found for: ' + TYPE + '. Try different keywords.', 'error');
                }}

            }} catch (e) {{
                console.error(e);
                show('❌ Error: ' + e.message, 'error');
            }}
        }}

        (async () => {{
            try {{
                if (USE_GPS) {{
                    // ✅ USE GPS LOCATION - INCREASED TIMEOUT TO 60 SECONDS
                    show('📍 Requesting location access...', 'info');

                    if (!navigator.geolocation) {{
                        show('❌ GPS not supported by your browser', 'error');
                        return;
                    }}

                    navigator.geolocation.getCurrentPosition(
                        async (position) => {{
                            const lat = position.coords.latitude;
                            const lon = position.coords.longitude;
                            console.log('✅ Got GPS location:', lat, lon);

                            show('✅ Location detected! Searching nearby...', 'success');

                            // Get location name from coordinates
                            try {{
                                const reverseResp = await fetch(
                                    `https://nominatim.openstreetmap.org/reverse?format=json&lat=${{lat}}&lon=${{lon}}`,
                                    {{ headers: {{ 'User-Agent': 'AllMindApp/1.0' }} }}
                                );
                                const reverseData = await reverseResp.json();
                                const locationName = reverseData.display_name || 'Your Location';

                                await searchNearby(lat, lon, locationName);
                            }} catch (e) {{
                                await searchNearby(lat, lon, 'Your Current Location');
                            }}
                        }},
                        (error) => {{
                            console.error('GPS Error:', error);
                            let errorMsg = '❌ Location access denied';
                            switch(error.code) {{
                                case error.PERMISSION_DENIED:
                                    errorMsg = '❌ Please allow location access to find nearby places';
                                    break;
                                case error.POSITION_UNAVAILABLE:
                                    errorMsg = '❌ Location unavailable. Please enter location manually.';
                                    break;
                                case error.TIMEOUT:
                                    errorMsg = '❌ Location request timeout. Click &quot;Use My Current Location&quot; again.';
                                    break;
                            }}
                            show(errorMsg, 'error');
                        }},
                        {{
                            enableHighAccuracy: true,
                            timeout: 60000,  // ✅ INCREASED TO 60 SECONDS
                            maximumAge: 0
                        }}
                    );

                }} else {{
                    // ✅ USE MANUAL LOCATION
                    show('🔍 Finding location: ' + LOC, '');

                    const locResp = await fetch(
                        'https://nominatim.openstreetmap.org/search?format=json&q=' + 
                        encodeURIComponent(LOC) + '&limit=1',
                        {{ headers: {{ 'User-Agent': 'AllMindApp/1.0' }} }}
                    );
                    const locData = await locResp.json();

                    if (!locData || locData.length === 0) {{
                        show('❌ Location not found', 'error');
                        return;
                    }}

                    const lat = parseFloat(locData[0].lat);
                    const lon = parseFloat(locData[0].lon);

                    await searchNearby(lat, lon, LOC);
                }}

            }} catch (e) {{
                console.error(e);
                show('❌ Error: ' + e.message, 'error');
            }}
        }})();
    </script>
</body>
</html>">
    </iframe>
</div>
'''

            info = f"🔍 **Searching for:** {nearby_type} near {location_text}\n\n🗺️ **Service:** OpenStreetMap + Overpass API (FREE)\n\n💡 **Tip:** {'You can request your location as many times as needed - no timeout limits!' if use_gps else 'Using your current GPS location'}"

            if current_user["is_guest"]:
                add_to_guest_history("maps", f"{nearby_type}@{location_text}", "Nearby search",
                                     {"service": "Overpass + Nominatim", "gps": use_gps})
            else:
                save_interaction_to_db("maps", f"{nearby_type}@{location_text}", "Nearby search",
                                       {"service": "Overpass + Nominatim", "gps": use_gps})
                increment_usage("maps")

            return map_html, info

    except Exception as e:
        logging.error(f"Maps error: {e}")
        logging.error(traceback.format_exc())
        return "", f"❌ Error: {str(e)}"

def get_public_ip():
    """Get public IP with history saving"""
    limit_msg = check_rate_limit("public_ip")
    if limit_msg:
        return limit_msg
    try:
        resp = requests.get('https://api.ipify.org', timeout=10)
        resp.raise_for_status()
        ip = resp.text.strip()
        result = f"Your current Public IP Address is: **{ip}**"

        if current_user["is_guest"]:
            add_to_guest_history("public_ip", "Get Public IP", result, {"ip": ip})
        else:
            save_interaction_to_db("public_ip", "Get Public IP", result, {"ip": ip})

        return result
    except Exception as e:
        error_msg = f"Error: {e}"

        if not current_user["is_guest"]:
            save_interaction_to_db("public_ip", "Get Public IP", error_msg)

        return error_msg


# --- File Content Extraction WITH TIMER ---
def extract_file_content_gemini(file, prompt):
    """
    ULTRA ROBUST: Extract file content with ALL available AI models
    Priority: Local Text Extraction → Groq (with retry) → Gemini (3 keys) → OpenAI
    """
    guest_check = check_guest_feature_access("File Q&A")
    if guest_check:
        yield "🔒 Access Denied", guest_check
        return

    if not file:
        yield "⏱️ 0s", "Error: No file uploaded."
        return

    uploaded_file = None
    try:
        start_time = time.time()
        yield "⏱️ 1s", "⏳ **Step 1/4:** Processing file..."

        file_path = file.name
        ext = os.path.splitext(file_path)[-1].lower()

        # ============================================================================
        # METHOD 1: LOCAL TEXT EXTRACTION (No API needed - FASTEST)
        # ============================================================================

        # --- TXT Files ---
        if ext in ['.txt']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                elapsed = int(time.time() - start_time)

                yield f"⏱️ {elapsed}s", "⏳ Analyzing text with AI..."
                result = answer_question_from_content(content, prompt)

                if current_user["is_guest"]:
                    add_to_guest_history("file_qa", prompt, result, {"filename": file.name})
                else:
                    save_interaction_to_db("file_qa", prompt, result, {"filename": file.name})

                yield f"⏱️ {elapsed}s ✅", result
                return
            except Exception as e:
                yield "⏱️ 2s", f"Error reading text file: {e}"
                return

        # --- PDF Files (Text Extraction) ---
        if ext in ['.pdf']:
            try:
                import PyPDF2

                yield "⏱️ 2s", "⏳ Extracting text from PDF (no API needed)..."

                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text_content = ""

                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n\n"

                if text_content.strip():
                    elapsed = int(time.time() - start_time)
                    yield f"⏱️ {elapsed}s", "⏳ Analyzing extracted text with AI..."

                    result = answer_question_from_content(text_content, prompt)

                    if current_user["is_guest"]:
                        add_to_guest_history("file_qa", prompt, result, {"filename": file.name})
                    else:
                        save_interaction_to_db("file_qa", prompt, result, {"filename": file.name})

                    elapsed = int(time.time() - start_time)
                    yield f"⏱️ {elapsed}s ✅", result
                    return
                else:
                    # PDF is image-based, continue to vision APIs
                    yield "⏱️ 3s", "⚠️ PDF appears to be image-based. Trying vision API..."

            except ImportError:
                yield "⏱️ 2s", "⚠️ PyPDF2 not installed. Install with: pip install PyPDF2\nTrying vision API..."
            except Exception as e:
                logging.error(f"PDF text extraction failed: {e}")
                yield "⏱️ 3s", f"⚠️ PDF text extraction failed. Trying vision API..."

        # --- DOCX Files ---
        if ext in ['.docx']:
            try:
                import docx

                yield "⏱️ 2s", "⏳ Extracting text from DOCX..."

                doc = docx.Document(file_path)
                text_content = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])

                if text_content.strip():
                    elapsed = int(time.time() - start_time)
                    result = answer_question_from_content(text_content, prompt)

                    if current_user["is_guest"]:
                        add_to_guest_history("file_qa", prompt, result, {"filename": file.name})
                    else:
                        save_interaction_to_db("file_qa", prompt, result, {"filename": file.name})

                    elapsed = int(time.time() - start_time)
                    yield f"⏱️ {elapsed}s ✅", result
                    return

            except ImportError:
                yield "⏱️ 2s", "Error: python-docx not installed. Install with: pip install python-docx"
                return
            except Exception as e:
                yield "⏱️ 2s", f"Error reading DOCX: {e}"
                return

        # ============================================================================
        # METHOD 2: VISION APIs (For images and image-based PDFs)
        # ============================================================================

        elapsed = int(time.time() - start_time)
        yield f"⏱️ {elapsed}s", "⏳ **Step 2/4:** Uploading to Gemini..."

        try:
            uploaded_file = genai.upload_file(path=file_path)
            elapsed = int(time.time() - start_time)
            yield f"⏱️ {elapsed}s", "⏳ **Step 3/4:** Extracting content with AI..."
            time.sleep(2)
        except Exception as e:
            logging.error(f"Gemini file upload failed: {e}")
            yield f"⏱️ {elapsed}s", "⚠️ File upload failed, trying direct image processing..."

        extraction_prompt = (
            f"Analyze the attached document/image thoroughly. Extract ALL text, tables, "
            f"charts, and key information. Format the output as clean, well-structured Markdown. "
            f"Then address this specific request: '{prompt}'"
        )

        # ============================================================================
        # METHOD 3: GROQ WITH RETRY (FREE, Fast, High Quality)
        # ============================================================================
        if GROQ_API_KEY and ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            max_groq_retries = 3
            for retry in range(max_groq_retries):
                try:
                    elapsed = int(time.time() - start_time)
                    retry_suffix = f" (Retry {retry + 1}/{max_groq_retries})" if retry > 0 else ""
                    yield f"⏱️ {elapsed}s", f"🚀 **Trying Groq Llama Vision (FREE){retry_suffix}...**"

                    import base64
                    with open(file_path, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.2-90b-vision-preview",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": extraction_prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            "max_tokens": 2000
                        },
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()["choices"][0]["message"]["content"]
                        elapsed = int(time.time() - start_time)
                        final_result = f"**✅ Extraction Complete! (took {elapsed}s, Groq Llama Vision)**\n\n{result}"

                        if current_user["is_guest"]:
                            add_to_guest_history("file_qa", prompt, final_result, {"filename": file.name})
                        else:
                            save_interaction_to_db("file_qa", prompt, final_result,
                                                   {"filename": file.name, "model": "Groq Llama Vision"})

                        yield f"⏱️ {elapsed}s ✅", final_result
                        return

                    elif response.status_code == 429:
                        # Rate limited - retry with exponential backoff
                        if retry < max_groq_retries - 1:
                            wait_time = 2 ** retry  # 1s, 2s, 4s
                            elapsed = int(time.time() - start_time)
                            yield f"⏱️ {elapsed}s", f"⚠️ Groq rate limited. Waiting {wait_time}s before retry..."
                            time.sleep(wait_time)
                            continue
                        else:
                            logging.warning(f"Groq failed after {max_groq_retries} retries: 429")
                            yield f"⏱️ {elapsed}s", "⚠️ Groq rate limit exceeded. Trying Gemini..."
                            break
                    else:
                        logging.warning(f"Groq failed with status {response.status_code}")
                        if retry < max_groq_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            yield f"⏱️ {elapsed}s", "⚠️ Groq unavailable. Trying Gemini..."
                            break

                except Exception as e:
                    logging.error(f"Groq vision processing failed (attempt {retry + 1}): {e}")
                    if retry < max_groq_retries - 1:
                        elapsed = int(time.time() - start_time)
                        yield f"⏱️ {elapsed}s", f"⚠️ Groq error. Retrying in {2 ** retry}s..."
                        time.sleep(2 ** retry)
                        continue
                    else:
                        elapsed = int(time.time() - start_time)
                        yield f"⏱️ {elapsed}s", f"⚠️ Groq error after retries. Trying Gemini..."
                        break

        # ============================================================================
        # METHOD 4: ALL GEMINI KEYS WITH ROTATION
        # ============================================================================
        if uploaded_file:
            contents = [uploaded_file, extraction_prompt]

            for attempt in range(len(GEMINI_API_KEYS)):
                api_key, key_index = get_next_available_gemini_key()

                if not api_key:
                    break

                try:
                    elapsed = int(time.time() - start_time)
                    yield f"⏱️ {elapsed}s", f"🔥 **Trying Gemini Key #{key_index + 1}...**"

                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    response = model.generate_content(contents)

                    if response and response.text:
                        elapsed = int(time.time() - start_time)
                        final_result = f"**✅ Extraction Complete! (took {elapsed}s, Gemini Key #{key_index + 1})**\n\n{response.text}"

                        if current_user["is_guest"]:
                            add_to_guest_history("file_qa", prompt, final_result, {"filename": file.name})
                        else:
                            save_interaction_to_db("file_qa", prompt, final_result,
                                                   {"filename": file.name, "model": f"Gemini #{key_index + 1}"})

                        yield f"⏱️ {elapsed}s ✅", final_result
                        return
                    else:
                        yield f"⏱️ {elapsed}s", f"Empty response from Gemini #{key_index + 1}, trying next..."
                        continue

                except Exception as e:
                    err_str = str(e).lower()
                    elapsed = int(time.time() - start_time)

                    if "quota" in err_str or "429" in err_str:
                        logging.warning(f"⚠️ File Q&A: Gemini key #{key_index + 1} quota exceeded")
                        mark_gemini_key_limited(key_index, 60)
                        yield f"⏱️ {elapsed}s", f"Gemini #{key_index + 1} quota exceeded, trying next..."
                        continue
                    else:
                        logging.error(f"Gemini key #{key_index + 1} error: {e}")
                        yield f"⏱️ {elapsed}s", f"Error with key #{key_index + 1}, trying next..."
                        continue

        # ============================================================================
        # METHOD 5: OPENAI GPT-4 VISION (Fallback)
        # ============================================================================
        if OPENAI_KEY and ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            try:
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s", "🤖 **Trying OpenAI GPT-4 Vision (Fallback)...**"

                import base64
                with open(file_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')

                client = openai.OpenAI(api_key=OPENAI_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": extraction_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000
                )

                result = response.choices[0].message.content
                elapsed = int(time.time() - start_time)
                final_result = f"**✅ Extraction Complete! (took {elapsed}s, OpenAI GPT-4)**\n\n{result}"

                if current_user["is_guest"]:
                    add_to_guest_history("file_qa", prompt, final_result, {"filename": file.name})
                else:
                    save_interaction_to_db("file_qa", prompt, final_result,
                                           {"filename": file.name, "model": "OpenAI GPT-4"})

                yield f"⏱️ {elapsed}s ✅", final_result
                return

            except Exception as e:
                logging.error(f"OpenAI GPT-4 Vision failed: {e}")
                elapsed = int(time.time() - start_time)
                yield f"⏱️ {elapsed}s ❌", f"OpenAI also failed: {str(e)}"

        # ============================================================================
        # ALL SERVICES FAILED
        # ============================================================================
        elapsed = int(time.time() - start_time)
        error_msg = (
            f"❌ **All AI services temporarily unavailable** (took {elapsed}s)\n\n"
            "**What happened:**\n"
            "1. 🚀 Groq Llama Vision - Rate limited after 3 retries\n"
            "2. 🔥 All 3 Gemini API keys - Quota exceeded\n"
            "3. 🤖 OpenAI GPT-4 Vision - Failed\n\n"
            "**✅ SOLUTIONS THAT WORK NOW:**\n\n"
            "**1. For Text PDFs:**\n"
            "   - Install PyPDF2: `pip install PyPDF2`\n"
            "   - Re-upload your file (text will be extracted without API)\n\n"
            "**2. For DOCX files:**\n"
            "   - Install python-docx: `pip install python-docx`\n"
            "   - Re-upload (text extracted locally)\n\n"
            "**3. For Images/Scanned PDFs:**\n"
            "   - ⏳ **Wait 60 seconds** - Gemini quota resets every minute\n"
            "   - 🔄 **Try again** - services refresh quickly\n\n"
            "**4. Check API Quotas:**\n"
            "   - Gemini: https://ai.google.dev/usage\n"
            "   - Groq: https://console.groq.com\n"
            "   - OpenAI: https://platform.openai.com/usage"
        )

        if not current_user["is_guest"]:
            save_interaction_to_db("file_qa", prompt, error_msg,
                                   {"filename": file.name, "error": "all_services_failed"})

        yield f"⏱️ {elapsed}s ❌", error_msg

    except Exception as e:
        elapsed = int(time.time() - start_time)
        error_msg = f"**❌ File extraction failed:** {str(e)}\n\nPlease try:\n- A different file format\n- A smaller file\n- Waiting a moment and trying again"
        logging.error(f"File extraction error: {e}")
        logging.error(traceback.format_exc())
        yield f"⏱️ {elapsed}s ❌", error_msg

    finally:
        # Clean up uploaded file from Gemini
        if uploaded_file and hasattr(uploaded_file, 'name') and uploaded_file.name:
            try:
                time.sleep(1)
                genai.delete_file(name=uploaded_file.name)
                logging.info(f"✅ Cleaned up file: {uploaded_file.name}")
            except Exception as cleanup_error:
                logging.warning(f"File cleanup warning: {cleanup_error}")


def answer_question_from_content(file_content, user_question):
    """
    FIXED: Use LLM to answer user's question based on extracted file content
    Now includes Groq as PRIMARY option (FREE, high limits)
    """
    if not GEMINI_CLIENT and not OPENAI_KEY and not GROQ_API_KEY:
        return f"**Extracted Content:**\n\n{file_content}\n\n---\n\n⚠️ No LLM available to answer your question. Please configure GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY."

    max_content_length = 30000
    if len(file_content) > max_content_length:
        file_content = file_content[:max_content_length] + "\n\n[Content truncated due to length...]"

    system_prompt = f"""You are a highly capable AI assistant. A user has uploaded a file and wants your help.

**File Content:**
{file_content}

**User's Request:**
{user_question}

Instructions:
- The user can ask you to do ANYTHING with this file content - be completely flexible and helpful.
- Understand what the user is asking for and provide exactly that.
- If you're unsure what they want, ask for clarification while still providing your best interpretation.

Now, based on the user's request above, provide the most helpful and appropriate response."""

    answer = None
    llm_name = "N/A"

    # ============================================================================
    # PRIORITY 1: Try Groq FIRST (FREE, high limits, fast)
    # ============================================================================
    if GROQ_API_KEY:
        llm_name = "Groq"
        try:
            logging.info("🚀 Trying Groq for File Q&A answer...")

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system",
                         "content": "You are a helpful AI assistant that analyzes file content and answers questions."},
                        {"role": "user", "content": system_prompt}
                    ]
                },
                timeout=60
            )

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                logging.info("✅ Groq success for File Q&A!")
            else:
                logging.warning(f"⚠️ Groq failed with status {response.status_code}, trying Gemini...")
                answer = None

        except Exception as e:
            logging.error(f"Groq File Q&A error: {e}")
            answer = None

    # ============================================================================
    # PRIORITY 2: Try Gemini (if Groq failed)
    # ============================================================================
    if (answer is None or not answer) and GEMINI_CLIENT:
        llm_name = "Gemini"
        try:
            logging.info("🔥 Trying Gemini for File Q&A answer...")
            model = GEMINI_CLIENT.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(system_prompt)

            if response and response.text:
                answer = response.text
                logging.info("✅ Gemini success for File Q&A!")
        except Exception as e:
            logging.error(f"Gemini API Error in File Q&A: {e}")
            answer = None

    # ============================================================================
    # PRIORITY 3: Try OpenAI (last resort)
    # ============================================================================
    if (answer is None or not answer) and OPENAI_KEY:
        llm_name = "OpenAI"
        try:
            logging.info("🤖 Trying OpenAI for File Q&A answer...")
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful AI assistant that can do anything the user requests with uploaded file content."},
                    {"role": "user", "content": system_prompt}
                ],
                timeout=60
            )
            answer = response.choices[0].message.content.strip()
            logging.info("✅ OpenAI success for File Q&A!")
        except Exception as e:
            logging.error(f"OpenAI API Error in File Q&A: {e}")
            answer = None

    # ============================================================================
    # ALL SERVICES FAILED
    # ============================================================================
    if answer is None or not answer.strip():
        return (
            f"**Extracted Content:**\n\n{file_content}\n\n---\n\n"
            f"⚠️ **All AI services temporarily unavailable**\n\n"
            f"**What we tried:**\n"
            f"1. Groq (FREE) - {('Success' if llm_name == 'Groq' else 'Failed or unavailable')}\n"
            f"2. Gemini - Quota exceeded\n"
            f"3. OpenAI - Quota exceeded\n\n"
            f"**Solutions:**\n"
            f"- Wait 60 seconds and try again\n"
            f"- Check your Groq API key: https://console.groq.com\n"
            f"- Here's the extracted content for manual review"
        )

    return f"**Response to your request:**\n\n{answer}\n\n---\n\n**Source:** Based on content from uploaded file (processed by {llm_name})"

def file_question_answer(file, question):
    """FIXED: File Q&A with proper generator handling"""
    access_check = check_feature_access("file_qa")
    if access_check:
        yield "⏱️ 0s", access_check
        return

    limit_msg = check_rate_limit("file_qa")
    if limit_msg:
        yield "⏱️ 0s", limit_msg
        return

    if not file or not question:
        yield "⏱️ 0s", "Please upload a file and enter a question."
        return

    increment_usage("file_qa")

    # Track the last timer and content
    last_timer = "⏱️ 0s"
    last_content = ""

    # Extract content
    for timer, content in extract_file_content_gemini(file, question):
        last_timer = timer
        last_content = content
        yield timer, content  # Stream updates

    # Check for extraction errors
    if last_content and (
            last_content.lower().startswith("error:") or
            last_content.lower().startswith("extraction failed:")
    ):
        # Save error to history
        if current_user["is_guest"]:
            add_to_guest_history("file_qa", question, last_content,
                                 {"filename": file.name if file else "unknown"})
        else:
            save_interaction_to_db("file_qa", question, last_content,
                                   {"filename": file.name if file else "unknown"})
        return  # Exit on error

    # Process successful extraction
    if last_content and "✅ Extraction Complete!" in last_content:
        # Extract just the content (remove status message)
        content_parts = last_content.split("\n\n", 1)
        extracted_content = content_parts[1] if len(content_parts) > 1 else last_content

        yield last_timer, "⏳ **Generating AI response based on extracted content...**"

        # Generate AI response
        result = answer_question_from_content(extracted_content, question)

        # Save to history
        if current_user["is_guest"]:
            add_to_guest_history("file_qa", question, result,
                                 {"filename": file.name if file else "unknown"})
        else:
            save_interaction_to_db("file_qa", question, result,
                                   {"filename": file.name if file else "unknown"})

        # CRITICAL: Final yield with complete result
        yield "⏱️ Complete ✅", result


def process_audio_for_file_qa(audio_filepath, file):
    """Process audio input for file Q&A"""
    guest_check = check_guest_feature_access("File Q&A")
    if guest_check:
        yield "⏱️ 0s", guest_check
        return

    if audio_filepath is None:
        yield "⏱️ 0s", ""
        return

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        yield "⏱️ 0s", transcribed_text
        return

    for timer, result in file_question_answer(file, transcribed_text):
        yield timer, result


# --- Image Generation Functions ---
def generate_image_for_gradio(prompt: str):
    """ENHANCED: Image generation with multiple FREE fallbacks"""
    guest_check = check_guest_feature_access("Image Generation")
    if guest_check:
        return None, guest_check

    access_check = check_feature_access("image_gen")
    if access_check:
        return None, access_check

    limit_msg = check_rate_limit("image_gen")
    if limit_msg:
        return None, f"{limit_msg}\n\n**FREE FALLBACK:** Use an external generator like Bing Image Creator."

    if not prompt:
        return None, "Please enter a prompt."

    increment_usage("image_gen")

    # ===== METHOD 1: Pollinations.ai (Primary FREE service) =====
    logging.info("🎨 Trying Pollinations.ai (FREE)...")
    try:
        encoded_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true&enhance=true"
        response = requests.get(url, timeout=45)
        if response.status_code == 200 and len(response.content) > 5000:
            try:
                image = Image.open(io.BytesIO(response.content))
                status_msg = "✅ Image generated successfully with Pollinations.ai (FREE)!"
                logging.info("✅ Pollinations.ai success")

                if current_user["is_guest"]:
                    add_to_guest_history("image_gen", prompt, status_msg, {"service": "Pollinations.ai"})
                else:
                    save_interaction_to_db("image_gen", prompt, status_msg,
                                           {"service": "Pollinations.ai", "size": f"{len(response.content)} bytes"})

                return image, status_msg
            except Exception as e:
                logging.error(f"Pollinations.ai image processing error: {e}")
        else:
            logging.warning(f"Pollinations.ai failed: status={response.status_code}, size={len(response.content)}")
    except Exception as e:
        logging.error(f"Pollinations.ai error: {e}")

    # ===== METHOD 2: Hugging Face Inference API (FREE alternative) =====
    if HF_API_KEY:
        logging.info("🎨 Trying Hugging Face Flux (FREE)...")
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            response = requests.post(hf_url, headers=headers, json={"inputs": prompt}, timeout=60)

            if response.status_code == 200 and len(response.content) > 5000:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    status_msg = "✅ Image generated successfully with Hugging Face Flux (FREE)!"
                    logging.info("✅ Hugging Face success")

                    if current_user["is_guest"]:
                        add_to_guest_history("image_gen", prompt, status_msg, {"service": "Hugging Face"})
                    else:
                        save_interaction_to_db("image_gen", prompt, status_msg,
                                               {"service": "Hugging Face", "size": f"{len(response.content)} bytes"})

                    return image, status_msg
                except Exception as e:
                    logging.error(f"HF image processing error: {e}")
            elif response.status_code == 503:
                logging.warning("⏳ Hugging Face model loading... (might work in 20-30 seconds)")
            else:
                logging.warning(f"Hugging Face failed: status={response.status_code}")
        except Exception as e:
            logging.error(f"Hugging Face error: {e}")

    # ===== METHOD 3: Segmind (FREE tier) =====
    logging.info("🎨 Trying Segmind (FREE)...")
    try:
        url = "https://api.segmind.com/v1/sd1.5-txt2img"
        data = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "samples": 1,
            "width": 768,
            "height": 768,
            "steps": 20,
            "seed": 0
        }
        response = requests.post(url, json=data, timeout=45)
        if response.status_code == 200 and len(response.content) > 5000:
            try:
                image = Image.open(io.BytesIO(response.content))
                status_msg = "✅ Image generated successfully with Segmind (FREE)!"
                logging.info("✅ Segmind success")

                if current_user["is_guest"]:
                    add_to_guest_history("image_gen", prompt, status_msg, {"service": "Segmind"})
                else:
                    save_interaction_to_db("image_gen", prompt, status_msg,
                                           {"service": "Segmind", "size": f"{len(response.content)} bytes"})

                return image, status_msg
            except Exception as e:
                logging.error(f"Segmind image processing error: {e}")
        else:
            logging.warning(f"Segmind failed: status={response.status_code}")
    except Exception as e:
        logging.error(f"Segmind error: {e}")

    # ===== METHOD 4: Stability AI (Premium - if API key available) =====
    if STABILITY_API_KEY:
        logging.info("🎨 Trying Stability AI (Premium)...")
        try:
            session = create_session_with_retries()
            headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}
            data = {"prompt": prompt, "output_format": "png", "aspect_ratio": "1:1", "seed": 0}
            response = session.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers=headers, files={"none": ''}, data=data, timeout=60
            )
            if response.status_code == 200:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    status_msg = "✅ Image generated successfully with Stability AI (Premium)!"
                    logging.info("✅ Stability AI success")

                    if current_user["is_guest"]:
                        add_to_guest_history("image_gen", prompt, status_msg, {"service": "Stability AI"})
                    else:
                        save_interaction_to_db("image_gen", prompt, status_msg,
                                               {"service": "Stability AI", "size": f"{len(response.content)} bytes"})

                    return image, status_msg
                except Exception as e:
                    logging.error(f"Stability AI image processing error: {e}")
            else:
                logging.warning(f"Stability AI failed: status={response.status_code}")
        except Exception as e:
            logging.error(f"Stability AI error: {e}")

    # ===== ALL SERVICES FAILED =====
    error_msg = (
        "❌ **All image generation services are currently unavailable.**\n\n"
        "**What happened:**\n"
        "- 🚫 Pollinations.ai - Failed or rate limited\n"
        "- 🚫 Hugging Face - Model unavailable or loading\n"
        "- 🚫 Segmind - Service unavailable\n"
        f"- 🚫 Stability AI - {'Not configured' if not STABILITY_API_KEY else 'Failed'}\n\n"
        "**FREE Alternatives (100% Working):**\n\n"
        "1. **Bing Image Creator** (Microsoft, FREE):\n"
        "   - 🔗 https://www.bing.com/images/create\n"
        "   - Uses DALL-E 3, very high quality\n"
        "   - No sign-up needed\n\n"
        "2. **Leonardo.ai** (FREE tier):\n"
        "   - 🔗 https://app.leonardo.ai\n"
        "   - 150 FREE credits daily\n\n"
        "3. **Craiyon** (FREE, no limits):\n"
        "   - 🔗 https://www.craiyon.com\n"
        "   - Instant generation\n\n"
        "4. **Ideogram** (FREE):\n"
        "   - 🔗 https://ideogram.ai\n"
        "   - Great for text in images\n\n"
        "**Try again in 10-15 minutes** - services may recover."
    )

    logging.error("❌ All image generation services failed")

    if not current_user["is_guest"]:
        save_interaction_to_db("image_gen", prompt, "Failed - all services unavailable", {"error": "all_failed"})

    return None, error_msg


def process_audio_for_image_gen(audio_filepath):
    """Process audio input for image generation"""
    guest_check = check_guest_feature_access("Image Generation")
    if guest_check:
        return None, guest_check

    if audio_filepath is None:
        return None, ""

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return None, transcribed_text

    return generate_image_for_gradio(transcribed_text)


# --- Image QA ---
def query_image_model(image, prompt):
    """
    ULTRA ROBUST: Image Q&A with ALL available vision models
    Priority: Groq Vision (FREE, high limits) → Gemini (3 keys) → OpenAI GPT-4
    """
    guest_check = check_guest_feature_access("Image Q&A")
    if guest_check:
        return guest_check

    access_check = check_feature_access("image_qa")
    if access_check:
        return access_check

    limit_msg = check_rate_limit("image_qa")
    if limit_msg:
        return limit_msg

    if image is None:
        return "Error: Please upload an image first."

    increment_usage("image_qa")

    # Prepare image for processing
    try:
        # Convert to RGB if needed (fixes issues with PNG/RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        resized_image = image.copy()
        resized_image.thumbnail((1024, 1024), Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
    except Exception as e:
        return f"Error processing image: {e}"

    # ============================================================================
    # METHOD 1: GROQ LLAMA VISION (PRIMARY - FREE, High Limits)
    # ============================================================================
    if GROQ_API_KEY:
        try:
            logging.info("🚀 Trying Groq Llama Vision (FREE, high limits)...")

            import base64
            img_byte_arr.seek(0)
            base64_image = base64.b64encode(img_byte_arr.read()).decode('utf-8')

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # Updated to Llama 4 Scout (current)
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 1
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()

                if current_user["is_guest"]:
                    add_to_guest_history("image_qa", prompt, result, {"model": "Groq Llama Vision"})
                else:
                    save_interaction_to_db("image_qa", prompt, result, {"model": "Groq Llama Vision"})

                logging.info("✅ Image Q&A success with Groq Llama Vision")
                return result
            else:
                error_detail = response.text
                logging.warning(f"⚠️ Groq failed with status {response.status_code}: {error_detail}")
                logging.warning("Trying Gemini...")

        except Exception as e:
            logging.error(f"Groq Vision error: {e}")
            logging.info("🔄 Falling back to Gemini...")

def process_audio_for_image_qa(audio_filepath, image):
    """Process audio input for image Q&A"""
    guest_check = check_guest_feature_access("Image Q&A")
    if guest_check:
        return guest_check

    if audio_filepath is None:
        return ""

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return transcribed_text

    return query_image_model(image, transcribed_text)

# --- Video Generation ---
def install_package(package):
    """Install a package if not already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except Exception as e:
            logging.error(f"Failed to install {package}: {e}")
            return False


def generate_video_huggingface_svd_fixed(prompt: str):
    """
    FIXED: Generate actual video using Hugging Face
    Uses the correct working model endpoint
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    try:
        logging.info("🎬 Generating real video with Hugging Face (FREE)...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        # Step 1: Generate initial frame
        logging.info("🎨 Step 1/2: Generating initial frame...")
        img_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        img_response = requests.post(
            img_url,
            headers=headers,
            json={"inputs": prompt},
            timeout=120
        )

        if img_response.status_code != 200:
            logging.warning(f"Initial image generation failed: {img_response.status_code}")
            return None, f"Failed to generate starting frame: {img_response.status_code}"

        initial_image = Image.open(io.BytesIO(img_response.content))
        initial_image = initial_image.resize((1024, 576), Resampling.LANCZOS)

        # Save as bytes for video generation
        img_bytes = io.BytesIO()
        initial_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        logging.info("✅ Initial frame generated. Now creating video with motion...")

        # Step 2: Try multiple working video generation models
        video_models = [
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "ali-vilab/i2vgen-xl",
            "damo-vilab/text-to-video-ms-1.7b"
        ]

        for model in video_models:
            try:
                logging.info(f"🎞️ Trying {model}...")
                video_url = f"https://api-inference.huggingface.co/models/{model}"

                video_response = requests.post(
                    video_url,
                    headers=headers,
                    data=img_bytes.getvalue(),
                    timeout=180
                )

                if video_response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_response.content)
                        video_path = tmp.name

                    file_size = Path(video_path).stat().st_size / 1024
                    logging.info(f"✅ Real video generated with {model}! Size: {file_size:.1f} KB")
                    return video_path, f"✅ Real video with motion generated! (Size: {file_size:.1f} KB)"

                elif video_response.status_code == 503:
                    logging.warning(f"{model} is loading (503)...")
                    continue
                else:
                    logging.warning(f"{model} failed: {video_response.status_code}")
                    continue

            except Exception as e:
                logging.warning(f"{model} error: {e}")
                continue

        # All HF models failed
        return None, "⏳ Hugging Face video models are loading. Try again in 20-30 seconds."

    except Exception as e:
        logging.error(f"Hugging Face video generation failed: {e}")
        return None, f"❌ Video generation failed: {str(e)}"


def generate_video_animatediff_hf(prompt: str):
    """
    Alternative: Generate video using AnimateDiff on Hugging Face (FREE)
    This is a direct text-to-video model
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    try:
        logging.info("🎬 Generating video with AnimateDiff (Hugging Face)...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        # Direct text-to-video generation
        video_url = "https://api-inference.huggingface.co/models/guoyww/animatediff-motion-adapter-v1-5-2"

        response = requests.post(
            video_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "num_frames": 16,
                    "num_inference_steps": 25
                }
            },
            timeout=180
        )

        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(response.content)
                video_path = tmp.name

            file_size = Path(video_path).stat().st_size / 1024
            return video_path, f"✅ Video generated with AnimateDiff! (Size: {file_size:.1f} KB)"

        elif response.status_code == 503:
            return None, "⏳ AnimateDiff model is loading (20-30s). Please try again."
        else:
            logging.error(f"AnimateDiff failed: {response.status_code}")
            return None, f"AnimateDiff failed: {response.status_code}"

    except Exception as e:
        logging.error(f"AnimateDiff HF failed: {e}")
        return None, f"❌ AnimateDiff failed: {str(e)}"


def generate_video_text2video_zero(prompt: str):
    """
    Generate video using Text2Video-Zero (100% FREE, works well)
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    try:
        logging.info("🎬 Generating video with Text2Video-Zero...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}

        video_url = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"

        response = requests.post(
            video_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "num_frames": 16
                }
            },
            timeout=180
        )

        if response.status_code == 200:
            # Response might be a video file or JSON with URL
            content_type = response.headers.get('content-type', '')

            if 'video' in content_type or len(response.content) > 100000:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name

                file_size = Path(video_path).stat().st_size / 1024
                return video_path, f"✅ Video generated with Text2Video! (Size: {file_size:.1f} KB)"
            else:
                # Try to parse JSON response
                try:
                    result = response.json()
                    video_url = result.get('video_url') or result.get('url')
                    if video_url:
                        vid_response = requests.get(video_url, timeout=60)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(vid_response.content)
                            video_path = tmp.name
                        return video_path, "✅ Video generated with Text2Video!"
                except:
                    pass

                return None, "Unexpected response format from Text2Video"

        elif response.status_code == 503:
            return None, "⏳ Text2Video model is loading. Try again in 20-30s."
        else:
            return None, f"Text2Video failed: {response.status_code}"

    except Exception as e:
        logging.error(f"Text2Video failed: {e}")
        return None, f"❌ Text2Video failed: {str(e)}"


def generate_video_replicate_zeroscope(prompt: str):
    """
    Generate video using Replicate's Zeroscope model
    Requires REPLICATE_API_TOKEN and billing credits
    """
    if not REPLICATE_AVAILABLE or not replicate:
        return None, "Replicate not configured or library not installed."

    try:
        logging.info("🎬 Generating video with Replicate Zeroscope...")

        output = replicate.run(
            "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
            input={
                "prompt": prompt,
                "num_frames": 24,
                "num_inference_steps": 50,
            }
        )

        # Download the video
        if output:
            video_url = output if isinstance(output, str) else output[0]
            response = requests.get(video_url, timeout=60)

            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name

                file_size = Path(video_path).stat().st_size / 1024
                return video_path, f"✅ Video generated with Zeroscope! (Size: {file_size:.1f} KB)"

        return None, "Failed to retrieve video from Replicate"

    except Exception as e:
        logging.error(f"Replicate Zeroscope failed: {e}")
        return None, f"❌ Replicate failed: {str(e)}"


def generate_simple_video_locally_enhanced(prompt: str, duration=3, fps=10):
    """
    LOCAL FALLBACK: Generate a simple animated video using images
    This creates a zoom/pan effect on a generated or placeholder image
    """
    if not IMAGEIO_AVAILABLE:
        return None, (
            "❌ Local video generation unavailable.\n\n"
            "**Install required packages:**\n"
            "```bash\n"
            "pip install imageio imageio-ffmpeg numpy\n"
            "```"
        )

    try:
        logging.info("🎨 Creating local animated video...")

        # Try to generate an image first
        base_image = None

        # Method 1: Try Pollinations
        try:
            encoded_prompt = quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 5000:
                base_image = Image.open(io.BytesIO(response.content))
                logging.info("✅ Got base image from Pollinations")
        except Exception as e:
            logging.warning(f"Pollinations failed for video base: {e}")

        # Method 2: Create a placeholder if image generation fails
        if base_image is None:
            logging.info("Creating placeholder image for animation...")
            base_image = Image.new('RGB', (1024, 576), color=(30, 30, 50))
            draw = ImageDraw.Draw(base_image)

            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()

            # Wrap text
            words = prompt.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

            # Draw text centered
            y_offset = 200
            for line in lines[:4]:  # Max 4 lines
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (1024 - text_width) // 2
                draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += 60

        # Create animation frames with zoom effect
        frames = []
        total_frames = duration * fps

        for i in range(total_frames):
            # Calculate zoom factor (1.0 to 1.3)
            zoom = 1.0 + (i / total_frames) * 0.3

            # Calculate new size
            new_width = int(1024 * zoom)
            new_height = int(576 * zoom)

            # Resize image
            zoomed = base_image.resize((new_width, new_height), Resampling.LANCZOS)

            # Crop to original size (centered)
            left = (new_width - 1024) // 2
            top = (new_height - 576) // 2
            cropped = zoomed.crop((left, top, left + 1024, top + 576))

            # Convert to numpy array
            frames.append(np.array(cropped))

        # Save as video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_path = tmp.name

        imageio.mimsave(video_path, frames, fps=fps, codec='libx264', quality=8)

        file_size = Path(video_path).stat().st_size / 1024

        return video_path, (
            f"✅ Local animated video created! (Size: {file_size:.1f} KB)\n\n"
            f"⚠️ This is a fallback animation since cloud services are unavailable.\n\n"
            f"**For better results, try:**\n"
            f"- Waiting 20-30 seconds (HF models need warm-up)\n"
            f"- Using external services: Pika Labs, RunwayML Gen-2"
        )

    except Exception as e:
        logging.error(f"Local video generation failed: {e}")
        return None, f"❌ Local video generation failed: {str(e)}"


# ============================================================================
# UPDATED generate_video() - Replace your current one with this
# ============================================================================
def generate_video(prompt: str):
    """
    MAIN VIDEO GENERATION with working FREE services
    """
    access_check = check_feature_access("video_gen")
    if access_check:
        yield access_check, None
        return

    limit_msg = check_rate_limit("video_gen")
    if limit_msg:
        yield limit_msg, None
        return

    if not prompt or not prompt.strip():
        yield "❌ Please enter a video prompt.", None
        return

    increment_usage("video_gen")

    # Method 1: Text2Video-Zero (Hugging Face - FREE & Works!)
    if HF_API_KEY:
        yield "⏳ Generating real video with Text2Video-Zero (FREE)...", None
        video_path, status = generate_video_text2video_zero(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "Text2Video-Zero"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "Text2Video-Zero"})
            yield status, video_path
            return
        else:
            yield f"⚠️ Text2Video: {status}. Trying AnimateDiff...", None

        # Method 2: AnimateDiff (Hugging Face - FREE)
        yield "⏳ Trying AnimateDiff (FREE)...", None
        video_path, status = generate_video_animatediff_hf(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "AnimateDiff HF"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "AnimateDiff HF"})
            yield status, video_path
            return
        else:
            yield f"⚠️ AnimateDiff: {status}. Trying Stable Video Diffusion...", None

        # Method 3: Stable Video Diffusion (Hugging Face - FREE)
        yield "⏳ Trying Stable Video Diffusion (FREE)...", None
        video_path, status = generate_video_huggingface_svd_fixed(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "Stable Video Diffusion"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "Stable Video Diffusion"})
            yield status, video_path
            return
        else:
            yield f"⚠️ SVD: {status}. Falling back to local animation...", None

    # Method 4: Replicate (if configured with credits)
    if REPLICATE_AVAILABLE:
        yield "⏳ Trying Replicate services...", None
        video_path, status = generate_video_replicate_zeroscope(prompt)
        if video_path:
            if current_user["is_guest"]:
                add_to_guest_history("video_gen", prompt, status, {"service": "Zeroscope"})
            else:
                save_interaction_to_db("video_gen", prompt, status, {"service": "Zeroscope"})
            yield status, video_path
            return

    # Method 5: Local fallback (animated image)
    yield "⏳ Creating animated video (local fallback)...", None
    video_path, status = generate_simple_video_locally_enhanced(prompt, duration=3, fps=10)

    if video_path:
        if current_user["is_guest"]:
            add_to_guest_history("video_gen", prompt, status, {"service": "Local Animation"})
        else:
            save_interaction_to_db("video_gen", prompt, status, {"service": "Local Animation"})
        yield status, video_path
        return

    # All methods failed
    error_msg = (
        "❌ **All video generation services temporarily unavailable.**\n\n"
        "**What happened:**\n"
        "- Hugging Face models are loading (cold start)\n"
        "- Replicate needs billing credit\n\n"
        "**Solution:**\n"
        "🔄 **Wait 20-30 seconds** and try again - HF models will be ready!\n\n"
        "**Or use these FREE online services:**\n"
        "- Pika Labs: https://pika.art (best quality, instant)\n"
        "- RunwayML Gen-2: https://app.runwayml.com (free tier)\n"
        "- Stable Video: https://huggingface.co/spaces/stabilityai/stable-video-diffusion"
    )

    if not current_user["is_guest"]:
        save_interaction_to_db("video_gen", prompt, "Failed - models loading", {"error": "cold_start"})

    yield error_msg, None


# Wrapper function for Gradio (handles generator properly)
def video_gen_wrapper(prompt):
    """
    FIXED: Proper generator handling for video generation
    """
    final_status = ""
    final_video = None

    try:
        # Iterate through generator
        for status, video in generate_video(prompt):
            final_status = status
            final_video = video
            yield status, video  # Stream updates to UI

        # CRITICAL: Ensure final values are yielded
        if final_status or final_video:
            yield final_status, final_video
        else:
            # Fallback if generator produced nothing
            yield "❌ Video generation failed - no output", None

    except Exception as e:
        logging.error(f"Video generation error in wrapper: {e}")
        yield f"❌ Error: {str(e)}", None


# Audio processing for video generation
def process_audio_for_video_gen(audio_filepath):
    """
    Process audio input for video generation
    """
    guest_check = check_guest_feature_access("Video Generation")
    if guest_check:
        yield guest_check, None
        return

    if audio_filepath is None:
        yield "", None
        return

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        yield transcribed_text, None
        return

    # Use the generator
    for status, video in generate_video(transcribed_text):
        yield status, video


# --- Google Image Search ---
def google_image_search(query: str):
    """FIXED: Image search with history saving"""
    access_check = check_feature_access("image_search")
    if access_check:
        return None, access_check

    limit_msg = check_rate_limit("image_search")
    if limit_msg:
        return None, limit_msg

    if build is None:
        return None, "Error: 'googleapiclient' not installed. Please `pip install google-api-python-client`."

    if not query:
        return None, "Please enter a search query."

    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        return None, "Error: Google Search API Key or CX missing."

    increment_usage("image_search")

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        result = service.cse().list(
            q=query,
            cx=GOOGLE_SEARCH_CX,
            searchType='image',
            num=5,
            safe='active'
        ).execute()

        if 'items' not in result or not result['items']:
            status_msg = f"No image results found for: **{query}**"

            if current_user["is_guest"]:
                add_to_guest_history("image_search", query, status_msg)
            else:
                save_interaction_to_db("image_search", query, status_msg)

            return None, status_msg

        for item in result['items']:
            image_url = item['link']
            if any(bad in image_url for bad in ["instagram.com", "facebook.com", "pinterest.com", ".svg", ".gif"]):
                logging.info(f"🚫 Skipping blocked domain/format: {image_url}")
                continue

            try:
                session = create_session_with_retries()
                resp = session.get(image_url, stream=True, timeout=15)
                resp.raise_for_status()

                ctype = resp.headers.get("Content-Type", "").lower()
                if not ctype.startswith("image/"):
                    logging.warning(f"⚠️ Non-image content-type received: {ctype}")
                    continue

                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                image.thumbnail((512, 512), Resampling.LANCZOS)

                status_msg = f"✅ Found and resized image for: **{query}**"

                if current_user["is_guest"]:
                    add_to_guest_history("image_search", query, status_msg, {"url": image_url})
                else:
                    save_interaction_to_db("image_search", query, status_msg, {"url": image_url})

                return image, status_msg
            except Exception as e:
                logging.warning(f"⚠️ Skipping invalid or unreachable image: {e}")
                continue

        status_msg = f"⚠️ No valid image results for: **{query}** after trying multiple options."

        if current_user["is_guest"]:
            add_to_guest_history("image_search", query, status_msg)
        else:
            save_interaction_to_db("image_search", query, status_msg)

        return None, status_msg
    except Exception as e:
        if "quotaExceeded" in str(e):
            set_rate_limit("image_search")
            error_msg = f"⚠️ Google Image Search: API quota exceeded. Try again after 12 hours."
        else:
            error_msg = f"Image Search failed: {e}"

        if not current_user["is_guest"]:
            save_interaction_to_db("image_search", query, error_msg)

        return None, error_msg


def process_audio_for_image_search(audio_filepath):
    """Process audio input for image search"""
    guest_check = check_guest_feature_access("Image Search")
    if guest_check:
        return None, guest_check

    if audio_filepath is None:
        return None, ""

    transcribed_text = transcribe_audio(audio_filepath)

    if transcribed_text.startswith("❌"):
        return None, transcribed_text

    return google_image_search(transcribed_text)


def handle_pasted_image(pasted_image):
    """Process pasted image"""
    if pasted_image is None:
        return None, "Please paste an image first."

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            pasted_image.save(tmp.name)
            temp_path = tmp.name

        return pasted_image, f"✅ **Image pasted successfully!**\n\n💡 *Tip: Use the search box above to find similar images, or use Image Q&A tab to analyze this image.*"

    except Exception as e:
        return None, f"❌ Failed to process pasted image: {e}"


# ================================
# AUTO-CHECK AUTH STATUS FUNCTION
# ================================
def auto_check_auth_status(counter):
    """Automatically check if user logged in via Firebase (runs every 500ms)"""
    global current_session_id, guest_chat_count

    new_counter = counter + 1

    # Check queue for instant login
    try:
        user_data = firebase_login_queue.get_nowait()
        if user_data:
            username = user_data.get("username", "User")
            logging.info(f"🚀 INSTANT FIREBASE LOGIN DETECTED: {username}")

            # ✅ MUST HAVE 17 OUTPUTS (including maps_tab)
            return (
                new_counter,  # auth_check_counter
                f"✅ **Welcome, {username}!** All features unlocked.",
                gr.update(visible=False),  # auth_section
                gr.update(visible=True),  # main_app
                f"**Logged in as:** {username}",
                gr.update(visible=True),  # file_qa_tab
                gr.update(visible=True),  # image_gen_tab
                gr.update(visible=True),  # image_qa_tab
                gr.update(visible=True),  # image_search_tab
                gr.update(visible=True),  # video_gen_tab
                gr.update(visible=True),  # translation_tab
                gr.update(visible=True),  # maps_tab ← CHECK THIS IS HERE!
                gr.update(visible=True),  # stats_btn
                [],  # history_chatbot
                gr.update(visible=False),  # guest_chat_warning
                [],  # chatbot
                current_session_id,  # session_id
                None  # mic_chat
            )
    except queue.Empty:
        pass

    # Fallback check
    if current_user.get("logged_in") and not current_user.get("is_guest"):
        username = current_user.get("username", "User")

        # ✅ MUST HAVE 17 OUTPUTS (including maps_tab)
        return (
            new_counter,
            f"✅ **Welcome, {username}!** All features unlocked.",
            gr.update(visible=False),  # auth_section
            gr.update(visible=True),  # main_app
            f"**Logged in as:** {username}",
            gr.update(visible=True),  # file_qa_tab
            gr.update(visible=True),  # image_gen_tab
            gr.update(visible=True),  # image_qa_tab
            gr.update(visible=True),  # image_search_tab
            gr.update(visible=True),  # video_gen_tab
            gr.update(visible=True),  # translation_tab
            gr.update(visible=True),  # maps_tab ← CHECK THIS IS HERE!
            gr.update(visible=True),  # stats_btn
            [],  # history_chatbot
            gr.update(visible=False),  # guest_chat_warning
            [],  # chatbot
            current_session_id,
            None  # mic_chat
        )
    else:
        # Guest mode - no changes (17 gr.update() calls)
        return (
            new_counter,
            gr.update(),  # login_status
            gr.update(),  # auth_section
            gr.update(),  # main_app
            gr.update(),  # user_info
            gr.update(),  # file_qa_tab
            gr.update(),  # image_gen_tab
            gr.update(),  # image_qa_tab
            gr.update(),  # image_search_tab
            gr.update(),  # video_gen_tab
            gr.update(),  # translation_tab
            gr.update(),  # maps_tab ← CHECK THIS IS HERE!
            gr.update(),  # stats_btn
            gr.update(),  # history_chatbot
            gr.update(),  # guest_chat_warning
            gr.update(),  # chatbot
            gr.update(),  # session_id
            gr.update()  # mic_chat
        )

# ================================
# FLASK ROUTES - SINGLE UNIFIED BLOCK
# ================================

@flask_app.route("/test", methods=["GET"])
def test_route():
    """Test if Flask is working"""
    return jsonify({
        "message": "✅ Flask is working!",
        "base_url": BASE_URL,
        "routes": [str(rule) for rule in flask_app.url_map.iter_rules()]
    })


@flask_app.route("/test-firebase", methods=["GET"])
def test_firebase():
    """Test Firebase configuration"""
    try:
        app = firebase_admin.get_app()
        return jsonify({
            "status": "✅ Firebase initialized",
            "app_name": app.name,
            "project_id": app.project_id if hasattr(app, 'project_id') else "N/A"
        }), 200
    except ValueError:
        return jsonify({
            "status": "❌ Firebase NOT initialized",
            "error": "No Firebase app found"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "❌ Error",
            "error": str(e)
        }), 500


@flask_app.route("/api/check_auth", methods=["GET"])
def api_check_auth():
    """Check authentication status - KEEP ONLY THIS ONE"""
    return jsonify({
        "logged_in": current_user.get("logged_in", False),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "is_guest": current_user.get("is_guest", True),
        "full_name": current_user.get("full_name", "")
    })


@flask_app.route("/check-auth", methods=["GET"])
def check_auth_status():
    """Legacy auth check endpoint - KEEP ONLY THIS ONE"""
    return jsonify({
        "logged_in": current_user.get("logged_in", False),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "is_guest": current_user.get("is_guest", True),
        "full_name": current_user.get("full_name", "")
    })


@flask_app.route("/firebase-auth", methods=["GET", "OPTIONS"])
def firebase_auth_page():
    """Serve Firebase authentication page with proper config injection"""

    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = flask_app.make_default_options_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response

    logging.info("🔥 Firebase auth page requested")
    logging.info(f"   Request from: {request.host}")
    logging.info(f"   BASE_URL: {BASE_URL}")
    logging.info(f"   API_BASE_URL: {API_BASE_URL}")

    # ✅ Get Firebase config
    config_str = json.dumps(firebase_config)

    # ✅ Determine correct API base URL for this request
    if IS_HUGGINGFACE:
        api_url = f"https://{request.host}".replace(':7860', ':5000')
    else:
        api_url = "http://localhost:5000"

    logging.info(f"   Using API URL: {api_url}")

    # ✅ RETURN THE ACTUAL HTML PAGE
    return f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>All Mind - Sign In</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 400px;
            width: 100%;
            text-align: center;
        }}
        .logo {{
            font-size: 48px;
            margin-bottom: 10px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .google-btn {{
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 14px 24px;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            color: #333;
            transition: all 0.2s;
        }}
        .google-btn:hover {{
            background: #f8f9fa;
            border-color: #4285f4;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.2);
        }}
        .status {{
            margin-top: 20px;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            display: none;
        }}
        .status.show {{
            display: block;
        }}
        .status.loading {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        .status.success {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .status.error {{
            background: #ffebee;
            color: #c62828;
        }}
        .spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4285f4;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 8px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🤖</div>
        <h1>Welcome to All Mind</h1>
        <p class="subtitle">Sign in to access all features</p>

        <button class="google-btn" onclick="signInWithGoogle()">
            <svg width="20" height="20" viewBox="0 0 48 48">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            </svg>
            Continue with Google
        </button>

        <div class="status" id="status"></div>
    </div>

    <script type="module">
        import {{ initializeApp }} from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
        import {{ getAuth, signInWithPopup, GoogleAuthProvider }} from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js';

        const firebaseConfig = {config_str};

        console.log('🔥 Initializing Firebase...');
        const app = initializeApp(firebaseConfig);
        const auth = getAuth(app);
        const provider = new GoogleAuthProvider();

        function showStatus(message, type) {{
            const status = document.getElementById('status');
            status.className = 'status show ' + type;
            status.innerHTML = type === 'loading' ? 
                '<span class="spinner"></span>' + message : message;
        }}

        window.signInWithGoogle = async function() {{
            try {{
                showStatus('Opening Google Sign-In...', 'loading');

                const result = await signInWithPopup(auth, provider);
                const user = result.user;

                console.log('✅ Google Sign-In successful:', user.email);
                showStatus('Signed in! Getting token...', 'loading');

                const idToken = await user.getIdToken();
                console.log('✅ Got Firebase ID token');

                showStatus('Logging you in...', 'loading');

                const response = await fetch('{api_url}/firebase-login', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ token: idToken }})
                }});

                const data = await response.json();

                if (data.success) {{
    console.log('✅ Backend login successful');
    showStatus('✅ Success! Redirecting...', 'success');

    // ✅ CRITICAL FIX: Signal Gradio about login
    localStorage.setItem('firebase_login_event', JSON.stringify({{
        timestamp: Date.now(),
        user: data.user
    }}));

    // ✅ FIX: Force parent window reload if opened in popup
    if (window.opener) {{
        window.opener.postMessage({{
            type: 'FIREBASE_LOGIN_SUCCESS',
            user: data.user
        }}, '*');
        
        setTimeout(() => {{
            window.close();
            if (!window.closed) {{
                // If popup didn't close, redirect
                window.location.href = '{BASE_URL}';
            }}
        }}, 500);
    }} else {{
        // ✅ FIX: If not a popup, force immediate reload
        setTimeout(() => {{
            window.location.replace('{BASE_URL}');
        }}, 500);
    }}
}} else {{
    throw new Error(data.error || 'Login failed');
}}

}} catch (error) {{
    console.error('❌ Sign-in error:', error);
    showStatus('❌ Error: ' + error.message, 'error');
}}
}};
        console.log('✅ Firebase auth page ready');
    </script>
</body>
</html>
'''

@flask_app.route("/firebase-login", methods=["POST", "OPTIONS"])
def firebase_login_endpoint():
    """Handle Firebase authentication"""

    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200

    logging.info("🔥 Firebase login endpoint called")

    try:
        data = request.get_json(force=True)
        id_token = data.get("token") if data else None

        if not id_token:
            response = jsonify({"success": False, "error": "No token provided"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400

        user_info = verify_firebase_token(id_token)

        if not user_info:
            response = jsonify({"success": False, "error": "Invalid token"})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 401

        success, message = register_or_login_firebase_user(user_info)

        if success:
            notify_gradio_login({
                "username": current_user["username"],
                "email": current_user["email"],
                "full_name": current_user.get("full_name", "")
            })

            response = jsonify({
                "success": True,
                "message": message,
                "user": {
                    "username": current_user["username"],
                    "email": current_user["email"],
                    "full_name": current_user.get("full_name", "")
                }
            })
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 200
        else:
            response = jsonify({"success": False, "error": message})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 500

    except Exception as e:
        logging.error(f"Firebase login error: {e}")
        logging.error(traceback.format_exc())
        response = jsonify({"success": False, "error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500


# ============================================================================
# ✅✅✅ ADD THIS NEW ENDPOINT HERE ✅✅✅
# ============================================================================

@flask_app.route("/api/reverse-geocode", methods=["POST"])
def reverse_geocode():
    """Backend proxy for reverse geocoding (avoids CORS issues)"""
    try:
        data = request.get_json()
        lat = data.get("lat")
        lon = data.get("lon")

        if not lat or not lon:
            return jsonify({"success": False, "error": "Missing coordinates"}), 400

        # Use backend to fetch from Nominatim (no CORS issues here)
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"

        response = requests.get(
            url,
            headers={"User-Agent": "AllMindApp/1.0"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            address = data.get("display_name", f"{lat}, {lon}")

            return jsonify({
                "success": True,
                "address": address,
                "full_data": data
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Nominatim error: {response.status_code}"
            }), response.status_code

    except Exception as e:
        logging.error(f"Reverse geocoding error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ------------------ GRADIO UI ------------------
with gr.Blocks(title="All Mind") as demo:
    gr.Markdown("# 🤖 All Mind")

    # Login/Register Section
    with gr.Group(visible=False) as auth_section:
        gr.Markdown("## 🔐 Welcome! Please Login or Register")
        guest_status = gr.Markdown("", visible=False)

        with gr.Tab("Login"):
            if FIREBASE_AVAILABLE:
                gr.Markdown("#### 🚀 Quick Login with Google (Recommended)")
                gr.HTML(login_html)
                gr.Markdown("---\n**OR**\n---")
                gr.Markdown("### Traditional Email/Username Login")
            else:
                gr.Markdown("⚠️ **Google Sign-In temporarily unavailable** - Use email login below")
                gr.Markdown("### Login with Email/Username")

            with gr.Row():
                login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
            login_btn = gr.Button("🔑 Login", variant="primary", size="lg")
            login_status = gr.Markdown()

            gr.Markdown("""
            **Don't have an account?**
            - Go to the **Register** tab to create a FREE account
            - Or click the **Continue with Google** button above
            """)

        with gr.Tab("Register"):
            gr.Markdown("### 🎉 Create Your Free Account!")

            if FIREBASE_AVAILABLE:
                gr.Markdown("#### 🚀 Quick Sign Up with Google (Recommended)")
                gr.HTML(register_html)
                gr.Markdown("---\n**OR**\n---")
                gr.Markdown("### Traditional Email Registration")
            else:
                gr.Markdown("⚠️ **Google Sign-In temporarily unavailable** - Use email registration below")
                gr.Markdown("### Register with Email")

            gr.Markdown("""
        **What you get with registration:**
        - ♾️ **Unlimited Chat** - No message limits
        - 📄 **File Q&A** - Upload and analyze any document
        - 🎨 **Image Generation** - Create AI art from text
        - 🎥 **Video Generation** - Generate AI videos
        - 🖼️ **Image Search** - Search Google Images  
        - 🔍 **Image Q&A** - Ask questions about images
        - 📊 **Usage Statistics** - Track your activity
        - 💾 **Persistent History** - All your conversations saved permanently
            """)
            with gr.Row():
                reg_username = gr.Textbox(label="Username *", placeholder="Choose a username (min 3 characters)")
                reg_password = gr.Textbox(label="Password *", type="password",
                                          placeholder="Choose a password (min 6 characters)")
            with gr.Row():
                reg_email = gr.Textbox(label="Email Address *",
                                       placeholder="your.email@example.com (Required for OTP verification)")
                reg_fullname = gr.Textbox(label="Full Name", placeholder="Your full name (optional)")

            request_otp_btn = gr.Button("📧 Send OTP to My Email", variant="primary", size="lg")

            otp_input = gr.Textbox(label="Enter 6-Digit OTP", placeholder="123456", max_lines=1, visible=False)
            verify_otp_btn = gr.Button("✅ Verify OTP & Complete Registration", variant="primary", size="lg",
                                       visible=False)

            register_status = gr.Markdown()

    # Main App Section
    with gr.Group(visible=True) as main_app:
        with gr.Row():
            user_info = gr.Markdown(f"👤 **Guest Mode** | 💬 {guest_chat_count}/{GUEST_CHAT_LIMIT} chats used")
            logout_btn = gr.Button("🚪 Logout", size="sm")
            stats_btn = gr.Button("📊 My Statistics", size="sm", visible=False)

        with gr.Group(visible=False) as stats_modal:
            stats_content = gr.Markdown("Loading statistics...")
            close_stats_btn = gr.Button("✖️ Close", variant="secondary")

        gr.Markdown(
            "### 🎤 **Voice Input Available on All Tabs!** Click the microphone icon to speak instead of typing.")

        with gr.Row():
            chat_timer = gr.Textbox(label="Chat Timer", interactive=False, value=get_timer_text("text_qa", "Chat"))
            file_qa_timer = gr.Textbox(label="File Q&A Timer", interactive=False,
                                       value=get_timer_text("file_qa", "File Q&A"))
            image_timer = gr.Textbox(label="Image Timer", interactive=False,
                                     value=get_timer_text("image_gen", "Image Gen"))
            video_timer = gr.Textbox(label="Video Timer", interactive=False,
                                     value=get_timer_text("video_gen", "Video Gen"))
            image_search_timer = gr.Textbox(label="Image Search Timer", interactive=False,
                                            value=get_timer_text("image_search", "Image Search"))
            ip_timer = gr.Textbox(label="Public IP Timer", interactive=False,
                                  value=get_timer_text("public_ip", "Public IP"))

        with gr.Row():
            show_history_btn = gr.Button("📋 Show My History", variant="primary", size="lg")

        session_id = gr.State(0)
        auth_check_counter = gr.State(0)

        with gr.Group(visible=False) as history_modal:
            with gr.Row():
                gr.Markdown("## 📚 Your Activity History")

            history_chatbot = gr.Chatbot(
                label="Activity Log (Your Data Only - Isolated)",
                height=500,
            )

            with gr.Row():
                refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary")
                clear_all_btn = gr.Button("🗑️ Clear My History", variant="stop")
                close_modal_btn = gr.Button("✖️ Close", variant="primary")
            history_status = gr.Textbox(label="Status", visible=True)

        show_history_btn.click(show_history_modal, outputs=[history_modal, history_chatbot])
        refresh_history_btn.click(format_history_for_chatbot, outputs=history_chatbot)
        clear_all_btn.click(clear_all_history_action, outputs=[history_chatbot, history_status])
        close_modal_btn.click(close_history_modal, outputs=history_modal)

        with gr.Tab("💬 Chat with Voice Input"):
            gr.Markdown("### 🎤 Use voice or text to chat with AI")

            guest_chat_warning = gr.Markdown("", visible=True)

            with gr.Row():
                mic_chat = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Click to Record Voice")

            chatbot = gr.Chatbot(label="Conversation", height=500)

            user_input = gr.Textbox(placeholder="Enter your message here... or use voice input above",
                                    label="Type your message")

            target_language_dropdown = gr.Dropdown(
                label="Select target language",
                choices=["fr", "es", "hi", "zh", "de"],
                visible=False
            )

            translation_output_textbox = gr.Textbox(label="Translated Text", interactive=False)
            translation_audio_output = gr.Audio(label="Spoken Output")

            with gr.Row():
                target_language_dropdown
                translation_output_textbox
                translation_audio_output

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                new_chat_btn = gr.Button("🆕 New Chat", variant="secondary")

            send_btn.click(
                query_and_update_warning,
                inputs=[user_input, chatbot, session_id],
                outputs=[chatbot, user_input, guest_chat_warning, session_id]
            )


            def safe_voice_handler(audio_filepath, history, session_id_state):
                """Wrapper that FORCES history validation before processing voice"""
                global current_session_id

                if session_id_state != current_session_id:
                    logging.error(
                        f"🚨 CRITICAL: Voice used stale session! Clearing. Expected: {current_session_id}, Got: {session_id_state}")
                    history = []
                    session_id_state = current_session_id

                return process_audio_and_chat(audio_filepath, history, session_id_state)


            mic_chat.change(
                safe_voice_handler,
                inputs=[mic_chat, chatbot, session_id],
                outputs=[chatbot, user_input, guest_chat_warning, session_id]
            )

            new_chat_btn.click(
                start_new_chat,
                inputs=None,
                outputs=[chatbot, session_id]
            )

        with gr.Tab("📄 File Q&A", visible=False) as file_qa_tab:
            gr.Markdown("### 🎤 Upload a file and ask ANYTHING - extract, analyze, generate, transform!")
            gr.Markdown(
                "**Examples:** Extract data, summarize, generate questions, reformat, translate, create study notes, find errors, etc.")

            with gr.Row():
                mic_file = gr.Audio(label="🎙️ Click to Record Question", sources=["microphone"], type="filepath")

            uploaded_file = gr.File(label="Upload Document or Image (PDF, JPG, PNG, DOCX, TXT, etc.)")
            file_question = gr.Textbox(label="Your Question/Request",
                                       placeholder="e.g., 'Summarize the document in 3 bullet points,' or 'Extract all names and addresses'")

            with gr.Row():
                file_processing_timer = gr.Textbox(label="⏱️ Processing Time", interactive=False, value="⏱️ 0s")

            file_output = gr.Markdown(label="AI Response")

            with gr.Row():
                file_submit_btn = gr.Button("Submit File & Question", variant="primary")

            file_submit_btn.click(
                file_question_answer,
                inputs=[uploaded_file, file_question],
                outputs=[file_processing_timer, file_output]
            )

            mic_file.change(
                process_audio_for_file_qa,
                inputs=[mic_file, uploaded_file],
                outputs=[file_processing_timer, file_output]
            )

        with gr.Tab("🎨 Image Generation", visible=False) as image_gen_tab:
            gr.Markdown("### 🎨 Generate stunning images from text prompts")

            with gr.Row():
                mic_img_gen = gr.Audio(label="🎙️ Click to Record Prompt", sources=["microphone"], type="filepath")

            image_prompt = gr.Textbox(label="Image Prompt",
                                      placeholder="e.g., 'A majestic lion wearing a crown, digital art'")
            image_output = gr.Image(label="Generated Image", type="pil", height=512)
            image_status = gr.Textbox(label="Status", interactive=False)
            image_gen_btn = gr.Button("Generate Image", variant="primary")

            image_gen_btn.click(
                generate_image_for_gradio,
                inputs=image_prompt,
                outputs=[image_output, image_status]
            )

            mic_img_gen.change(
                process_audio_for_image_gen,
                inputs=mic_img_gen,
                outputs=[image_output, image_status]
            )

        with gr.Tab("🔍 Image Q&A", visible=False) as image_qa_tab:
            gr.Markdown("### 🖼️ Ask questions about an uploaded image")

            with gr.Row():
                mic_img_qa = gr.Audio(label="🎙️ Click to Record Question", sources=["microphone"], type="filepath")

            qa_image = gr.Image(label="Upload Image", type="pil", height=300)
            qa_prompt = gr.Textbox(label="Question about the Image",
                                   placeholder="e.g., 'Describe what is happening in this photo,' or 'What is the brand of the product?'")
            qa_output = gr.Markdown(label="AI Analysis")
            qa_btn = gr.Button("Analyze Image", variant="primary")

            qa_btn.click(
                query_image_model,
                inputs=[qa_image, qa_prompt],
                outputs=qa_output
            )

            mic_img_qa.change(
                process_audio_for_image_qa,
                inputs=[mic_img_qa, qa_image],
                outputs=qa_output
            )

            gr.Markdown("#### 📋 Or Paste an Image (Ctrl+V):")
            with gr.Row():
                pasted_image = gr.Image(
                    label="Paste Image Here (Ctrl+V or right-click paste)",
                    type="pil",
                    height=300,
                    sources=["clipboard", "upload"]
                )

            search_output = gr.Image(label="Search Result Image", type="pil", height=512)
            search_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                process_paste_btn = gr.Button("Process Pasted Image", variant="secondary")

            process_paste_btn.click(
                handle_pasted_image,
                inputs=pasted_image,
                outputs=[search_output, search_status]
            )

        with gr.Tab("🖼️ Image Search", visible=False) as image_search_tab:
            gr.Markdown("### 🌐 Search and retrieve a relevant image from Google")
            gr.Markdown("**💡 NEW: Paste images with Ctrl+V!** Use the paste area below to paste images from clipboard.")

            with gr.Row():
                mic_img_search = gr.Audio(label="🎙️ Click to Record Query", sources=["microphone"], type="filepath")

            search_query = gr.Textbox(label="Image Search Query", placeholder="e.g., 'latest Mars rover photo'")

            search_output2 = gr.Image(label="Search Result Image", type="pil", height=512)
            search_status2 = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                search_btn = gr.Button("Search for Image", variant="primary")

            search_btn.click(
                google_image_search,
                inputs=search_query,
                outputs=[search_output2, search_status2]
            )

            mic_img_search.change(
                process_audio_for_image_search,
                inputs=mic_img_search,
                outputs=[search_output2, search_status2]
            )

        with gr.Tab("🎥 Video Generation", visible=False) as video_gen_tab:
            gr.Markdown("### 🎬 Generate AI Videos from Text Prompts")
            gr.Markdown(
                "**Smart Fallback System:** Tries Replicate (premium) first, then falls back to FREE services (Hugging Face + Pollinations)!")
            gr.Markdown(
                "💡 *No billing? No problem!* Free services work automatically if Replicate isn't set up.")

            with gr.Row():
                mic_video_gen = gr.Audio(label="🎙️ Click to Record Prompt", sources=["microphone"], type="filepath")

            video_prompt = gr.Textbox(label="Video Prompt",
                                      placeholder="e.g., 'A futuristic car flying over a neon city at night'")

            video_output = gr.Video(label="Generated Video", height=512, autoplay=False)
            video_status = gr.Textbox(label="Status", interactive=False)
            video_gen_btn = gr.Button("Generate Video", variant="primary")

            video_gen_btn.click(
                video_gen_wrapper,
                inputs=video_prompt,
                outputs=[video_status, video_output]
            )

            mic_video_gen.change(
                process_audio_for_video_gen,
                inputs=mic_video_gen,
                outputs=[video_status, video_output]
            )

        # ✅ CRITICAL FIX: Maps tab moved OUTSIDE video_gen_tab (reduced indentation)
        with gr.Tab("🗺️ Maps & Location", visible=False) as maps_tab:
            gr.Markdown("### 🌍 Google Maps - Search Locations, Get Directions & More")
            gr.Markdown(
                "**Features:**\n"
                "- 📍 Search any location worldwide\n"
                "- 🧭 Get directions between two places\n"
                "- 🏢 Find nearby businesses (restaurants, hotels, etc.)\n"
                "- 🌐 View satellite imagery\n"
                "- 🚗 Real-time traffic information"
            )

            with gr.Row():
                map_mode = gr.Radio(
                    choices=["Search Location", "Get Directions", "Find Nearby"],
                    value="Search Location",
                    label="Select Mode"
                )

            with gr.Row():
                location_input = gr.Textbox(
                    label="📍 Enter Location",
                    placeholder="e.g., 'Eiffel Tower, Paris' or 'Times Square, New York'",
                    visible=True
                )

            with gr.Row():
                origin_input = gr.Textbox(
                    label="📍 From (Origin)",
                    placeholder="e.g., 'Central Park, New York'",
                    visible=False
                )
                destination_input = gr.Textbox(
                    label="🎯 To (Destination)",
                    placeholder="e.g., 'Statue of Liberty, New York'",
                    visible=False
                )

            with gr.Row():
                use_location_btn = gr.Button(
                    "📍 Use My Current Location",
                    variant="secondary",
                    visible=False,
                    size="sm"
                )

            with gr.Row():
                nearby_location = gr.Textbox(
                    label="📍 Detected Location (or enter manually)",
                    placeholder="Click 'Use My Current Location' above or type a location manually",
                    visible=False,
                    interactive=True
                )
                nearby_type = gr.Textbox(
                    label="🔍 Search for (e.g., 'pizza', 'gym', 'school', 'park')",
                    placeholder="Enter any place type: restaurant, hotel, cafe, gym, school, park, etc.",
                    visible=False
                )

            location_status = gr.Markdown("", visible=False)

            map_btn = gr.Button("🗺️ Show Map", variant="primary", size="lg")

            map_output = gr.HTML(label="Map View")
            map_info = gr.Markdown(label="Location Info")

            gr.Examples(
                examples=[
                    ["Search Location", "Taj Mahal, India", "", "", "", ""],
                    ["Get Directions", "", "Times Square, New York", "Central Park, New York", "", ""],
                    ["Find Nearby", "", "", "", "", "pizza"],  # ✅ Empty location = use GPS
                    ["Find Nearby", "", "", "", "London", "gym"],  # ✅ With manual location
                    ["Find Nearby", "", "", "", "", "coffee"],  # ✅ GPS + coffee
                ],
                inputs=[map_mode, location_input, origin_input, destination_input, nearby_location, nearby_type],
                label="💡 Try These Examples"
            )


            def update_map_inputs(mode):
                """Update visible inputs based on selected mode"""
                if mode == "Search Location":
                    return (
                        gr.update(visible=True),  # location_input
                        gr.update(visible=False),  # origin_input
                        gr.update(visible=False),  # destination_input
                        gr.update(visible=False),  # use_location_btn
                        gr.update(visible=False),  # nearby_location
                        gr.update(visible=False),  # nearby_type
                        gr.update(visible=False)  # location_status
                    )
                elif mode == "Get Directions":
                    return (
                        gr.update(visible=False),  # location_input
                        gr.update(visible=True),  # origin_input
                        gr.update(visible=True),  # destination_input
                        gr.update(visible=False),  # use_location_btn
                        gr.update(visible=False),  # nearby_location
                        gr.update(visible=False),  # nearby_type
                        gr.update(visible=False)  # location_status
                    )
                else:  # Find Nearby
                    return (
                        gr.update(visible=False),  # location_input
                        gr.update(visible=False),  # origin_input
                        gr.update(visible=False),  # destination_input
                        gr.update(visible=True),  # use_location_btn
                        gr.update(visible=True),  # nearby_location
                        gr.update(visible=True),  # nearby_type
                        gr.update(visible=True)  # location_status
                    )


            map_mode.change(
                update_map_inputs,
                inputs=map_mode,
                outputs=[location_input, origin_input, destination_input, use_location_btn, nearby_location,
                         nearby_type, location_status]
            )

            # GPS location fetcher with JavaScript
            use_location_btn.click(
                fn=None,
                inputs=None,
                outputs=[nearby_location, location_status],
                js="""
                async () => {
                    console.log('📍 GPS location requested by user');

                    if (!navigator.geolocation) {
                        return ['', '❌ GPS not supported by your browser'];
                    }

                    return new Promise((resolve) => {
                        navigator.geolocation.getCurrentPosition(
                            async (position) => {
                                const lat = position.coords.latitude;
                                const lon = position.coords.longitude;

                                console.log('✅ GPS coordinates:', lat, lon);

                                // ✅ Use backend proxy to avoid CORS
                                try {
                                    const response = await fetch(window.location.origin.replace('7860', '5000') + '/api/reverse-geocode', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ lat, lon })
                                    });

                                    if (response.ok) {
                                        const data = await response.json();

                                        if (data.success) {
                                            const address = data.address;
                                            console.log('✅ Location detected:', address);

                                            resolve([
                                                address,
                                                `✅ Location detected: ${address.substring(0, 100)}...`
                                            ]);
                                        } else {
                                            throw new Error(data.error || 'Reverse geocoding failed');
                                        }
                                    } else {
                                        throw new Error(`Backend error: ${response.status}`);
                                    }

                                } catch (error) {
                                    console.error('Reverse geocoding failed:', error);
                                    // Fallback to coordinates if backend fails
                                    resolve([
                                        `${lat}, ${lon}`,
                                        `✅ Location: ${lat.toFixed(4)}, ${lon.toFixed(4)} (Address lookup failed)`
                                    ]);
                                }
                            },
                            (error) => {
                                console.error('GPS error:', error);
                                let errorMsg = '❌ Location access denied';

                                switch(error.code) {
                                    case error.PERMISSION_DENIED:
                                        errorMsg = '❌ Please allow location access in your browser';
                                        break;
                                    case error.POSITION_UNAVAILABLE:
                                        errorMsg = '❌ Location unavailable. Please check your GPS settings';
                                        break;
                                    case error.TIMEOUT:
                                        errorMsg = '❌ Location request timeout. Please try again';
                                        break;
                                }

                                resolve(['', errorMsg]);
                            },
                            {
                                enableHighAccuracy: true,
                                timeout: 30000, 
                                maximumAge: 0
                            }
                        );
                    });
                }
                """
            )

            map_btn.click(
                generate_ultra_robust_map,
                inputs=[map_mode, location_input, origin_input, destination_input, nearby_location, nearby_type],
                outputs=[map_output, map_info]
            )

            with gr.Tab("Public IP"):
                gr.Markdown("### Check your current public IP address")
                ip_output = gr.Markdown(label="IP Address")
                ip_btn = gr.Button("Get Public IP", variant="primary")

                ip_btn.click(
                    get_public_ip,
                    inputs=None,
                    outputs=ip_output
                )

        with gr.Tab("🌐 Translation", visible=False) as translation_tab:
            gr.Markdown("## 🌍 Universal Translator - Any Language to Any Language")
            gr.Markdown(
                "**Supports:**\n"
                "- 🇮🇳 **All 22 Indian Languages**: Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Urdu, Kannada, Malayalam, Punjabi, Odia, Assamese, and more\n"
                "- 🌎 **30+ Foreign Languages**: Spanish, French, German, Chinese, Japanese, Korean, Russian, Arabic, Portuguese, and more\n"
                "- 🎤 **Voice Input**: Speak in any language\n"
                "- 🔊 **Audio Output**: Hear translation in target language\n"
                "- 🤖 **Auto-detection**: Automatically detects source language"
            )

            with gr.Row():
                mic_translate = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ Speak Your Text (Any Language)"
                )

            translate_input = gr.Textbox(
                label="📝 Or Type Your Text",
                placeholder="Enter text in any language (Hindi, English, Spanish, etc.)...",
                lines=3
            )

            with gr.Row():
                target_lang_dropdown = gr.Dropdown(
                    label="🎯 Target Language (Select from 50+ languages)",
                    choices=[(name, code) for name, code in sorted(SUPPORTED_LANGUAGES.items())],
                    value="hi",
                    filterable=True
                )
                translate_btn = gr.Button("🔄 Translate & Speak", variant="primary", size="lg")

            translation_result = gr.Markdown(label="📄 Translation Result")
            translation_audio = gr.Audio(label="🔊 Listen to Translation", autoplay=True)

            gr.Examples(
                examples=[
                    ["Hello, how are you?", "hi"],
                    ["नमस्ते, आप कैसे हैं?", "en"],
                    ["Bonjour, comment allez-vous?", "te"],
                    ["你好吗？", "ta"],
                    ["مرحبا، كيف حالك؟", "bn"],
                ],
                inputs=[translate_input, target_lang_dropdown],
                label="💡 Try These Examples"
            )

            translate_btn.click(
                perform_translation,
                inputs=[translate_input, target_lang_dropdown],
                outputs=[translate_input, translation_audio, translation_result]
            )


            def transcribe_and_translate(audio_filepath, target_lang):
                if current_user["is_guest"]:
                    return "", None, "**🔒 Register** to use voice translation."

                if not audio_filepath:
                    return "", None, "No audio recorded."

                text = transcribe_audio(audio_filepath)
                if text.startswith("❌"):
                    return "", None, text

                translated, audio_path, full_msg = perform_translation(text, target_lang)
                return translated, audio_path, full_msg


            mic_translate.change(
                transcribe_and_translate,
                inputs=[mic_translate, target_lang_dropdown],
                outputs=[translate_input, translation_audio, translation_result]
            )

    # Authentication Event Handlers
    login_btn.click(
        login_user,
        inputs=[login_username, login_password],
        outputs=[login_status, auth_section, main_app, user_info, file_qa_tab, image_gen_tab, image_qa_tab,
                 image_search_tab, video_gen_tab, translation_tab, maps_tab, stats_btn, history_chatbot,
                 guest_chat_warning,
                 chatbot, session_id, mic_chat]
    )

    request_otp_btn.click(
        request_otp,
        inputs=[reg_username, reg_password, reg_email, reg_fullname],
        outputs=[register_status, otp_input, verify_otp_btn]
    )

    verify_otp_btn.click(
        verify_otp_and_register,
        inputs=[reg_email, otp_input],
        outputs=[register_status, otp_input, verify_otp_btn]
    )

    logout_btn.click(
        logout_user,
        inputs=None,
        outputs=[login_status, auth_section, main_app, user_info, file_qa_tab, image_gen_tab, image_qa_tab,
                 image_search_tab, video_gen_tab, translation_tab, maps_tab, stats_btn, history_chatbot,
                 guest_chat_warning,
                 chatbot, session_id, mic_chat]
    )


    def show_stats():
        stats = get_user_stats()
        return gr.update(visible=True), stats


    def close_stats():
        return gr.update(visible=False)


    stats_btn.click(show_stats, outputs=[stats_modal, stats_content])
    close_stats_btn.click(close_stats, outputs=stats_modal)

    demo.load(
        check_auth_and_load,
        inputs=None,
        outputs=[
            guest_status, auth_section, main_app, user_info,
            file_qa_tab, image_gen_tab, image_qa_tab,
            image_search_tab, video_gen_tab, translation_tab,
            maps_tab, stats_btn, history_chatbot, guest_chat_warning,
            chatbot, session_id, mic_chat
        ]
    )

    # ============================================================================
    # ✅ FIX: JavaScript-based auth polling (replaces Timer)
    # ============================================================================
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
            () => {
                console.log('✅ Auth polling started (using Flask on port 5000)');

                // Check if we just reloaded from a login
                const justReloaded = sessionStorage.getItem('just_reloaded');
                if (justReloaded) {
                    console.log('⏸️ Skipping auth checks - just reloaded from login');
                    sessionStorage.removeItem('just_reloaded');
                    return; // Don't poll if we just reloaded
                }

                window.isLoggedIn = false;
                let pollInterval = null;
                let localStorageInterval = null;

                const authCheckUrl = window.location.origin.replace('7860', '5000') + '/api/check_auth';

                // Function to stop all polling
                function stopPolling() {
                    if (pollInterval) {
                        clearInterval(pollInterval);
                        pollInterval = null;
                        console.log('🛑 Auth polling stopped');
                    }
                    if (localStorageInterval) {
                        clearInterval(localStorageInterval);
                        localStorageInterval = null;
                    }
                }

                // Function to reload page (only once)
                function reloadOnce() {
                    if (!window.isLoggedIn) {
                        window.isLoggedIn = true;
                        stopPolling(); // Stop polling immediately
                        sessionStorage.setItem('just_reloaded', 'true'); // Mark that we're reloading
                        console.log('✅ Login detected! Reloading ONCE...');
                        setTimeout(() => window.location.reload(), 500);
                    }
                }

                // Function to check auth status
                async function checkAuth() {
                    try {
                        const response = await fetch(authCheckUrl);
                        const contentType = response.headers.get('content-type');

                        if (contentType && contentType.includes('application/json')) {
                            const data = await response.json();

                            if (data.logged_in && !window.isLoggedIn) {
                                reloadOnce();
                            }
                        }
                    } catch (e) {
                        console.log('Auth check error:', e);
                    }
                }

                // Check immediately
                checkAuth();

                // Poll every 1 second (will be stopped after login detected)
                pollInterval = setInterval(checkAuth, 1000);

                // Check localStorage for instant Firebase login
                // ✅ ENHANCED: Listen for postMessage from popup
window.addEventListener('message', (event) => {
    if (event.data.type === 'FIREBASE_LOGIN_SUCCESS' && !window.isLoggedIn) {
        console.log('✅ Received postMessage from Firebase popup!');
        reloadOnce();
    }
});

// Check localStorage for instant Firebase login
localStorageInterval = setInterval(() => {
    const loginEvent = localStorage.getItem('firebase_login_event');
    if (loginEvent && !window.isLoggedIn) {
        try {
            const data = JSON.parse(loginEvent);

            if (Date.now() - data.timestamp < 10000) {
                console.log('✅ Fresh Firebase login event detected!');
                localStorage.removeItem('firebase_login_event');
                reloadOnce();
            } else {
                localStorage.removeItem('firebase_login_event');
            }
        } catch (e) {
            localStorage.removeItem('firebase_login_event');
        }
    }
}, 500);
            }
            """
    )


# ============================================================================
# GRADIO API INTERFACES (add BEFORE demo.queue())
# ============================================================================

def check_auth_status_gradio():
    """Gradio API endpoint to check authentication status"""
    return {
        "logged_in": current_user.get("logged_in", False),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "is_guest": current_user.get("is_guest", True),
        "full_name": current_user.get("full_name", "")
    }


def firebase_login_gradio_endpoint(token):
    """Gradio API endpoint for Firebase login"""
    try:
        logging.info("🔥 Firebase login via Gradio API")

        if not token:
            return {"success": False, "error": "No token provided"}

        user_info = verify_firebase_token(token)

        if not user_info:
            return {"success": False, "error": "Invalid or expired token"}

        success, message = register_or_login_firebase_user(user_info)

        if success:
            notify_gradio_login({
                "username": current_user["username"],
                "email": current_user["email"],
                "full_name": current_user.get("full_name", "")
            })

            return {
                "success": True,
                "message": message,
                "user": {
                    "username": current_user["username"],
                    "email": current_user["email"],
                    "full_name": current_user.get("full_name", "")
                }
            }
        else:
            return {"success": False, "error": message}

    except Exception as e:
        logging.error(f"Firebase login error: {e}")
        return {"success": False, "error": str(e)}


# Register API endpoints
check_auth_api = gr.Interface(
    fn=check_auth_status_gradio,
    inputs=None,
    outputs=gr.JSON(),
    api_name="check_auth"
)

firebase_login_api = gr.Interface(
    fn=firebase_login_gradio_endpoint,
    inputs=gr.Textbox(visible=False),
    outputs=gr.JSON(),
    api_name="firebase_login"
)

# ============================================================================
# ✅ CRITICAL: Queue MUST be called here (after all components defined)
# ============================================================================
demo.queue()

# ============================================================================
# ✅ MOUNT API INTERFACES
# ============================================================================
# Combine main demo with API interfaces
from gradio import mount_gradio_app

# This creates the combined app with all endpoints
combined_app = gr.TabbedInterface(
    [demo, check_auth_api, firebase_login_api],
    ["Main", "Auth Check", "Firebase Login"],
    title="All Mind"
)

# ============================================================================
# CORRECTED SERVER STARTUP - ORIGINAL THREADING METHOD
# ============================================================================
if __name__ == "__main__":
    from threading import Thread

    logging.info("=" * 80)
    logging.info("🚀 APPLICATION STARTING")
    logging.info(f"   Gradio UI: http://localhost:7860")
    logging.info(f"   API Server: http://localhost:5000")
    if FIREBASE_AVAILABLE:
        logging.info("   Firebase: ✅ Ready")
    else:
        logging.info("   Firebase: ⚠️ Not configured")
    logging.info("=" * 80)


    # ============================================================================
    # ✅ START FLASK API SERVER (Port 5000) - Runs in separate thread
    # ============================================================================
    def run_flask():
        """Run Flask API server on port 5000"""
        logging.info("🔥 Starting Flask API server on port 5000...")
        flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logging.info("✅ Flask API server running on http://localhost:5000")

    # Small delay to ensure Flask starts
    import time

    time.sleep(1)

    # ============================================================================
    # ✅ START GRADIO (Port 7860)
    # ============================================================================
    logging.info("🚀 Starting Gradio UI on port 7860...")
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False
    )
