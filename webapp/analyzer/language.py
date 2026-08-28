"""
Language detection and translation service for VADER Sentiment Analyzer.

Provides:
1. Fast, offline, heuristic-based language detection (Unicode script inspection
   and VADER/English lexicon coverage).
2. Fail-safe, non-blocking translation to English via the public MyMemory API
   (aligned with the official VADER demo).
"""

import html
import logging
import re
from typing import Any, Dict, List, Optional
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 8
MAX_TRANSLATE_CHARS = 1500
CHUNK_MAX_CHARS = 500

# Common English function words and high-frequency vocabulary for Latin script detection
COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "been", "has", "had", "am", "did", "does", "done", "where", "why", "here", "such", "very", "much",
    "more", "many", "between", "through", "during", "before", "should", "each", "those", "both", "under",
    "never", "always", "same", "another", "while", "last", "might", "great", "off", "still", "find",
    "again", "few", "house", "world", "tell", "feel", "high", "every", "own", "against", "right",
    "place", "long", "small", "large", "point", "home", "hand", "part", "number", "system", "water",
    "group", "room", "lot", "bad", "best", "worst", "product", "service", "customer", "really", "nice",
    "amazing", "terrible", "horrible", "awesome", "review", "experience", "item", "order", "buy", "bought"
}

_analyzer_instance = SentimentIntensityAnalyzer()
_ENGLISH_VOCABULARY = set(_analyzer_instance.lexicon.keys()) | COMMON_ENGLISH_WORDS

LATIN_LANG_STOPWORDS = {
    "es": {
        "el", "la", "de", "que", "los", "del", "las", "por", "para", "con", "una", "su", "al", "lo",
        "como", "mas", "pero", "sus", "le", "ya", "fue", "este", "si", "porque", "esta", "son",
        "entre", "cuando", "muy", "sin", "sobre", "ser", "tiene", "tambien", "hasta", "hay", "donde",
        "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros",
        "ese", "eso", "ante", "ellos", "esto", "antes", "algunos", "unos", "otro", "otras", "otra",
        "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "sea", "poco", "ella",
        "estar", "haber", "estas", "estaba", "estamos", "bueno", "buena", "excelente", "comida",
        "servicio", "lugar", "increible", "genial", "pesimo", "maravilloso", "maravillosa", "restaurante", "hotel"
    },
    "fr": {
        "le", "la", "les", "de", "des", "du", "une", "et", "est", "que", "qui", "dans", "pour",
        "pas", "sur", "ce", "plus", "au", "par", "avec", "tout", "faire", "son", "mettre", "autre",
        "mais", "nous", "comme", "ou", "si", "leur", "dire", "elle", "doit", "sans", "bon",
        "bonne", "magnifique", "tres", "cette", "merci", "bien", "beaucoup", "jadore", "adore",
        "experience", "absolument"
    },
    "de": {
        "der", "die", "das", "und", "den", "von", "zu", "mit", "sich", "des", "auf", "fur",
        "ist", "im", "dem", "nicht", "ein", "eine", "einen", "einem", "einer", "als", "auch",
        "werden", "aus", "hat", "dass", "sie", "nach", "wird", "bei", "sind", "noch", "wie",
        "uber", "zum", "war", "haben", "nur", "oder", "aber", "wunderbar", "schon", "wirklich", "gut", "sehr"
    },
    "it": {
        "il", "la", "di", "che", "per", "una", "sono", "non", "si", "lo", "ma", "come", "del",
        "dei", "delle", "della", "questo", "molto", "tutto", "anche", "grazie", "bello", "buono", "ottimo"
    },
    "pt": {
        "os", "as", "de", "do", "da", "uma", "para", "com", "nao", "que", "se", "por", "mais",
        "como", "dos", "das", "muito", "excelente", "bom", "boa", "obrigado"
    },
}

SCRIPT_LANG_MAP = {
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "he": "Hebrew",
    "th": "Thai",
    "ru": "Russian",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "und": "Detected non-English",
    "en": "English",
}


def _get_char_script(char: str) -> Optional[str]:
    """Identify the Unicode script block for a given character."""
    cp = ord(char)
    # Japanese Kana (Hiragana & Katakana)
    if (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
        return "ja"
    # CJK Unified Ideographs
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0xF900 <= cp <= 0xFAFF)
    ):
        return "zh"
    # Korean Hangul
    if (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
        return "ko"
    # Arabic
    if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
        return "ar"
    # Hebrew
    if 0x0590 <= cp <= 0x05FF:
        return "he"
    # Thai
    if 0x0E00 <= cp <= 0x0E7F:
        return "th"
    # Cyrillic
    if (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F):
        return "ru"
    # Devanagari (Hindi)
    if 0x0900 <= cp <= 0x097F:
        return "hi"
    return None


def detect_language(text: str) -> Dict[str, Any]:
    """
    Heuristically detect the language of the input text using Unicode script analysis
    and English vocabulary/lexicon hit rates.

    Returns:
        Dict with keys:
            - code: ISO language code (e.g. 'en', 'zh', 'ja', 'ru', 'ar', 'es', 'fr', 'de', 'und')
            - name: Human-readable language name (e.g. 'English', 'Japanese', 'Spanish')
            - is_english: bool indicating if the text is English or treated as English.
    """
    clean_text = (text or "").strip()
    if not clean_text or len(clean_text) < 3:
        return {"code": "en", "name": "English", "is_english": True}

    letters = [c for c in clean_text if c.isalpha()]
    if not letters:
        # Emojis or symbols only: default to English
        return {"code": "en", "name": "English", "is_english": True}

    # 1. Non-Latin Unicode Script Checks
    script_counts: Dict[str, int] = {}
    for c in letters:
        script = _get_char_script(c)
        if script:
            script_counts[script] = script_counts.get(script, 0) + 1

    total_letters = len(letters)
    # If Japanese Kana are present, prioritize Japanese over generic CJK
    if script_counts.get("ja", 0) > 0:
        return {"code": "ja", "name": SCRIPT_LANG_MAP["ja"], "is_english": False}

    if script_counts:
        top_script, top_count = max(script_counts.items(), key=lambda x: x[1])
        # If at least 30% of alphabetic characters belong to a distinct non-Latin script
        if (top_count / total_letters) >= 0.30:
            return {
                "code": top_script,
                "name": SCRIPT_LANG_MAP.get(top_script, "Detected non-English"),
                "is_english": False,
            }

    # 2. Latin Script: Tokenize and compute hit rate against English vocabulary
    alpha_tokens = [
        re.sub(r"^[^\w]+|[^\w]+$", "", t.lower())
        for t in clean_text.split()
        if any(c.isalpha() for c in t)
    ]
    alpha_tokens = [t for t in alpha_tokens if t]

    # If text is too short (< 3 words), treat as English to avoid false positives
    if len(alpha_tokens) < 3:
        return {"code": "en", "name": "English", "is_english": True}

    hits_vocab = sum(1 for t in alpha_tokens if t in _ENGLISH_VOCABULARY)
    hits_lexicon = sum(1 for t in alpha_tokens if t in _analyzer_instance.lexicon)
    vocab_rate = hits_vocab / len(alpha_tokens)
    lex_rate = hits_lexicon / len(alpha_tokens)

    # High English vocabulary confidence
    if vocab_rate >= 0.50:
        return {"code": "en", "name": "English", "is_english": True}

    # Check for specific European Latin languages when English confidence is low
    latin_lang_hits: Dict[str, int] = {}
    for lang_code, words in LATIN_LANG_STOPWORDS.items():
        count = sum(1 for t in alpha_tokens if t in words or t.strip("'\"`’") in words)
        if count >= 1:
            latin_lang_hits[lang_code] = count

    if latin_lang_hits and (vocab_rate < 0.40 or lex_rate < 0.12):
        best_lang = max(latin_lang_hits.items(), key=lambda x: x[1])[0]
        return {
            "code": best_lang,
            "name": SCRIPT_LANG_MAP.get(best_lang, "Detected non-English"),
            "is_english": False,
        }

    # If hit rate is significantly below standard English baseline, classify as non-English
    if vocab_rate < 0.25 and lex_rate < 0.08:
        return {"code": "und", "name": "Detected non-English", "is_english": False}

    return {"code": "en", "name": "English", "is_english": True}


def _split_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Split text into manageable chunks respecting sentence or word boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Split by sentences or line breaks
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    chunks: List[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = f"{current} {s}".strip() if current else s
        else:
            if current:
                chunks.append(current)
                current = s
            else:
                # Single sentence exceeds chunk size: split by words
                words = s.split()
                w_current = ""
                for w in words:
                    if len(w_current) + len(w) + 1 <= max_chars:
                        w_current = f"{w_current} {w}".strip() if w_current else w
                    else:
                        if w_current:
                            chunks.append(w_current)
                        w_current = w
                if w_current:
                    current = w_current

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def translate_to_english(text: str, source_lang: str = "und") -> Dict[str, Any]:
    """
    Translate text from detected source language to English via the public MyMemory API.
    Enforces maximum character cap (1500 chars), ~500-char chunking, HTML entity decoding,
    timeouts, and fail-open error handling.

    Args:
        text: Original text to translate.
        source_lang: Detected source language code (e.g. 'es', 'ja', 'und', 'Autodetect').

    Returns:
        Dict with keys:
            - text: The translated English string (or original string on failure).
            - used: bool indicating whether translation was successfully applied.
            - truncated: bool indicating whether input exceeded MAX_TRANSLATE_CHARS.
            - error: Optional error string if translation failed.
    """
    clean_text = (text or "").strip()
    if not clean_text or source_lang == "en":
        return {"text": clean_text, "used": False, "truncated": False, "error": None}

    truncated = len(clean_text) > MAX_TRANSLATE_CHARS
    to_translate = clean_text[:MAX_TRANSLATE_CHARS]

    src = source_lang if source_lang and source_lang != "und" else "Autodetect"
    chunks = _split_into_chunks(to_translate, max_chars=CHUNK_MAX_CHARS)
    translated_chunks: List[str] = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                "q": chunk,
                "langpair": f"{src}|en",
            }
            resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT)

            if resp.status_code != 200:
                return {
                    "text": clean_text,
                    "used": False,
                    "truncated": truncated,
                    "error": f"Translation service returned status {resp.status_code}",
                }

            data = resp.json()
            response_data = data.get("responseData") or {}
            translated_text = response_data.get("translatedText")

            # Check for API error status inside JSON payload (e.g. rate limit responseStatus = 403 / 429)
            resp_status = data.get("responseStatus")
            if resp_status and str(resp_status) not in ("200", 200):
                details = data.get("responseDetails") or f"Translation status {resp_status}"
                return {
                    "text": clean_text,
                    "used": False,
                    "truncated": truncated,
                    "error": str(details),
                }

            if not translated_text:
                return {
                    "text": clean_text,
                    "used": False,
                    "truncated": truncated,
                    "error": "Empty translation returned from service.",
                }

            clean_trans = html.unescape(translated_text.strip())
            translated_chunks.append(clean_trans)

        except requests.exceptions.Timeout:
            return {
                "text": clean_text,
                "used": False,
                "truncated": truncated,
                "error": "Translation service timed out.",
            }
        except requests.exceptions.RequestException as req_err:
            return {
                "text": clean_text,
                "used": False,
                "truncated": truncated,
                "error": f"Network error connecting to translation service: {req_err}",
            }
        except Exception as ex:
            return {
                "text": clean_text,
                "used": False,
                "truncated": truncated,
                "error": f"Translation processing error: {ex}",
            }

    if translated_chunks:
        full_translation = " ".join(translated_chunks)
        return {
            "text": full_translation,
            "used": True,
            "truncated": truncated,
            "error": None,
        }

    return {
        "text": clean_text,
        "used": False,
        "truncated": truncated,
        "error": "No translated content produced.",
    }
