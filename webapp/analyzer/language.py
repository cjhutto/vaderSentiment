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
    "tl": {
        "ang", "mga", "ng", "sa", "ako", "ikaw", "siya", "kami", "kayo", "sila", "ito", "iyan", "iyon",
        "dito", "diyan", "doon", "hindi", "oo", "wala", "meron", "may", "napaka", "sobrang", "masaya",
        "lungkot", "maganda", "pangit", "mahal", "salamat", "bwisit", "lintik", "ulol", "tanga", "gago",
        "tarantado", "putang", "ina", "tangina", "putangina", "mo", "ko", "naman", "talaga", "kasi",
        "pero", "dahil", "para", "paano", "kailan", "saan", "sino", "bakit", "lahat", "tao", "buhay",
        "araw", "oras", "taon", "trabaho", "bahay", "kaibigan", "pamilya", "pagkain", "serbisyo",
        "produkto", "kuya", "ate", "anak", "magulang", "bata", "matanda", "babae", "lalaki", "ayos",
        "palpak", "sayang", "lodi", "petmalu", "astig", "bastos", "walanghiya", "hayop", "leche",
        "punyeta", "inutil", "bobo", "kupal", "yawa", "piste", "buwisit", "siraulo", "salot", "ungas",
        "pucha", "pakshet", "budol", "sulit", "sarap", "galing", "husay", "tamad", "kadiri",
        "dismayado", "marikit", "panalo", "mura", "bilis", "mabilis", "bagal", "mabagal",
        "basura", "bulok", "panis", "baho", "linis", "malinis", "bwiset", "bwct", "tngina", "tngna"
    },
    "ceb": {
        "ang", "mga", "sa", "og", "ug", "ako", "ikaw", "siya", "kami", "kamo", "sila", "kini", "kana",
        "kadto", "diri", "diha", "didto", "dili", "oo", "wala", "naa", "kaayo", "pirti", "lipay",
        "guol", "gwapa", "gwapo", "bati", "mahal", "salamat", "daghang", "yawa", "piste", "atay",
        "bilat", "buang", "yawaa", "pisting", "kayata", "amaw", "kol", "lami", "nindot", "baho", "hugaw",
        "samok", "hilas", "giatay", "pastilan", "namit"
    },
    "ilo": {
        "ti", "dagiti", "iti", "siak", "sika", "isu", "dakami", "dakayo", "isuda", "daytoy", "dayta",
        "daydiay", "ditoy", "dita", "sadiay", "saan", "wen", "awan", "adda", "unay", "nagimas",
        "nagsayaat", "napintas", "naglaing", "agbiag", "agyaman", "ukinam", "ukininam", "bagtit",
        "torpe", "agsubli", "naimbag", "bigat", "malem", "rabi", "nagalas", "bulsit"
    },
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

# Comprehensive idiom, slang, and profanity normalizer for Philippine dialects (Tagalog/Filipino, Bisaya/Cebuano, Ilocano)
# Supports formal, colloquial, SMS/text-speak, chat abbreviations, and elongated chat spellings
FILIPINO_DIALECT_MAP = {
    # Strong Profanities, Curses & Vulgar Insults
    r"\b(p+u+t+a+n+g*|p+o+t+a+n+g*|p+t+a+n+g*|p+u+t+a*)\s*i+n+a+\s*(m+o+|n+y+o+|k+a+)\b": "you motherfucker",
    r"\b(t+a+n+g*|t+n+g*|t+a+n+g*e*|t+e+n+g*)\s*i+n+a+\s*(m+o+|n+y+o+|k+a+)\b": "you motherfucker",
    r"\b(p+u+t+a+n+g*i+n+a+|p+o+t+a+n+g*i+n+a+|p+t+a+n+g*i+n+a+|t+a+n+g*i+n+a+|t+n+g*i+n+a+|t+a+n+g*e+n+a+|t+n+g*n+a+)\s*(m+o+|n+y+o+|k+a+)\b": "you motherfucker",
    r"\b(p+u+t+a+n+g*i+n+a+|p+o+t+a+n+g*i+n+a+|p+t+a+n+g*i+n+a+|t+a+n+g*i+n+a+|t+n+g*i+n+a+|t+a+n+g*e+n+a+|t+n+g*n+a+|p+u+t+a+n+g*\s*i+n+a+|t+a+n+g*\s*i+n+a+)\b": "fucking damn it",
    r"\b(p+u+k+i+n+g*|p+k+n+g*)\s*i+n+a+\s*(m+o+|n+y+o+|k+a+)\b": "you motherfucker",
    r"\b(p+u+k+i+n+g*|p+k+n+g*)\s*i+n+a+\b": "fucking damn it",
    r"\b(p+a+k+s+h+e+t+|p+a+k+s+h+i+t+|p+k+s+h+t+|p+a+k+s+h+e+e+t+)\b": "fucking shit",
    r"\b(p+u+c+h+a+|p+u+t+s+a+|p+u+t+r+a+g+i+s+)\b": "damn it",
    r"\b(p+a+k+y+u+|f+a+k+y+u+)\b": "fuck you",
    r"\b(k+a*n+t+o+t+|k+a*n+t+u+t+a*n+|h+i*n+d+o+t+)\b": "fucking vulgar",
    r"\b(b+u+r+a+t+|t+i+t+e+|b+a+y+a+g+|p+e+k+p+e+k+|p+u+k+i+|t+a+m+o+d+)\b": "disgusting vulgar",
    r"\b(g+a+g+o+|g+a+g+a+|g+g+o+|g+g+a+|g+o+g+o+)\b": "fucking idiot asshole",
    r"\b(t+a+r+a+n+t+a+d+o+|t+a+r+a+n+t+a+d+a+|t+r+n+t+d+o+|t+r+n+t+d+a+)\b": "bastard asshole",
    r"\b(u+l+o+l+|o+l+o+l+|u+l+u+l+|o+l+o+)\b": "crazy stupid idiot",
    r"\b(b+o+b+o+|b+b+o+|b+u+b+u+|b+o+b+a+)\b": "stupid idiot dumb",
    r"\b(t+a+n+g+a+|t+n+g+a+|e+n+g+o+t+|g+u+n+g+g+o+n+g+|t+i+m+a+n+g+|a+b+n+o+y+)\b": "stupid fool idiot dumb",
    r"\b(i+n+u+t+i+l+|i+n+u+t+e+l+|n+u+t+i+l+|h+u+n+g+k+a+g+)\b": "useless incompetent idiot",
    r"\b(k+u+p+a+l+|k+p+a+l+|q+p+a+l+)\b": "worthless jerk asshole",
    r"\b(u+n+g+a+s+|u+n+g+g+o+y+)\b": "foolish idiot",
    r"\b(s+i+r+a+u+l+o+|s+i+r+a+\s*u+l+o+|s+r+a+u+l+o+|l+o+k+o+\s*l+o+k+o+|l+u+k+o+\s*l+u+k+o+)\b": "crazy lunatic",
    r"\b(w+a*l+a*n+g*\s*h+i+y+a+|w+l+a+n+g*h+y+a+|w+l+n+g*\s*h+y+a+)\b": "shameless scoundrel bastard",
    r"\b(b+w+i+s+i+t+|b+w+i+s+e+t+|b+u+w+i+s+i+t+|b+w+s+i+t+|b+w+c+t+|b+w+s+e+t+)\b": "annoying nuisance cursed",
    r"\b(l+i+n+t+i+k+|l+n+t+k+|l+i+n+t+i+a+n+)\b": "damn you cursed",
    r"\b(l+e+c+h+e+|l+e+t+s+e+|l+t+s+e+|l+i+t+s+i+|l+i+t+s+e+)\b": "damn it garbage",
    r"\b(p+u+n+y+e+t+a+|p+n+y+t+a+|p+o+n+y+e+t+a+|p+u+n+y+e+t+a+h+a*n*)\b": "fucking damn it",
    r"\b(s+a+l+o+t+|s+l+t+|s+a+l+b+a+h+e+)\b": "scourge pest disaster villain",
    r"\b(p+e+s+t+e+|p+s+t+e+)\b": "fucking pest nuisance",
    r"\b(h+a+y+o+p+|h+a+y+u+p+|h+y+o+p+)\s*(k+a+|m+o+)?\b": "you animal beast bastard",
    r"\b(d+e+m+o+n+y+o+|d+m+n+y+o+)\s*(k+a+|m+o+)?\b": "you evil devil demon",
    r"\b(h+u+d+a+s+|h+d+s+|t+a+k+s+i+l+|t+r+a+y+d+o+r+|a+h+a+s+)\b": "traitor betrayer deceitful backstabber",
    r"\b(h+i+n+a+y+u+p+a+k+|h+n+y+p+k+)\b": "cursed animal bastard",
    r"\b(b+a+s+t+o+s+|b+s+t+s+|w+a*l+a*n+g*\s*m+o+d+o+)\b": "disrespectful rude offensive ill-mannered",
    r"\b(m+a*s+u+n+g+i+t+|m+s+n+g+t+|s+u+p+l+a+d+o+|s+u+p+l+a+d+a+)\b": "grumpy rude hostile unwelcoming",
    r"\b(m+a*n+y+a*k+i*s*|m+a*n+y+a*k+|b+a*b+a*e+r+o+)\b": "pervert harasser creep",
    r"\b(b+u+r+a+o+t+|s+a*k+i*m+|g+a*h+a*m+a*n+|k+u+r+i+p+o+t+|s+w+a*p+a*n+g+)\b": "greedy stingy selfish",
    r"\b(p+a*t+a*y*\s*g+u+t+o+m+|p+a*t+a*y+-g+u+t+o+m+)\b": "greedy impoverished freeloader",
    r"\b(e+p+a+l+|p+a*p+a*n+s+i+n+|p+a*-*b+i+d+a+|p+a*b+i+d+a+)\b": "annoying attention-seeking obnoxious nuisance",
    r"\b(m+a*y+a*b+a*n+g+|h+a*m+b+o*g*|h+a*m+b+u+g*|h+a*m+b+u+g+e+r+o*)\b": "arrogant boastful obnoxious",
    r"\b(p+l+a*s+t+i*k+|p+e+k+e+|s+i+p+s+i+p+)\b": "fake hypocrite brownnoser deceptive",
    r"\b(b+u+d+o+l+|n+a*b+u+d+o+l+|n+a*-*s+c+a*m+|m+a*n+l+o+l+o+k+o+|m+a*n+g*g+a*g+a*n+t+s+o+)\b": "scammer fraud fraudster con thief",

    # Cebuano / Bisaya Profanities & Slang
    r"\b(p+i+s+t+i+n+g*|p+s+t+n+g*)\s*(y+a+w+a+|y+w+a+)\b": "fucking devil bastard",
    r"\b(y+a+w+a+|y+w+a+|y+a+w+a+a+)\s*(k+a+|m+o+)?\b": "you fucking devil bastard",
    r"\b(p+i+s+t+e+|p+s+t+e+)\b": "fucking pest damn",
    r"\b(b+i+l+a+t+|b+l+t+)\s*s+a*\s*i+n+a+\b": "fucking cunt motherfucker",
    r"\b(b+i+l+a+t+|b+l+t+|b+i+l+a+t+i+b+a+y+)\b": "fucking cunt",
    r"\b(a+t+a+y+|a+t+y+|g+i+a+t+a+y+)\b": "damn it hell cursed",
    r"\b(k+a+y+a+t+a+|k+y+t+a+)\b": "fucking shit",
    r"\b(b+u+a+n+g+|b+w+a+n+g+|b+n+g+|b+u+a+n+g+-b+u+a+n+g+)\b": "crazy insane idiot",
    r"\b(a+m+a+w+|a+m+w+|k+o+l+o+k+o+)\b": "stupid fool lunatic",
    r"\b(s+a+m+o+k+|s+a+m+o+k+a+|h+i+l+a+s+|h+i+l+a+s+a+)\b": "annoying troublesome arrogant obnoxious",
    r"\b(p+a+s+t+i+l+a+n+)\b": "damn it my goodness frustrating",

    # Ilocano Profanities
    r"\b(u+k+i+n+a+m+|u+k+i+n+i+n+a+m+|o+k+i+n+a+m+|u+k+i+n+a+n+a+)\b": "motherfucker you",
    r"\b(b+a+g+t+i+t+|b+g+t+t+)\b": "crazy insane idiot",
    r"\b(b+u+l+s+i+t+|t+o+r+p+e+)\b": "foolish stupid cursed",

    # Negative Customer Service, Food, Product & Experience Reviews
    r"\b(w+a*l+a*n+g*|w+l+n+g*)\s*(k+w+e+n+t+a+|k+w+n+t+a+|s+i+l+b+i+|s+l+b+i+)\b": "worthless useless garbage waste",
    r"\b(s+a*y+a*n+g*|s+y+n+g*)\s*(p+e+r+a+|b+a*y+a*d+|o+r+a*s+|k+w+a+r+t+a+)\b": "waste of money and time loss regret",
    r"\b(t+a*p+o+n*\s*p+e+r+a+|l+u+g+i+|l+u+g+i+n+g*\s*l+u+g+i+)\b": "total financial loss ruined regret",
    r"\b(p+a*l+p+a*k+|p+l+p+k+|b+a*g+s+a*k+|s+a*b+l+a*y+)\b": "failed defective ruined disaster failure",
    r"\b(b+a*s+u+r+a+|b+s+r+a+|b+u+l+o+k+|s+i+r+a+)\b": "garbage trash worthless spoiled broken",
    r"\b(p+a*n+i+s+|m+a*b+a*h+o+|b+a*h+o+|m+a*a*s+i+m+|m+a*p+a*i+t+|s+u+n+o+g+|m+a*t+a*b+a*n+g+|m+a*l+a*n+s+a+)\b": "spoiled stinky disgusting unpalatable rotten",
    r"\b(m+a*d+u+m+i+|m+a*r+u+m+i+|k+a*d+i+r+i+|n+a*k+a*k+a*d+i+r+i+|k+d+r+i+|n+a*k+a*k+a*s+u+k+a+|n+a*k+a*k+a*s+u+k+l+a*m+)\b": "disgusting filthy revolting gross nauseating",
    r"\b(n+a*k+a*k+a*i+n+i+s+|n+k+k+i+n+i+z*|n+k+k+a*i+n+i+s+|k+a*i+n+i+s+|k+n+s+|n+a*k+a*k+a*a*s+a*r+|a*s+a*r+)\b": "very annoying irritating frustrating obnoxious",
    r"\b(n+a*k+a*k+a*g+a*l+i+t+|n+k+k+g+l+t+|k+a*g+a*l+i+t+|n+a*k+a*k+a*p+i+k+o+n+|p+i+k+o+n+)\b": "enraging infuriating upsetting maddening",
    r"\b(n+a*k+a*k+a*h+i+y+a+|k+a*h+i+y+a+|k+h+y+a+)\b": "embarrassing shameful humiliating disgraceful",
    r"\b(d+i+s+m+a*y+a*d+o+|d+s+m+y+d+o+|n+a*k+a*k+a*d+i+s+m+a*y+a+)\b": "disappointed dissatisfied frustrated letdown",
    r"\b(s+o+b+r+a*n+g*|s+b+r+n+g*|n+a*p+a*k+a*)\s*(l+u+n+g*k+o+t+|l+n+g*k+t+)\b": "extremely sad sorrowful depressed",
    r"\b(m+a*l+u+n+g*k+o+t+|l+u+n+g*k+o+t+|l+n+g*k+t+|n+a*k+a*k+a*i*y+a*k+|n+a*k+a*k+a*a*w+a+)\b": "very sad sorrowful tragic pitiful",
    r"\b(s+o+b+r+a*n+g*|m+a*s+y+a*d+o*n+g*|m+s+y+d+n+g*)\s*(m+a*h+a*l+|m+h+l+|g+i+n+t+o+)\b": "overpriced extremely expensive rip-off",
    r"\b(s+o+b+r+a*n+g*|m+a*s+y+a*d+o*n+g*|m+s+y+d+n+g*)\s*(b+a*g+a*l+|m+a*b+a*g+a*l+|k+u+p+a*d+|m+a*k+u+p+a*d+|u+s+a*d*\s*p+a*g+o+n+g*)\b": "extremely slow sluggish delayed tardy",
    r"\b(t+a*m+a*d+|a*y+a*w*\s*m+a*g*\s*a*s+s+i+s+t+|n+a*k+a*s+i*m+a*n+g+o+t+)\b": "lazy rude unhelpful hostile staff",
    r"\b(s+i+r+a*\s*a*g+a*d+|b+a*s+a*g+|d+e+p+e+k+t+o+|d+e+p+e+k+t+i+b+o+|k+u+l+a*n+g*\s*k+u+l+a*n+g*)\b": "defective broken damaged incomplete faulty",
    r"\b(d+i+|h+i+n+d+i+|h+n+d+i+|n+d+i+)\s*(s+u+l+i+t+|w+o+r+t+h*\s*i+t+)\b": "not worth it poor value rip-off",
    r"\b(d+i+|h+i+n+d+i+|h+n+d+i+|n+d+i+)\s*(m+a*s+a*r+a*p+|s+a*r+a*p+|s+r+p+)\b": "terrible tasting bad awful food",
    r"\b(d+i+|h+i+n+d+i+|h+n+d+i+|n+d+i+)\s*(m+a*a*s+a*h+a*n+|m+a*k+a*a*s+a+|r+e+k+o+m+e+n+d+a+d+o+|r+e+c+o+m+m+e+n+d+e+d+)\b": "unreliable not recommended untrustworthy",
    r"\b(w+a*g*\s*k+a*y+o*\s*b+i*b+i*l+i+|w+a*g*\s*s+u*b+u*k+a*n+|h+u+w+a*g*\s*b+i*b+i*l+i+)\b": "do not buy avoid terrible warning",
    r"\b(n+a*p+a*k+a*|s+o+b+r+a*n+g*|s+b+r+n+g*)\s*(p+a*n+g+[e|i]+t+|p+n+g+[e|i]+t+)\b": "extremely ugly terrible horrible",
    r"\b(p+a*n+g+[e|i]+t+|p+n+g+[e|i]+t+|b+a*t+i+|b+t+i+)\b": "ugly bad terrible poor",

    # Positive Customer Service, Food, Product & Experience Reviews
    r"\b(n+a*p+a*k+a*|s+o+b+r+a*n+g*|s+b+r+n+g*)\s*(g+a*n+d+a+|g+n+d+a+|g+a*n+d+a*h+)\b": "extremely beautiful wonderful fantastic",
    r"\b(m+a*r+i+k+i+t+|m+r+k+t+|n+a*p+a*k+a*r+i+k+i+t+)\b": "beautiful lovely exquisite",
    r"\b(m+a*g+a*n+d+a+|g+a*n+d+a+|g+n+d+a+)\b": "beautiful nice good pleasant",
    r"\b(n+a*p+a*k+a*|s+o+b+r+a*n+g*|s+b+r+n+g*)\s*(s+a*r+a*p+|s+r+p+)\b": "extremely delicious tasty mouthwatering",
    r"\b(m+a*s+a*r+a*p+|s+a*r+a*p+|s+r+p+|n+a*m+i+t+|l+a*m+i+)\b": "delicious tasty great yummy",
    r"\b(n+a*p+a*k+a*|s+o+b+r+a*n+g*|s+b+r+n+g*)\s*(h+u+s+a+y+|g+a+l+i+n+g+|g+l+n+g+)\b": "extremely excellent outstanding stellar brilliant",
    r"\b(m+a*h+u+s+a+y+|g+a*l+i+n+g+|g+l+n+g+)\b": "excellent great skilled wonderful",
    r"\b(n+a*p+a*k+a*|s+o+b+r+a*n+g*|s+b+r+n+g*)\s*(s+a+y+a+|s+y+a+)\b": "extremely happy joyful delighted thrilling",
    r"\b(m+a*s+a*y+a+|s+a*y+a+|s+y+a+)\b": "happy joyful pleased glad",
    r"\b(n+a*k+a*k+a*t+u+w+a+|n+a*k+a*k+a*t+a*b+a*\s*n+g*\s*p+u+s+o+)\b": "delightful heartwarming charming touching",
    r"\b(m+a*b+a*i+t+|b+a*i+t+|m+b+i+t+|m+a*a*s+i+k+a*s+o+|m+a*t+u+l+u+n+g+i+n+|m+a*g+a*l+a*n+g+)\b": "very kind friendly accommodating helpful polite staff",
    r"\b(a+s+t+i+g+|a+s+t+e+e+g+|p+e+t+m+a+l+u+|p+e+t+m+a+l+o+o+|l+o+d+i+|l+o+d+e+e+|w+e+r+p+a+)\b": "awesome incredible fantastic powerhouse great",
    r"\b(p+a*n+a*l+o+|p+n+l+o+|d+a*b+e+s+t+|p+a*n+a*l+o*n+g*\s*p+a*n+a*l+o+)\b": "winner champion outstanding number one best",
    r"\b(s+o+l+i+d+|s+l+d+)\b": "solid fantastic reliable top-notch",
    r"\b(s+u+l+i+t*\s*n+a*\s*s+u+l+i+t+|s+u+l+i+t+|w+o+r+t+h*\s*i+t+|s+o+b+r+a*n+g*\s*w+o+r+t+h*\s*i+t+)\b": "super worth it great value bargain satisfying",
    r"\b(a*b+o+t*\s*k+a*y+a+|b+u+d+g+e+t*\s*f+r+i+e+n+d+l+y+|m+u+r+a+)\b": "affordable budget-friendly economical cheap",
    r"\b(s+o+b+r+a*n+g*|n+a*p+a*k+a*)\s*(b+i+l+i+s+|m+a*b+i+l+i+s+|f+l+a*s+h*)\b": "super fast speedy prompt swift delivery",
    r"\b(m+a*l+i+n+i+s+|m+a*a*l+i+w+a*l+a*s+|p+r+e+s+k+o+|m+a*b+a*n+g+o+)\b": "very clean fresh fragrant pleasant spotless relaxing",
    r"\b(l+e+g+i+t*\s*n+a*\s*l+e+g+i+t+|o+r+i+g*\s*n+a*\s*o+r+i+g*|o+r+i+h+i+n+a+l+|m+a*g+a*n+d+a*n+g*\s*q+u+a*l+i+t+y+)\b": "authentic genuine original high quality reliable",
    r"\b(m+a*r+a*m+i+n+g*\s*s+a*l+a*m+a*t+|s+a*l+a*m+a*t+|s+l+m+a*t+|s+a*l+a*m+u+c+h+|t+h+a*n+k*\s*y+o+u+|t+y*\s*p+o+)\b": "thank you very much grateful appreciative",
    r"\b(r+e+c+o+m+m+e+n+d+e+d*\s*k+o+|i+r+e+c+o+m+m+e+n+d+|r+e+k+o+m+e+n+d+a+d+o+|p+e+r+p+e+k+t+o+)\b": "highly recommended five stars excellent perfect top rating",
    r"\b(u+u+l+i+t*\s*u+l+i+t+i+n+|b+a*b+a*l+i+k*\s*b+a*l+i+k+a*n+|b+a*b+a*l+i+k*\s*a*k+o+|o+r+d+e+r*\s*u+l+i+t+)\b": "will definitely return repeat customer highly satisfied",
    r"\b(m+a*h+a*l*\s*k+i+t+a+|i+n+i+i+b+i+g*\s*k+i+t+a+)\b": "I love you dearly with all my heart",
    r"\b(s+u*w+e+r+t+e+|s+w+r+t+e+)\b": "lucky fortunate blessed",
    r"\b(n+a+i+m+b+a+g+|n+a*g+s+a*y+a*a*t+|n+a*p+i+n+t+a+s+|n+a*g+l+a*i+n+g+)\b": "good wonderful excellent beautiful skilled",
    r"\b(n+i+n+d+o+t+|n+n+d+t+|l+a+m+i+|l+m+i+)\b": "nice wonderful delicious great fantastic",
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
    "tl": "Tagalog / Filipino",
    "ceb": "Cebuano / Bisaya",
    "ilo": "Ilocano",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "und": "Detected non-English",
    "en": "English",
}


CEBUANO_PATTERNS = [
    r"\b(p+i+s+t+i+n+g*|p+s+t+n+g*)\s*(y+a+w+a+|y+w+a+)\b",
    r"\b(y+a+w+a+|y+w+a+|y+a+w+a+a+)\b",
    r"\b(p+i+s+t+e+|p+s+t+e+)\b",
    r"\b(b+i+l+a+t+|b+l+t+)\s*s+a*\s*i+n+a+\b",
    r"\b(b+i+l+a+t+|b+l+t+|b+i+l+a+t+i+b+a+y+)\b",
    r"\b(a+t+a+y+|a+t+y+|g+i+a+t+a+y+)\b",
    r"\b(k+a+y+a+t+a+|k+y+t+a+)\b",
    r"\b(b+u+a+n+g+|b+w+a+n+g+|b+n+g+|b+u+a+n+g+-b+u+a+n+g+)\b",
    r"\b(a+m+a+w+|a+m+w+|k+o+l+o+k+o+)\b",
    r"\b(l+a+m+i+|l+m+i+|n+i+n+d+o+t+|n+n+d+t+|n+a*m+i+t+)\b",
    r"\b(s+a+m+o+k+|s+a+m+o+k+a+|h+i+l+a+s+|h+i+l+a+s+a+)\b",
    r"\b(k+a+a+y+o+|k+a+y+o+|p+i+r+t+i+)\b",
    r"\b(g+w+a+p+a+|g+w+a+p+o+)\b",
    r"\b(d+a+g+h+a+n+g*\s*s+a*l+a*m+a*t+)\b",
    r"\b(p+a+s+t+i+l+a+n+)\b",
]

ILOCANO_PATTERNS = [
    r"\b(u+k+i+n+a+m+|u+k+i+n+i+n+a+m+|o+k+i+n+a+m+|u+k+i+n+a+n+a+)\b",
    r"\b(b+a+g+t+i+t+|b+g+t+t+)\b",
    r"\b(b+u+l+s+i+t+|t+o+r+p+e+)\b",
    r"\b(n+a+i+m+b+a+g+|n+a*g+s+a*y+a*a*t+)\b",
    r"\b(n+a*p+i+n+t+a+s+|n+a*g+l+a*i+n+g+)\b",
    r"\b(a+g+y+a*m+a*n+)\b",
]


def _normalize_informal_text(text: str) -> str:
    """
    Normalize informal Philippine texting, SMS shortcuts, leetspeak, and vowel/consonant elongations.
    Preserves casing structure while normalizing substitutions.
    """
    if not text:
        return ""
    t = text
    # Leetspeak within or boundary of words
    t = re.sub(r"(?<=[a-zA-Z0-9])[@4](?=[a-zA-Z0-9])", "a", t)
    t = re.sub(r"^[@4](?=[a-zA-Z0-9])", "a", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])[@4]$", "a", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])[0](?=[a-zA-Z0-9])", "o", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])[1!](?=[a-zA-Z0-9])", "i", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])[3](?=[a-zA-Z0-9])", "e", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])[\$5](?=[a-zA-Z0-9])", "s", t)

    # Collapse 3+ repeated characters down to 1 (e.g., gagooooo -> gago, tanginaaaaa -> tangina)
    t = re.sub(r"([a-zA-Z])\1{2,}", r"\1", t)
    return t


def _apply_dialect_translation(text: str) -> str:
    """Normalize and translate recognized Philippine dialect profanities and sentiment idioms."""
    translated = text
    # First check patterns on original text
    for pattern, replacement in FILIPINO_DIALECT_MAP.items():
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

    # If unchanged, try applying on normalized informal text
    if translated == text:
        norm = _normalize_informal_text(text)
        norm_trans = norm
        for pattern, replacement in FILIPINO_DIALECT_MAP.items():
            norm_trans = re.sub(pattern, replacement, norm_trans, flags=re.IGNORECASE)
        if norm_trans.lower() != norm.lower():
            translated = norm_trans

    return translated


def _matches_dialect_patterns(text: str) -> Optional[str]:
    """Check if text directly matches Philippine dialect sentiment or profanity patterns."""
    norm = _normalize_informal_text(text)
    for pattern in CEBUANO_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE) or re.search(pattern, norm, flags=re.IGNORECASE):
            return "ceb"
    for pattern in ILOCANO_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE) or re.search(pattern, norm, flags=re.IGNORECASE):
            return "ilo"
    for pattern in FILIPINO_DIALECT_MAP:
        if re.search(pattern, text, flags=re.IGNORECASE) or re.search(pattern, norm, flags=re.IGNORECASE):
            return "tl"
    return None


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
            - code: ISO language code (e.g. 'en', 'tl', 'ceb', 'ilo', 'es', 'fr', 'de', 'und')
            - name: Human-readable language name (e.g. 'English', 'Tagalog / Filipino', 'Spanish')
            - is_english: bool indicating if the text is English or treated as English.
    """
    clean_text = (text or "").strip()
    if not clean_text or len(clean_text) < 2:
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

    # Check for direct Philippine dialect idiom / profanity patterns
    dialect_match = _matches_dialect_patterns(clean_text)
    if dialect_match:
        return {
            "code": dialect_match,
            "name": SCRIPT_LANG_MAP.get(dialect_match, "Tagalog / Filipino"),
            "is_english": False,
        }

    # 2. Latin Script: Tokenize and compute hit rate against English vocabulary
    alpha_tokens = [
        re.sub(r"^[^\w]+|[^\w]+$", "", t.lower())
        for t in clean_text.split()
        if any(c.isalpha() for c in t)
    ]
    alpha_tokens = [t for t in alpha_tokens if t]

    hits_vocab = sum(1 for t in alpha_tokens if t in _ENGLISH_VOCABULARY)
    hits_lexicon = sum(1 for t in alpha_tokens if t in _analyzer_instance.lexicon)
    vocab_rate = hits_vocab / len(alpha_tokens) if alpha_tokens else 1.0
    lex_rate = hits_lexicon / len(alpha_tokens) if alpha_tokens else 0.0

    # High English vocabulary confidence
    if vocab_rate >= 0.50:
        return {"code": "en", "name": "English", "is_english": True}

    # Check for specific European or Philippine Latin languages
    latin_lang_hits: Dict[str, int] = {}
    for lang_code, words in LATIN_LANG_STOPWORDS.items():
        count = sum(1 for t in alpha_tokens if t in words or t.strip("'\"`’") in words)
        if count >= 1:
            latin_lang_hits[lang_code] = count

    if latin_lang_hits and (hits_vocab == 0 or vocab_rate < 0.40 or lex_rate < 0.12):
        best_lang = max(latin_lang_hits.items(), key=lambda x: x[1])[0]
        return {
            "code": best_lang,
            "name": SCRIPT_LANG_MAP.get(best_lang, "Detected non-English"),
            "is_english": False,
        }

    # If text has 1-2 words and has no English vocabulary match, check if foreign
    if len(alpha_tokens) < 3 and hits_vocab == 0 and lex_rate == 0:
        return {"code": "und", "name": "Detected non-English", "is_english": False}

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
    Translate text from detected source language to English via the public MyMemory API
    and local dialect normalization. Enforces maximum character cap (1500 chars),
    ~500-char chunking, HTML entity decoding, timeouts, and fail-open error handling.

    Args:
        text: Original text to translate.
        source_lang: Detected source language code (e.g. 'tl', 'ceb', 'ilo', 'es', 'ja', 'und').

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

    # Pre-check dialect translation for local normalization
    dialect_translated = _apply_dialect_translation(to_translate)
    has_dialect_matches = dialect_translated != to_translate

    # If dialect translation resolved the text to English, return directly without network latency/rate limit
    if has_dialect_matches and detect_language(dialect_translated)["is_english"]:
        return {
            "text": dialect_translated,
            "used": True,
            "truncated": truncated,
            "error": None,
        }

    src = source_lang if source_lang and source_lang not in ("und", "auto") else "Autodetect"
    chunks = _split_into_chunks(to_translate, max_chars=CHUNK_MAX_CHARS)
    translated_chunks: List[str] = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    translation_failed = False
    fail_reason = None

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                "q": chunk,
                "langpair": f"{src}|en",
                "de": "vaderSentimentAnalyzer@gmail.com",
            }
            resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT)

            if resp.status_code != 200:
                translation_failed = True
                fail_reason = f"Translation service returned status {resp.status_code}"
                break

            data = resp.json()
            response_data = data.get("responseData") or {}
            translated_text = response_data.get("translatedText")

            # Check for API error status inside JSON payload (e.g. rate limit responseStatus = 403 / 429)
            resp_status = data.get("responseStatus")
            if resp_status and str(resp_status) not in ("200", 200):
                details = data.get("responseDetails") or f"Translation status {resp_status}"
                translation_failed = True
                fail_reason = str(details)
                break

            if not translated_text:
                translation_failed = True
                fail_reason = "Empty translation returned from service."
                break

            clean_trans = html.unescape(translated_text.strip())

            # Guard against known crowd-sourced euphemisms or mistranslations for profane slang
            if has_dialect_matches:
                crowd_euphemisms = (
                    "but you're going to be charged with a felony",
                    "miss na kita",
                    "gathered information",
                )
                if any(eup in clean_trans.lower() for eup in crowd_euphemisms):
                    clean_trans = _apply_dialect_translation(chunk)

            translated_chunks.append(clean_trans)

        except requests.exceptions.Timeout:
            translation_failed = True
            fail_reason = "Translation service timed out."
            break
        except requests.exceptions.RequestException as req_err:
            translation_failed = True
            fail_reason = f"Network error connecting to translation service: {req_err}"
            break
        except Exception as ex:
            translation_failed = True
            fail_reason = f"Translation processing error: {ex}"
            break

    if not translation_failed and translated_chunks:
        full_translation = " ".join(translated_chunks)
        # Apply any lingering dialect phrase mapping to ensure strong valence keywords are preserved
        if has_dialect_matches:
            full_translation = _apply_dialect_translation(full_translation)
            score_full = _analyzer_instance.polarity_scores(full_translation)["compound"]
            score_dialect = _analyzer_instance.polarity_scores(dialect_translated)["compound"]
            if abs(score_dialect) > abs(score_full):
                full_translation = dialect_translated

        return {
            "text": full_translation,
            "used": True,
            "truncated": truncated,
            "error": None,
        }

    # Fallback to local dialect translation if available
    if has_dialect_matches:
        return {
            "text": dialect_translated,
            "used": True,
            "truncated": truncated,
            "error": None,
        }

    return {
        "text": clean_text,
        "used": False,
        "truncated": truncated,
        "error": fail_reason or "Translation service unavailable.",
    }
