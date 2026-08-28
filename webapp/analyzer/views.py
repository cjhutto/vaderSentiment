"""
Views and API handlers for the VADER Sentiment Analyzer web application.
Supports direct text analysis, URL content extraction (tweets, articles, web pages),
document parsing (PDF, TXT, MD), image OCR ingestion, batch CSV processing,
measure graphs (radial speedometer, distribution charts, sentence flow), and emoji sentiment graphs.
"""

import csv
import io
import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
import pypdf

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .language import detect_language, translate_to_english

_analyzer = SentimentIntensityAnalyzer()
_sorted_emojis = sorted(_analyzer.emojis.keys(), key=len, reverse=True) if hasattr(_analyzer, "emojis") else []

EXAMPLES = [
    "VADER is smart, handsome, and funny! 🤩✨",
    "VADER is not smart, handsome, nor funny. 🙁",
    "The book was only kind of good. 😐",
    "Today only kinda sux! But I'll get by, lol 😊",
    "Worst customer service ever, completely unacceptable and disgusted! 😡💔",
]

URL_EXAMPLES = [
    {
        "label": "Twitter/X Post Example",
        "url": "https://twitter.com/8Kaal/status/2092847324665876963",
    },
    {
        "label": "Python News & Articles",
        "url": "https://www.python.org/blogs/",
    },
    {
        "label": "Wikipedia: Natural Language Processing",
        "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
    },
]

HISTORY_LIMIT = 12
DEFAULT_REQUEST_TIMEOUT = 8
MAX_EXTRACTED_CHARS = 15000
MAX_BATCH_ROWS = 500

SOCIAL_GENERIC_TEXT = {
    "facebook",
    "instagram",
    "threads",
    "linkedin",
    "linkedin login, sign in",
    "tiktok",
    "reddit",
    "x",
    "youtube",
    "post",
    "video",
    "reel",
    "watch",
    "comments",
    "share",
}


def _domain_matches(domain: str, *hosts: str) -> bool:
    """Return True only for an exact host or one of its subdomains."""
    hostname = domain.partition(":")[0].rstrip(".")
    return any(hostname == host or hostname.endswith(f".{host}") for host in hosts)


def _clean_social_candidate(text: str, platform: str) -> str:
    """Normalize public-preview text and remove common social chrome."""
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", text or "").strip()
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    cleaned = re.sub(
        r"^[\dKMkm.,\s]+(views|reactions|shares|likes)\s*·\s*"
        r"[\dKMkm.,\s]+(views|reactions|shares|likes)\s*\|\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(rf"\s*\|\s*{re.escape(platform)}$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _is_meaningful_social_text(text: str, platform: str = "") -> bool:
    """Reject empty, generic, and authentication-wall text before scoring."""
    cleaned = _clean_social_candidate(text, platform)
    non_url_text = re.sub(r"https?://\S+", "", cleaned).strip()
    if not non_url_text or len(non_url_text) < 3 or not any(c.isalpha() for c in non_url_text):
        return False

    normalized = non_url_text.casefold().strip(" .,:;|-_")
    if normalized in SOCIAL_GENERIC_TEXT:
        return False

    login_wall_markers = (
        f"sign in to {platform.casefold()}",
        f"log in to {platform.casefold()}",
        "log in or sign up",
        "login or sign up",
        "join now to view",
        "create an account to continue",
        "create an account or log in to",
        "see instagram photos and videos",
        "log into facebook",
        "join facebook to connect",
        "connect with friends, family",
        "welcome to linkedin",
        "manage your professional identity",
        "join linkedin",
        "grow your career with linkedin",
        "join millions of professionals",
    )
    preview = normalized[:500]
    return not any(marker in preview for marker in login_wall_markers)


def _social_unavailable(platform: str) -> ValueError:
    return ValueError(
        f"{platform} did not expose public post text. The post may be private, "
        "login-only, deleted, or blocking automated access. Copy the post text "
        "into 'Direct Text', or upload a screenshot in 'Image OCR'."
    )


def _social_fetch_error(platform: str) -> ValueError:
    return ValueError(
        f"Unable to reach {platform} right now. Check the connection and try "
        "again; the platform may also be temporarily unavailable."
    )


def _social_result(
    *,
    text: str,
    title: str,
    url: str,
    platform: str,
    author: str = "",
) -> Dict[str, Any]:
    clean_text = _clean_social_candidate(text, platform)
    if not _is_meaningful_social_text(clean_text, platform):
        raise _social_unavailable(platform)

    result = {
        "text": clean_text[:MAX_EXTRACTED_CHARS],
        "title": title,
        "url": url,
        "source_type": "url",
        "platform": platform,
    }
    if author:
        result["author"] = author
    return result


def classify(compound: float) -> str:
    """
    Classify a VADER compound score into sentiment polarity labels.

    Args:
        compound: The compound sentiment score ranging from -1.0 to +1.0.

    Returns:
        One of 'positive' (>= 0.05), 'negative' (<= -0.05), or 'neutral'.
    """
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def get_mood_data(compound: float) -> Dict[str, Any]:
    """
    Determine emotional mood tier, primary/secondary emojis, and human-friendly explanation.

    Args:
        compound: Compound score from -1.0 to +1.0.

    Returns:
        Dictionary containing tier index (1-5), mood label, main emoji, secondary emojis,
        CSS mood class, and layman description.
    """
    if compound >= 0.60:
        return {
            "tier": 5,
            "label": "Extremely Positive",
            "mood_name": "Enthusiastic & Delighted",
            "emoji": "🤩",
            "secondary_emojis": ["😍", "🎉", "🔥", "🚀"],
            "css_class": "mood-extreme-pos",
            "color": "#059669",
            "description": "High enthusiasm, strong praise, or joyful celebration.",
        }
    elif compound >= 0.05:
        return {
            "tier": 4,
            "label": "Positive",
            "mood_name": "Pleased & Encouraged",
            "emoji": "😊",
            "secondary_emojis": ["🙂", "👍", "✨", "👏"],
            "css_class": "mood-pos",
            "color": "#10b981",
            "description": "Generally favorable, polite, or constructive tone.",
        }
    elif compound > -0.05:
        return {
            "tier": 3,
            "label": "Neutral",
            "mood_name": "Objective & Factual",
            "emoji": "😐",
            "secondary_emojis": ["🧐", "⚖️", "📋", "ℹ️"],
            "css_class": "mood-neu",
            "color": "#64748b",
            "description": "Balanced, descriptive, or informational without strong sentiment.",
        }
    elif compound > -0.60:
        return {
            "tier": 2,
            "label": "Negative",
            "mood_name": "Critical & Concerned",
            "emoji": "🙁",
            "secondary_emojis": ["😕", "👎", "⚠️", "📉"],
            "css_class": "mood-neg",
            "color": "#f97316",
            "description": "Moderate criticism, skepticism, or dissatisfied reaction.",
        }
    else:
        return {
            "tier": 1,
            "label": "Extremely Negative",
            "mood_name": "Distressed & Frustrated",
            "emoji": "😡",
            "secondary_emojis": ["🤬", "💔", "🚨", "💥"],
            "css_class": "mood-extreme-neg",
            "color": "#dc2626",
            "description": "Intense frustration, severe disappointment, or anger.",
        }


def extract_detected_emojis(text: str) -> List[Dict[str, Any]]:
    """
    Extract and score all emojis found within the text against VADER's emoji lexicon.

    Args:
        text: Input text containing Unicode emojis.

    Returns:
        List of emoji dictionaries with emoji char, description, count, and sentiment metrics.
    """
    if not text or not _sorted_emojis:
        return []

    found: Dict[str, Dict[str, Any]] = {}
    remaining = text

    for em in _sorted_emojis:
        if em in remaining:
            count = remaining.count(em)
            desc = _analyzer.emojis.get(em, "emoji")
            scores = _analyzer.polarity_scores(desc)
            compound = scores["compound"]
            label = classify(compound)

            found[em] = {
                "emoji": em,
                "desc": desc,
                "count": count,
                "compound": compound,
                "label": label,
                "pos": scores["pos"],
                "neu": scores["neu"],
                "neg": scores["neg"],
                "compound_pct": round((compound + 1) * 50, 1),
            }
            remaining = remaining.replace(em, "")

    results = list(found.values())
    results.sort(key=lambda x: (x["count"], abs(x["compound"])), reverse=True)
    return results


def get_lexicon_highlights(text: str) -> List[Dict[str, Any]]:
    """
    Identify and extract words in the text that match VADER's sentiment lexicon.

    Args:
        text: The input text to inspect.

    Returns:
        List of dictionaries with 'word', 'score', and 'type' ('positive' or 'negative')
        sorted by absolute sentiment intensity descending.
    """
    if not text:
        return []

    raw_tokens = re.findall(r"[\w']+|[^\s\w]+", text)
    matched: Dict[str, float] = {}

    for token in raw_tokens:
        clean = token.strip().lower()
        if not clean:
            continue

        word_candidate = clean.strip(".,!?;:\"'()[]{}")
        if clean in _analyzer.lexicon:
            matched[clean] = _analyzer.lexicon[clean]
        elif word_candidate and word_candidate in _analyzer.lexicon:
            matched[word_candidate] = _analyzer.lexicon[word_candidate]

    highlights = []
    for word, score in matched.items():
        if abs(score) >= 0.1:
            highlights.append({
                "word": word,
                "score": round(score, 2),
                "type": "positive" if score > 0 else "negative",
            })

    highlights.sort(key=lambda x: abs(x["score"]), reverse=True)
    return highlights[:25]


def analyze_sentences(text: str) -> List[Dict[str, Any]]:
    """
    Split text into constituent sentences and score each sentence individually.

    Args:
        text: The input text.

    Returns:
        List of sentence score dictionaries containing sentence text, compound score,
        and polarity metrics.
    """
    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if len(sentences) <= 1:
        return []

    results = []
    for idx, s in enumerate(sentences[:40]):
        scores = _analyzer.polarity_scores(s)
        label = classify(scores["compound"])
        results.append({
            "index": idx + 1,
            "sentence": s,
            "compound": scores["compound"],
            "label": label,
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"],
            "compound_pct": round((scores["compound"] + 1) * 50, 1),
            "flow_height_pct": round(abs(scores["compound"]) * 100, 1),
        })

    return results


def generate_plain_summary(
    compound: float,
    label: str,
    pos_pct: float,
    neu_pct: float,
    neg_pct: float,
    highlights: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
    emojis: List[Dict[str, Any]],
    translation: Optional[Dict[str, Any]] = None,
    language: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Produce a concise, natural summary of the sentiment findings.
    """
    summary_parts = []

    # Translation context note
    if translation and translation.get("used"):
        lang_name = language.get("name", "non-English text") if language else "non-English text"
        summary_parts.append(f"Scored from an English translation of detected {lang_name}.")
    elif translation and translation.get("warning"):
        summary_parts.append("Note: VADER scored the original text directly because translation was unavailable.")

    # Main tone statement
    summary_parts.append(
        f"Overall tone is {label.title()} with a compound score of {compound:+.4f}."
    )

    # Proportion breakdown
    if neu_pct >= 75.0:
        if compound != 0.0:
            summary_parts.append(
                f"The text is predominantly factual or neutral ({neu_pct}%), with {pos_pct if label == 'positive' else neg_pct}% {label} emotion."
            )
        else:
            summary_parts.append("The text is balanced and objective with no strong positive or negative language.")
    else:
        summary_parts.append(
            f"Sentiment breakdown: {pos_pct}% positive, {neu_pct}% neutral, and {neg_pct}% negative."
        )

    # Key words influencer notice
    if highlights:
        top_words = [f"'{h['word']}' ({h['score']:+})" for h in highlights[:3]]
        summary_parts.append(f"Key influencer words: {', '.join(top_words)}.")

    # Emoji influence
    if emojis:
        emoji_str = " ".join([e["emoji"] for e in emojis[:4]])
        summary_parts.append(f"Detected emojis ({emoji_str}) reinforced the score.")

    # Sentence flow notice
    if len(sentences) > 1:
        strongest = max(sentences, key=lambda s: abs(s["compound"]))
        if abs(strongest["compound"]) >= 0.05:
            summary_parts.append(
                f"Highest emotional polarity in sentence #{strongest['index']} ({strongest['compound']:+.2f})."
            )

    return " ".join(summary_parts)


def score_payload(
    text: str,
    source_type: str = "text",
    source_title: Optional[str] = None,
    source_url: Optional[str] = None,
    translate_non_english: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive sentiment analysis, visual gauge metrics, emoji graphs,
    and plain-English summaries. Supports optional automatic translation for non-English text.
    """
    lang_info = detect_language(text)

    if lang_info["is_english"]:
        scored_text = text
        translation_info = {
            "used": False,
            "truncated": False,
            "provider": None,
            "warning": None,
        }
    elif translate_non_english:
        trans_res = translate_to_english(text, source_lang=lang_info["code"])
        if trans_res["used"]:
            scored_text = trans_res["text"]
            translation_info = {
                "used": True,
                "truncated": trans_res["truncated"],
                "provider": "mymemory",
                "warning": None,
            }
        else:
            scored_text = text
            warn_msg = trans_res.get("error") or "translation service unavailable"
            translation_info = {
                "used": False,
                "truncated": trans_res["truncated"],
                "provider": None,
                "warning": (
                    f"{lang_info['name']} detected, but translation failed ({warn_msg}). "
                    "VADER lexicon is English-only; scores may be near-neutral or inaccurate."
                ),
            }
    else:
        scored_text = text
        translation_info = {
            "used": False,
            "truncated": False,
            "provider": None,
            "warning": (
                f"{lang_info['name']} detected, but translation is disabled. "
                "VADER lexicon is English-only; scores may be near-neutral or inaccurate."
            ),
        }

    scores = _analyzer.polarity_scores(scored_text)
    label = classify(scores["compound"])
    compound = scores["compound"]
    highlights = get_lexicon_highlights(scored_text)
    sentences = analyze_sentences(scored_text)
    detected_emojis = extract_detected_emojis(text)
    mood = get_mood_data(compound)

    pos_pct = round(scores["pos"] * 100, 1)
    neu_pct = round(scores["neu"] * 100, 1)
    neg_pct = round(scores["neg"] * 100, 1)
    compound_pct = round((compound + 1) * 50, 1)

    # Needle angle for speedometer graph: -90deg (at -1.0) to +90deg (at +1.0)
    needle_angle = round(compound * 90.0, 1)

    plain_summary = generate_plain_summary(
        compound=compound,
        label=label,
        pos_pct=pos_pct,
        neu_pct=neu_pct,
        neg_pct=neg_pct,
        highlights=highlights,
        sentences=sentences,
        emojis=detected_emojis,
        translation=translation_info,
        language=lang_info,
    )

    words = re.findall(r"\b\w+\b", text)

    return {
        "text": text,
        "scored_text": scored_text,
        "language": lang_info,
        "translation": translation_info,
        "source_type": source_type,
        "source_title": source_title or "",
        "source_url": source_url or "",
        "word_count": len(words),
        "char_count": len(text),
        "neg": scores["neg"],
        "neu": scores["neu"],
        "pos": scores["pos"],
        "compound": compound,
        "label": label,
        "compound_pct": compound_pct,
        "needle_angle": needle_angle,
        "neg_pct": neg_pct,
        "neu_pct": neu_pct,
        "pos_pct": pos_pct,
        "mood": mood,
        "detected_emojis": detected_emojis,
        "plain_summary": plain_summary,
        "highlights": highlights,
        "sentences": sentences,
        "sentence_count": len(sentences) if sentences else (1 if text.strip() else 0),
    }


def _result_api_payload(result: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """Build the complete JSON + HTML contract used by dynamic result views."""
    return {
        "status": "ok",
        "result": result,
        "result_html": render_to_string("analyzer/_result_panel.html", {"result": result}),
        **extra,
    }


def extract_url_content(url: str) -> Dict[str, Any]:
    """
    Extract readable public text and metadata from supported social posts and
    general web pages. Never return login chrome as sentiment input.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc:
        raise ValueError("Invalid URL format. Please provide a valid web address.")

    domain = parsed.netloc.lower()

    # 1. Twitter / X Posts
    is_twitter = None
    if _domain_matches(domain, "twitter.com", "x.com"):
        is_twitter = re.search(r"^/([^/]+)/status/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
    if is_twitter:
        oembed_endpoint = f"https://publish.twitter.com/oembed?url={urllib.parse.quote(url)}"
        try:
            resp = requests.get(oembed_endpoint, headers={"User-Agent": "Twitterbot/1.0"}, timeout=DEFAULT_REQUEST_TIMEOUT)
            if resp.status_code >= 500:
                raise _social_fetch_error("X / Twitter")
            if resp.status_code == 200:
                data = resp.json()
                soup = BeautifulSoup(data.get("html", ""), "html.parser")
                p = soup.find("p")
                extracted = p.get_text(" ", strip=True) if p else soup.get_text(" ", strip=True)
                author_name = data.get("author_name", is_twitter.group(1))
                return _social_result(
                    text=extracted,
                    title=f"Post by {author_name} (@{is_twitter.group(1)})",
                    author=author_name,
                    url=url,
                    platform="X / Twitter",
                )
        except (TypeError, KeyError, json.JSONDecodeError):
            raise _social_unavailable("X / Twitter")
        except requests.exceptions.RequestException:
            raise _social_fetch_error("X / Twitter")
        raise _social_unavailable("X / Twitter")

    if _domain_matches(domain, "twitter.com", "x.com"):
        raise _social_unavailable("X / Twitter")

    # 2. YouTube Videos & Shorts
    if _domain_matches(domain, "youtube.com", "youtu.be"):
        yt_oembed = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
        try:
            resp = requests.get(yt_oembed, timeout=DEFAULT_REQUEST_TIMEOUT)
            if resp.status_code >= 500:
                raise _social_fetch_error("YouTube")
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", "")
                author = data.get("author_name", "")
                if not _is_meaningful_social_text(title, "YouTube"):
                    raise _social_unavailable("YouTube")
                full_text = f"{title}\n\nVideo by {author}"
                return _social_result(
                    text=full_text,
                    title=f"YouTube: {title} ({author})",
                    author=author,
                    url=url,
                    platform="YouTube",
                )
        except (TypeError, KeyError, json.JSONDecodeError):
            raise _social_unavailable("YouTube")
        except requests.exceptions.RequestException:
            raise _social_fetch_error("YouTube")
        raise _social_unavailable("YouTube")

    # 3. TikTok public posts (official oEmbed endpoint)
    if _domain_matches(domain, "tiktok.com"):
        tiktok_oembed = f"https://www.tiktok.com/oembed?url={urllib.parse.quote(url)}"
        try:
            resp = requests.get(tiktok_oembed, timeout=DEFAULT_REQUEST_TIMEOUT)
            if resp.status_code >= 500:
                raise _social_fetch_error("TikTok")
            if resp.status_code == 200:
                data = resp.json()
                caption = data.get("title", "")
                author = data.get("author_name", "")
                return _social_result(
                    text=caption,
                    title=f"TikTok by {author}" if author else "TikTok post",
                    author=author,
                    url=url,
                    platform="TikTok",
                )
        except (TypeError, KeyError, json.JSONDecodeError):
            raise _social_unavailable("TikTok")
        except requests.exceptions.RequestException:
            raise _social_fetch_error("TikTok")
        raise _social_unavailable("TikTok")

    # 4. Reddit public posts (public JSON representation)
    if _domain_matches(domain, "reddit.com", "redd.it"):
        if _domain_matches(domain, "reddit.com") and "/comments/" not in parsed.path:
            raise _social_unavailable("Reddit")

        reddit_json_url = parsed._replace(
            path=parsed.path.rstrip("/") + ".json",
            query="raw_json=1",
            fragment="",
        ).geturl()
        try:
            resp = requests.get(
                reddit_json_url,
                headers={"User-Agent": "VADERReviewIntelligence/1.0"},
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            if resp.status_code >= 500:
                raise _social_fetch_error("Reddit")
            if resp.status_code == 200:
                payload = resp.json()
                post = payload[0]["data"]["children"][0]["data"]
                post_title = str(post.get("title", "")).strip()
                post_body = str(post.get("selftext", "")).strip()
                full_text = "\n\n".join(part for part in (post_title, post_body) if part)
                author = str(post.get("author", "")).strip()
                return _social_result(
                    text=full_text,
                    title=f"Reddit: {post_title}" if post_title else "Reddit post",
                    author=author,
                    url=url,
                    platform="Reddit",
                )
        except (IndexError, TypeError, KeyError, json.JSONDecodeError):
            raise _social_unavailable("Reddit")
        except requests.exceptions.RequestException:
            raise _social_fetch_error("Reddit")
        raise _social_unavailable("Reddit")

    # 5. Facebook / Instagram / Threads (public social preview metadata only)
    is_meta = _domain_matches(domain, "facebook.com", "fb.watch", "fb.com", "instagram.com", "threads.net")
    if is_meta:
        if _domain_matches(domain, "instagram.com"):
            platform = "Instagram"
        elif _domain_matches(domain, "threads.net"):
            platform = "Threads"
        else:
            platform = "Facebook"

        meta_headers = {
            "User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = requests.get(url, headers=meta_headers, timeout=DEFAULT_REQUEST_TIMEOUT, allow_redirects=True)
            if resp.status_code >= 500:
                raise _social_fetch_error(platform)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                candidates = []

                og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
                if og_desc and og_desc.get("content"):
                    candidates.append(og_desc["content"].strip())

                for prop in ["twitter:description"]:
                    m = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
                    if m and m.get("content"):
                        candidates.append(m["content"].strip())

                cleaned_candidates = []
                for c in candidates:
                    cleaned = _clean_social_candidate(c, platform)
                    if _is_meaningful_social_text(cleaned, platform):
                        cleaned_candidates.append(cleaned)

                # Prioritize untruncated (doesn't end with '...' or '…') and longest text
                def rank_candidate(s):
                    is_trunc = s.endswith("...") or s.endswith("…") or s.endswith(" ...")
                    return (0 if is_trunc else 1, len(s))

                cleaned_candidates.sort(key=rank_candidate, reverse=True)
                clean_text = cleaned_candidates[0] if cleaned_candidates else ""

                if clean_text:
                    return _social_result(
                        text=clean_text,
                        title=f"{platform} public preview",
                        url=url,
                        platform=platform,
                    )
        except requests.exceptions.RequestException:
            raise _social_fetch_error(platform)

        raise _social_unavailable(platform)

    # 6. Standard Web Scraping
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=DEFAULT_REQUEST_TIMEOUT,
            stream=True,
        )
    except requests.exceptions.RequestException as err:
        raise ValueError(f"Unable to connect to URL: {str(err)}")

    if resp.status_code >= 400:
        # Retry with social preview crawler header if 403 or 400
        try:
            resp = requests.get(url, headers={"User-Agent": "Twitterbot/1.0"}, timeout=DEFAULT_REQUEST_TIMEOUT)
        except Exception:
            pass

        if resp.status_code >= 400:
            raise ValueError(f"Server returned HTTP status {resp.status_code} for {url}. The site may require authentication or block automated scraping.")

    raw_content = b""
    for chunk in resp.iter_content(chunk_size=65536):
        raw_content += chunk
        if len(raw_content) > 2 * 1024 * 1024:
            break

    content_type = resp.headers.get("Content-Type", "").lower()
    if "json" in content_type:
        try:
            json_obj = json.loads(raw_content.decode("utf-8", errors="ignore"))
            text = json.dumps(json_obj, indent=2)
            return {
                "text": text[:MAX_EXTRACTED_CHARS],
                "title": f"JSON data from {parsed.netloc}",
                "url": url,
                "source_type": "url",
            }
        except Exception:
            pass

    soup = BeautifulSoup(raw_content, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("meta", property="og:title"):
        title = soup.find("meta", property="og:title").get("content", "").strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text().strip()

    if not title:
        title = f"Page from {parsed.netloc}"

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "noscript", "iframe"]):
        tag.decompose()

    article = soup.find("article") or soup.find("main") or soup.find(class_=re.compile(r"content|post|article|entry", re.I))

    extracted_paragraphs = []
    container = article if article else soup.body if soup.body else soup

    if container:
        for p in container.find_all(["p", "h2", "h3", "li"]):
            p_text = p.get_text().strip()
            if len(p_text) > 20:
                extracted_paragraphs.append(p_text)

    if extracted_paragraphs:
        full_text = "\n\n".join(extracted_paragraphs)
    else:
        full_text = soup.get_text(separator="\n", strip=True)

    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    if not full_text:
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
        if meta_desc and meta_desc.get("content"):
            full_text = meta_desc["content"].strip()

    if not full_text:
        raise ValueError("Could not extract readable text from the provided URL.")

    if _domain_matches(domain, "linkedin.com"):
        if not _is_meaningful_social_text(full_text, "LinkedIn"):
            raise _social_unavailable("LinkedIn")

    return {
        "text": full_text[:MAX_EXTRACTED_CHARS],
        "title": title,
        "url": url,
        "source_type": "url",
    }


def extract_document_text(file_obj, filename: str) -> Dict[str, Any]:
    """
    Extract plaintext from uploaded documents (PDF, TXT, MD, JSON).
    """
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(file_obj)
            extracted_pages = []
            max_pages = min(len(reader.pages), 50)
            for i in range(max_pages):
                page_text = reader.pages[i].extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                raise ValueError("PDF contains no extractable text layer (may be scanned images).")
            return {
                "text": full_text[:MAX_EXTRACTED_CHARS],
                "title": filename,
                "source_type": "document",
            }
        except Exception as err:
            raise ValueError(f"Error parsing PDF document: {str(err)}")

    try:
        content_bytes = file_obj.read()
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1", errors="ignore")

        text = text.strip()
        if not text:
            raise ValueError("Uploaded document is empty.")

        return {
            "text": text[:MAX_EXTRACTED_CHARS],
            "title": filename,
            "source_type": "document",
        }
    except Exception as err:
        raise ValueError(f"Error reading document text: {str(err)}")


def analyze_batch_csv(
    file_obj,
    text_column: Optional[str] = None,
    max_rows: int = MAX_BATCH_ROWS,
) -> Dict[str, Any]:
    """
    Process a CSV or JSON file in batch mode, score each item, and compute aggregate metrics.
    """
    content = file_obj.read()
    try:
        text_data = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text_data = content.decode("latin-1", errors="ignore")

    reader = csv.DictReader(io.StringIO(text_data))
    rows = list(reader)

    if not rows or not reader.fieldnames:
        raise ValueError("CSV file is empty or could not be parsed with header columns.")

    columns = list(reader.fieldnames)

    rating_column = None
    rating_candidates = {
        "rating",
        "ratings",
        "star",
        "stars",
        "star_rating",
        "review_rating",
        "review_score",
    }
    for col in columns:
        normalized_col = col.lower().strip().replace(" ", "_").replace("-", "_")
        if normalized_col in rating_candidates:
            rating_column = col
            break

    if not text_column or text_column not in columns:
        candidate_names = ["text", "tweet", "review", "comment", "message", "content", "body", "feedback", "sentence", "description", "post"]
        selected_col = None
        for col in columns:
            if col.lower().strip() in candidate_names:
                selected_col = col
                break

        if not selected_col:
            max_avg_len = -1
            for col in columns:
                sample_lens = [len(str(r.get(col, ""))) for r in rows[:20]]
                avg_len = sum(sample_lens) / max(len(sample_lens), 1)
                if avg_len > max_avg_len:
                    max_avg_len = avg_len
                    selected_col = col

        text_column = selected_col or columns[0]

    scored_rows = []
    pos_count = 0
    neu_count = 0
    neg_count = 0
    total_compound = 0.0
    rated_rows = 0
    mismatch_count = 0
    rating_buckets: Dict[int, Dict[str, Any]] = {}

    for idx, row in enumerate(rows[:max_rows]):
        cell_text = str(row.get(text_column, "")).strip()
        if not cell_text:
            continue

        scores = _analyzer.polarity_scores(cell_text)
        label = classify(scores["compound"])
        lang = detect_language(cell_text)
        is_non_eng = not lang["is_english"]

        rating_value = None
        rating_label = None
        is_mismatch = None
        if rating_column:
            raw_rating = str(row.get(rating_column, "")).strip()
            rating_match = re.search(r"(?:^|\s)([1-5](?:\.\d+)?)(?:\s|$)", raw_rating)
            if rating_match:
                parsed_rating = float(rating_match.group(1))
                if 1.0 <= parsed_rating <= 5.0:
                    rating_value = parsed_rating
                    rating_label = "positive" if parsed_rating >= 4.0 else "negative" if parsed_rating <= 2.0 else "neutral"
                    is_mismatch = rating_label != label
                    rated_rows += 1
                    if is_mismatch:
                        mismatch_count += 1

                    rating_key = int(round(parsed_rating))
                    bucket = rating_buckets.setdefault(
                        rating_key,
                        {"rating": rating_key, "count": 0, "compound_total": 0.0},
                    )
                    bucket["count"] += 1
                    bucket["compound_total"] += scores["compound"]

        if label == "positive":
            pos_count += 1
        elif label == "negative":
            neg_count += 1
        else:
            neu_count += 1

        total_compound += scores["compound"]

        scored_rows.append({
            "id": idx + 1,
            "text": cell_text,
            "compound": scores["compound"],
            "label": label,
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"],
            "rating": rating_value,
            "rating_label": rating_label,
            "is_mismatch": is_mismatch,
            "language": lang,
            "is_non_english": is_non_eng,
            "raw_row": row,
        })

    total_valid = len(scored_rows)
    if total_valid == 0:
        raise ValueError(f"No valid text found in column '{text_column}'.")

    avg_compound = total_compound / total_valid
    avg_label = classify(avg_compound)
    non_english_count = sum(1 for r in scored_rows if r.get("is_non_english"))

    sorted_rows = sorted(scored_rows, key=lambda x: x["compound"], reverse=True)
    top_positive = sorted_rows[:3]
    top_negative = sorted_rows[-3:] if len(sorted_rows) >= 3 else []
    top_negative.reverse()

    rating_distribution = []
    for rating_key in sorted(rating_buckets):
        bucket = rating_buckets[rating_key]
        rating_distribution.append({
            "rating": rating_key,
            "count": bucket["count"],
            "avg_compound": round(bucket["compound_total"] / bucket["count"], 4),
        })

    return {
        "columns": columns,
        "selected_column": text_column,
        "total_rows": total_valid,
        "pos_count": pos_count,
        "neu_count": neu_count,
        "neg_count": neg_count,
        "pos_pct": round((pos_count / total_valid) * 100, 1),
        "neu_pct": round((neu_count / total_valid) * 100, 1),
        "neg_pct": round((neg_count / total_valid) * 100, 1),
        "avg_compound": round(avg_compound, 4),
        "avg_label": avg_label,
        "rating_column": rating_column,
        "rated_rows": rated_rows,
        "mismatch_count": mismatch_count,
        "match_count": rated_rows - mismatch_count,
        "mismatch_pct": round((mismatch_count / rated_rows) * 100, 1) if rated_rows else None,
        "rating_distribution": rating_distribution,
        "non_english_count": non_english_count,
        "rows": scored_rows,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


@ensure_csrf_cookie
def analyze(request):
    """
    Main entrypoint view for rendering the analyzer page and handling traditional form posts.
    """
    result = None
    batch_result = None
    error = None
    text = ""
    active_tab = "text"

    if request.method == "POST":
        input_type = request.POST.get("input_type", "text")
        active_tab = input_type
        translate_param = request.POST.get("translate_non_english")
        if "has_translate_option" in request.POST:
            translate_non_english = translate_param in ("1", "true", "on", "True")
        else:
            translate_non_english = translate_param not in ("0", "false", "False", "off") if translate_param is not None else True

        try:
            if input_type == "url":
                url = request.POST.get("url", "").strip()
                if not url:
                    error = "Please enter a public webpage or public social post URL to analyze."
                else:
                    extracted = extract_url_content(url)
                    text = extracted["text"]
                    result = score_payload(
                        text,
                        source_type="url",
                        source_title=extracted.get("title"),
                        source_url=extracted.get("url"),
                        translate_non_english=translate_non_english,
                    )

            elif input_type == "image":
                text = request.POST.get("ocr_text", "").strip()
                if not text:
                    error = "No text recognized from the image. Please upload a clearer image or paste text."
                else:
                    result = score_payload(
                        text,
                        source_type="image",
                        source_title="Image OCR Analysis",
                        translate_non_english=translate_non_english,
                    )

            elif input_type == "document":
                if "doc_file" not in request.FILES:
                    error = "Please choose a file (.pdf, .txt, .md, .csv) to analyze."
                else:
                    uploaded = request.FILES["doc_file"]
                    fname = uploaded.name
                    if fname.lower().endswith(".csv"):
                        batch_result = analyze_batch_csv(uploaded)
                    else:
                        extracted = extract_document_text(uploaded, fname)
                        text = extracted["text"]
                        result = score_payload(
                            text,
                            source_type="document",
                            source_title=extracted.get("title"),
                            translate_non_english=translate_non_english,
                        )

            else:
                text = request.POST.get("text", "").strip()
                if not text:
                    error = "Enter some text to analyze."
                else:
                    result = score_payload(
                        text,
                        source_type="text",
                        translate_non_english=translate_non_english,
                    )

            if result:
                history = request.session.get("history", [])
                history_item = {
                    "text": result["text"][:160] + ("..." if len(result["text"]) > 160 else ""),
                    "compound": result["compound"],
                    "label": result["label"],
                    "source_type": result["source_type"],
                    "source_title": result.get("source_title", ""),
                }
                history.insert(0, history_item)
                request.session["history"] = history[:HISTORY_LIMIT]
                request.session.modified = True

        except Exception as ex:
            error = str(ex)

    history = request.session.get("history", [])
    return render(
        request,
        "analyzer/index.html",
        {
            "text": text,
            "result": result,
            "batch_result": batch_result,
            "error": error,
            "active_tab": active_tab,
            "examples": EXAMPLES,
            "url_examples": URL_EXAMPLES,
            "history": history,
        },
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_analyze(request):
    """
    JSON API endpoint to score arbitrary text payload.
    """
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        text = body.get("text", "").strip()
        source_type = body.get("source_type", "text")
        source_title = body.get("source_title", "")
        source_url = body.get("source_url", "")
        translate_non_english = body.get("translate_non_english", True)

        if not text:
            return JsonResponse({"status": "error", "message": "Text content is required."}, status=400)

        result = score_payload(
            text,
            source_type=source_type,
            source_title=source_title,
            source_url=source_url,
            translate_non_english=translate_non_english,
        )

        history = request.session.get("history", [])
        history.insert(0, {
            "text": result["text"][:160] + ("..." if len(result["text"]) > 160 else ""),
            "compound": result["compound"],
            "label": result["label"],
            "source_type": result["source_type"],
            "source_title": result.get("source_title", ""),
        })
        request.session["history"] = history[:HISTORY_LIMIT]
        request.session.modified = True

        return JsonResponse(_result_api_payload(result))
    except Exception as err:
        return JsonResponse({"status": "error", "message": str(err)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def api_analyze_url(request):
    """
    JSON API endpoint to fetch a URL, extract content, and return sentiment analysis.
    """
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        url = body.get("url", "").strip()
        translate_non_english = body.get("translate_non_english", True)
        if not url:
            return JsonResponse({"status": "error", "message": "URL parameter is required."}, status=400)

        extracted = extract_url_content(url)
        result = score_payload(
            extracted["text"],
            source_type="url",
            source_title=extracted.get("title"),
            source_url=extracted.get("url"),
            translate_non_english=translate_non_english,
        )

        history = request.session.get("history", [])
        history.insert(0, {
            "text": result["text"][:160] + ("..." if len(result["text"]) > 160 else ""),
            "compound": result["compound"],
            "label": result["label"],
            "source_type": "url",
            "source_title": extracted.get("title", ""),
        })
        request.session["history"] = history[:HISTORY_LIMIT]
        request.session.modified = True

        return JsonResponse(_result_api_payload(result, extracted=extracted))
    except Exception as err:
        return JsonResponse({"status": "error", "message": str(err)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def api_analyze_document(request):
    """
    JSON API endpoint to parse uploaded document (.pdf, .txt, .md, .csv).
    """
    try:
        if "file" not in request.FILES:
            return JsonResponse({"status": "error", "message": "No file uploaded."}, status=400)

        uploaded = request.FILES["file"]
        fname = uploaded.name.lower()

        if fname.endswith(".csv"):
            column = request.POST.get("column")
            batch_result = analyze_batch_csv(uploaded, text_column=column)
            return JsonResponse({"status": "ok", "is_batch": True, "batch": batch_result})

        translate_param = request.POST.get("translate_non_english")
        translate_non_english = translate_param.lower() in ("true", "1", "on") if translate_param is not None else True

        extracted = extract_document_text(uploaded, uploaded.name)
        result = score_payload(
            extracted["text"],
            source_type="document",
            source_title=extracted.get("title"),
            translate_non_english=translate_non_english,
        )

        history = request.session.get("history", [])
        history.insert(0, {
            "text": result["text"][:160] + ("..." if len(result["text"]) > 160 else ""),
            "compound": result["compound"],
            "label": result["label"],
            "source_type": "document",
            "source_title": extracted.get("title", ""),
        })
        request.session["history"] = history[:HISTORY_LIMIT]
        request.session.modified = True

        return JsonResponse(_result_api_payload(
            result,
            is_batch=False,
            extracted=extracted,
        ))
    except Exception as err:
        return JsonResponse({"status": "error", "message": str(err)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def api_analyze_batch(request):
    """
    JSON API endpoint to score a batch dataset (CSV) with a specific column.
    """
    try:
        if "file" not in request.FILES:
            return JsonResponse({"status": "error", "message": "No CSV file uploaded."}, status=400)

        uploaded = request.FILES["file"]
        column = request.POST.get("column")
        batch_result = analyze_batch_csv(uploaded, text_column=column)
        return JsonResponse({"status": "ok", "batch": batch_result})
    except Exception as err:
        return JsonResponse({"status": "error", "message": str(err)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def export_batch_csv(request):
    """
    Export batch analysis results as a downloadable CSV file.
    """
    try:
        if "file" not in request.FILES:
            return HttpResponse("No file provided for export.", status=400)

        uploaded = request.FILES["file"]
        column = request.POST.get("column")
        batch_result = analyze_batch_csv(uploaded, text_column=column)

        output = io.StringIO()
        fieldnames = list(batch_result["columns"]) + [
            "vader_compound",
            "vader_label",
            "vader_pos",
            "vader_neu",
            "vader_neg",
            "vader_language",
            "vader_rating_sentiment",
            "vader_rating_mismatch",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in batch_result["rows"]:
            row_dict = dict(r["raw_row"])
            row_dict["vader_compound"] = r["compound"]
            row_dict["vader_label"] = r["label"]
            row_dict["vader_pos"] = r["pos"]
            row_dict["vader_neu"] = r["neu"]
            row_dict["vader_neg"] = r["neg"]
            row_dict["vader_language"] = r.get("language", {}).get("name", "English")
            row_dict["vader_rating_sentiment"] = r["rating_label"] or ""
            row_dict["vader_rating_mismatch"] = "yes" if r["is_mismatch"] else "no" if r["is_mismatch"] is False else ""
            writer.writerow(row_dict)

        response = HttpResponse(output.getvalue(), content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="vader_sentiment_results.csv"'
        return response
    except Exception as err:
        return HttpResponse(f"Error exporting CSV: {str(err)}", status=400)


@csrf_exempt
def view_report(request):
    """
    Render the standalone printable executive report.
    Supports GET (via query params or session) and POST.
    """
    text = ""
    source_type = "text"
    source_title = ""
    source_url = ""

    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        source_type = request.POST.get("source_type", "text")
        source_title = request.POST.get("source_title", "")
        source_url = request.POST.get("source_url", "")
        translate_param = request.POST.get("translate_non_english")
    else:
        text = request.GET.get("text", "").strip()
        source_type = request.GET.get("source_type", "text")
        source_title = request.GET.get("source_title", "")
        source_url = request.GET.get("source_url", "")
        translate_param = request.GET.get("translate_non_english")

    translate_non_english = translate_param not in ("0", "false", "False", "off") if translate_param is not None else True

    if not text:
        last = request.session.get("last_result")
        if last:
            result = last
        else:
            text = "VADER sentiment evaluation and multi-modal intelligence assessment."
            result = score_payload(
                text,
                source_type="demo",
                source_title="Sample Assessment",
                translate_non_english=translate_non_english,
            )
    else:
        result = score_payload(
            text,
            source_type=source_type,
            source_title=source_title,
            source_url=source_url,
            translate_non_english=translate_non_english,
        )

    return render(request, "analyzer/report.html", {"result": result})


@csrf_exempt
@require_http_methods(["POST"])
def export_report_json(request):
    """
    Download complete analysis payload as a formatted JSON document.
    """
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        text = body.get("text", "").strip()
        source_type = body.get("source_type", "text")
        source_title = body.get("source_title", "")
        source_url = body.get("source_url", "")
        translate_non_english = body.get("translate_non_english", True)

        if not text:
            return JsonResponse({"status": "error", "message": "Text content required."}, status=400)

        result = score_payload(
            text,
            source_type=source_type,
            source_title=source_title,
            source_url=source_url,
            translate_non_english=translate_non_english,
        )

        response = HttpResponse(
            json.dumps({"report_type": "VADER_Sentiment_Analysis", "data": result}, indent=2),
            content_type="application/json",
        )
        response["Content-Disposition"] = 'attachment; filename="vader_sentiment_report.json"'
        return response
    except Exception as err:
        return HttpResponse(f"Error exporting JSON report: {str(err)}", status=400)


@csrf_exempt
@require_http_methods(["POST"])
def export_report_html(request):
    """
    Download complete self-contained report as an HTML file attachment.
    """
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        text = body.get("text", "").strip()
        source_type = body.get("source_type", "text")
        source_title = body.get("source_title", "")
        source_url = body.get("source_url", "")
        translate_non_english = body.get("translate_non_english", True)

        if not text:
            text = request.POST.get("text", "").strip()
            source_type = request.POST.get("source_type", "text")
            source_title = request.POST.get("source_title", "")
            source_url = request.POST.get("source_url", "")
            translate_param = request.POST.get("translate_non_english")
            translate_non_english = translate_param not in ("0", "false", "False", "off") if translate_param is not None else True

        if not text:
            return HttpResponse("No text provided for report generation.", status=400)

        result = score_payload(
            text,
            source_type=source_type,
            source_title=source_title,
            source_url=source_url,
            translate_non_english=translate_non_english,
        )

        html_content = render(request, "analyzer/report.html", {"result": result}).content
        response = HttpResponse(html_content, content_type="text/html; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="vader_sentiment_report.html"'
        return response
    except Exception as err:
        return HttpResponse(f"Error exporting HTML report: {str(err)}", status=400)
