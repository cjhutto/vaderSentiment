import io
import json
from unittest.mock import MagicMock, patch

import requests
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from bs4 import BeautifulSoup

from .language import SCRIPT_LANG_MAP, detect_language, translate_to_english
from .views import (
    _is_meaningful_social_text,
    analyze_batch_csv,
    analyze_sentences,
    classify,
    extract_detected_emojis,
    extract_document_text,
    extract_url_content,
    get_lexicon_highlights,
    get_mood_data,
    score_payload,
)


class ClassifyTests(TestCase):
    def test_thresholds(self):
        self.assertEqual(classify(0.05), "positive")
        self.assertEqual(classify(0.8), "positive")
        self.assertEqual(classify(0.049), "neutral")
        self.assertEqual(classify(0.0), "neutral")
        self.assertEqual(classify(-0.049), "neutral")
        self.assertEqual(classify(-0.05), "negative")
        self.assertEqual(classify(-0.9), "negative")


class MoodDataTests(TestCase):
    def test_mood_tiers(self):
        m5 = get_mood_data(0.75)
        self.assertEqual(m5["tier"], 5)
        self.assertEqual(m5["emoji"], "🤩")

        m4 = get_mood_data(0.35)
        self.assertEqual(m4["tier"], 4)
        self.assertEqual(m4["emoji"], "😊")

        m3 = get_mood_data(0.0)
        self.assertEqual(m3["tier"], 3)
        self.assertEqual(m3["emoji"], "😐")

        m2 = get_mood_data(-0.35)
        self.assertEqual(m2["tier"], 2)
        self.assertEqual(m2["emoji"], "🙁")

        m1 = get_mood_data(-0.85)
        self.assertEqual(m1["tier"], 1)
        self.assertEqual(m1["emoji"], "😡")


class EmojiExtractionTests(TestCase):
    def test_extract_detected_emojis(self):
        text = "This product is fantastic! 😍🎉 But I was disappointed earlier 😢"
        emojis = extract_detected_emojis(text)
        self.assertGreaterEqual(len(emojis), 2)
        chars = [e["emoji"] for e in emojis]
        self.assertIn("😍", chars)
        self.assertIn("😢", chars)

    def test_no_emojis(self):
        text = "Just standard plaintext without any emojis."
        emojis = extract_detected_emojis(text)
        self.assertEqual(emojis, [])


class LexiconHighlightsTests(TestCase):
    def test_finds_positive_and_negative_words(self):
        text = "This movie is fantastic and wonderful, but the ending was horrible and tragic."
        highlights = get_lexicon_highlights(text)
        words = [h["word"] for h in highlights]
        self.assertIn("fantastic", words)
        self.assertIn("wonderful", words)
        self.assertIn("horrible", words)
        self.assertIn("tragic", words)
        types = {h["word"]: h["type"] for h in highlights}
        self.assertEqual(types["fantastic"], "positive")
        self.assertEqual(types["horrible"], "negative")

    def test_empty_text(self):
        self.assertEqual(get_lexicon_highlights(""), [])


class SentenceBreakdownTests(TestCase):
    def test_multi_sentence_breakdown(self):
        text = "I love this product so much! However, the delivery took way too long and was awful."
        sentences = analyze_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["label"], "positive")
        self.assertEqual(sentences[1]["label"], "negative")

    def test_single_sentence_returns_empty_list(self):
        text = "Just a single sentence here."
        sentences = analyze_sentences(text)
        self.assertEqual(len(sentences), 0)


class ScorePayloadTests(TestCase):
    def test_score_payload_enrichment(self):
        text = "Great job on the project! 😍"
        payload = score_payload(text, source_type="text")
        self.assertEqual(payload["label"], "positive")
        self.assertGreaterEqual(payload["compound"], 0.05)
        self.assertIn("needle_angle", payload)
        self.assertIn("mood", payload)
        self.assertIn("plain_summary", payload)
        self.assertEqual(payload["source_type"], "text")
        self.assertEqual(payload["word_count"], 5)


class ExtractUrlContentTests(TestCase):
    @patch("requests.get")
    def test_twitter_oembed_extraction(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "author_name": "Tech Enthusiast",
            "html": '<blockquote class="twitter-tweet"><p lang="en">VADER sentiment is super fast and lightweight!</p></blockquote>',
        }
        mock_get.return_value = mock_resp

        result = extract_url_content("https://twitter.com/user/status/123456789")
        self.assertIn("super fast and lightweight", result["text"])
        self.assertEqual(result["author"], "Tech Enthusiast")
        self.assertEqual(result["source_type"], "url")

    @patch("requests.get")
    def test_x_status_text_in_an_article_query_does_not_trigger_oembed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.iter_content.return_value = [
            b"<html><head><title>Link Safety Guide</title></head>"
            b"<body><article><p>This article explains how redirect links are reviewed safely.</p></article></body></html>"
        ]
        mock_get.return_value = mock_resp

        result = extract_url_content(
            "https://example.com/guide?next=https://x.com/user/status/123456789"
        )

        self.assertEqual(result["title"], "Link Safety Guide")
        self.assertIn("redirect links are reviewed safely", result["text"])

    @patch("requests.get")
    def test_webpage_article_extraction(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.iter_content.return_value = [
            b"<html><head><title>Exciting Product Launch</title></head><body><article><p>We are delighted to announce our new revolutionary toolkit today.</p></article></body></html>"
        ]
        mock_get.return_value = mock_resp

        result = extract_url_content("https://example.com/blog/launch")
        self.assertEqual(result["title"], "Exciting Product Launch")
        self.assertIn("delighted to announce", result["text"])

    @patch("requests.get")
    def test_facebook_generic_shell_is_not_analyzed_as_post_content(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>Facebook</title></head><body></body></html>"
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.facebook.com/photo/?fbid=123")

    @patch("requests.get")
    def test_facebook_public_preview_uses_meaningful_post_description(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '<html><head><title>Facebook</title>'
            '<meta property="og:description" '
            'content="I absolutely love this thoughtful community project!">'
            "</head><body></body></html>"
        )
        mock_get.return_value = mock_resp

        result = extract_url_content("https://www.facebook.com/posts/123")

        self.assertEqual(result["text"], "I absolutely love this thoughtful community project!")
        self.assertEqual(result["platform"], "Facebook")

    @patch("requests.get")
    def test_facebook_image_alt_without_post_copy_is_not_scored(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '<html><head><title>Facebook</title>'
            '<meta property="og:image:alt" content="A person standing beside a car">'
            "</head><body></body></html>"
        )
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.facebook.com/photo/?fbid=456")

    @patch("requests.get")
    def test_facebook_title_without_description_is_not_scored(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '<html><head><title>Someone shared a post | Facebook</title>'
            '<meta property="og:title" content="Someone shared a post">'
            "</head><body></body></html>"
        )
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.facebook.com/posts/789")

    @patch("requests.get")
    def test_instagram_signup_preview_is_not_scored(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '<html><head><title>Instagram</title>'
            '<meta property="og:description" '
            'content="Create an account or log in to Instagram to see photos and videos.">'
            "</head><body></body></html>"
        )
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.instagram.com/p/example/")

    @patch("requests.get")
    def test_threads_public_preview_uses_meaningful_description(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '<html><head><title>Threads</title>'
            '<meta property="og:description" '
            'content="The support team solved this quickly and I am genuinely grateful.">'
            "</head><body></body></html>"
        )
        mock_get.return_value = mock_resp

        result = extract_url_content("https://www.threads.net/@example/post/abc")

        self.assertEqual(
            result["text"],
            "The support team solved this quickly and I am genuinely grateful.",
        )
        self.assertEqual(result["platform"], "Threads")

    @patch("requests.get")
    def test_social_network_failure_is_not_reported_as_private_content(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("temporary DNS failure")

        with self.assertRaisesRegex(ValueError, "Unable to reach Facebook"):
            extract_url_content("https://www.facebook.com/posts/123")

    @patch("requests.get")
    def test_social_server_failure_is_not_reported_as_private_content(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "Unable to reach Facebook"):
            extract_url_content("https://www.facebook.com/posts/123")

    @patch("requests.get")
    def test_tiktok_public_post_uses_oembed_caption(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "title": "This launch is genuinely brilliant and surprisingly useful!",
            "author_name": "Helpful Creator",
            "author_url": "https://www.tiktok.com/@helpfulcreator",
            "html": '<blockquote class="tiktok-embed"></blockquote>',
            "thumbnail_url": "https://example.com/thumb.jpg",
        }
        mock_get.return_value = mock_resp

        result = extract_url_content("https://www.tiktok.com/@helpfulcreator/video/123456789")

        self.assertEqual(result["text"], "This launch is genuinely brilliant and surprisingly useful!")
        self.assertEqual(result["author"], "Helpful Creator")
        self.assertEqual(result["platform"], "TikTok")

    @patch("requests.get")
    def test_youtube_empty_oembed_metadata_is_not_scored(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "title": "",
            "author_name": "",
            "author_url": "",
            "type": "video",
        }
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.youtube.com/watch?v=missing")

    @patch("requests.get")
    def test_reddit_public_post_uses_json_title_and_body(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "A wonderful improvement",
                                "selftext": "The new release is faster, clearer, and much easier to use.",
                                "author": "sample_user",
                                "permalink": "/r/example/comments/abc/a_wonderful_improvement/",
                            },
                        }
                    ]
                },
            }
        ]
        mock_get.return_value = mock_resp

        result = extract_url_content("https://www.reddit.com/r/example/comments/abc/a_wonderful_improvement/")

        self.assertEqual(
            result["text"],
            "A wonderful improvement\n\nThe new release is faster, clearer, and much easier to use.",
        )
        self.assertEqual(result["author"], "sample_user")
        self.assertEqual(result["platform"], "Reddit")

    @patch("requests.get")
    def test_reddit_feed_is_not_misrepresented_as_one_post(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "Unrelated first feed item",
                                "selftext": "This is not the URL the user selected.",
                                "author": "someone_else",
                            },
                        }
                    ]
                },
            }
        ]
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.reddit.com/r/example/")

    @patch("requests.get")
    def test_linkedin_login_wall_is_not_analyzed_as_content(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.iter_content.return_value = [
            b"<html><head><title>LinkedIn Login, Sign in</title></head>"
            b"<body><main><p>Sign in to LinkedIn to continue to this post.</p></main></body></html>"
        ]
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.linkedin.com/posts/example-123")

    @patch("requests.get")
    def test_linkedin_marketing_shell_is_not_analyzed_as_content(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.iter_content.return_value = [
            b"<html><head><title>LinkedIn</title></head><body><main>"
            b"<p>Grow your career with LinkedIn. Join millions of professionals today.</p>"
            b"</main></body></html>"
        ]
        mock_get.return_value = mock_resp

        with self.assertRaisesRegex(ValueError, "public post text"):
            extract_url_content("https://www.linkedin.com/posts/example-456")

    def test_invalid_url_raises_error(self):
        with self.assertRaises(ValueError):
            extract_url_content("ftp://")


class ExtractDocumentTextTests(TestCase):
    def test_txt_extraction(self):
        content = b"This is a test text file with great positive vibes."
        file_obj = io.BytesIO(content)
        result = extract_document_text(file_obj, "test.txt")
        self.assertEqual(result["text"], content.decode("utf-8"))
        self.assertEqual(result["title"], "test.txt")

    def test_empty_document_raises_error(self):
        file_obj = io.BytesIO(b"   ")
        with self.assertRaises(ValueError):
            extract_document_text(file_obj, "empty.txt")


class BatchCsvTests(TestCase):
    def test_analyze_batch_csv(self):
        csv_data = (
            "id,review,rating\n"
            "1,This is absolutely amazing and fantastic!,5\n"
            "2,The service was terrible and disappointing.,1\n"
            "3,The package arrived on Tuesday.,3\n"
        ).encode("utf-8")
        file_obj = io.BytesIO(csv_data)
        batch = analyze_batch_csv(file_obj)

        self.assertEqual(batch["total_rows"], 3)
        self.assertEqual(batch["selected_column"], "review")
        self.assertEqual(batch["pos_count"], 1)
        self.assertEqual(batch["neg_count"], 1)
        self.assertEqual(batch["neu_count"], 1)
        self.assertEqual(len(batch["rows"]), 3)

    def test_compares_vader_sentiment_with_star_ratings(self):
        csv_data = (
            "id,review,rating\n"
            "1,I absolutely love this product!,5\n"
            "2,I hate this terrible product.,5\n"
            "3,The package arrived on Tuesday.,3\n"
            "4,Excellent quality and wonderful service!,1\n"
        ).encode("utf-8")

        batch = analyze_batch_csv(io.BytesIO(csv_data))

        self.assertEqual(batch["rating_column"], "rating")
        self.assertEqual(batch["rated_rows"], 4)
        self.assertEqual(batch["mismatch_count"], 2)
        self.assertEqual(batch["mismatch_pct"], 50.0)
        self.assertFalse(batch["rows"][0]["is_mismatch"])
        self.assertTrue(batch["rows"][1]["is_mismatch"])
        self.assertEqual(batch["rows"][2]["rating_label"], "neutral")
        self.assertTrue(batch["rows"][3]["is_mismatch"])

    def test_ignores_rating_comparison_when_dataset_has_no_rating_column(self):
        csv_data = (
            "id,review\n"
            "1,This is absolutely amazing!\n"
            "2,This is completely awful.\n"
        ).encode("utf-8")

        batch = analyze_batch_csv(io.BytesIO(csv_data))

        self.assertIsNone(batch["rating_column"])
        self.assertEqual(batch["rated_rows"], 0)
        self.assertEqual(batch["mismatch_count"], 0)
        self.assertIsNone(batch["mismatch_pct"])


class AnalyzeViewTests(TestCase):
    def test_get_renders_form_and_tabs(self):
        response = self.client.get(reverse("analyze"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Review Analysis")
        self.assertContains(response, "Analyze via URL")
        self.assertContains(response, "Image OCR")
        self.assertContains(response, "Docs &amp; CSV Batch")
        self.assertContains(response, "Analyze customer reviews with VADER")
        self.assertContains(response, "Rating mismatch analysis")

    def test_file_inputs_have_accessible_names(self):
        response = self.client.get(reverse("analyze"))
        document = BeautifulSoup(response.content, "html.parser")

        self.assertEqual(document.select_one("#image-file-input").get("aria-label"), "Choose image for text recognition")
        self.assertEqual(document.select_one("#doc-file-input").get("aria-label"), "Choose document or CSV dataset")

    def test_empty_text_post_shows_error(self):
        response = self.client.post(reverse("analyze"), {"input_type": "text", "text": "   "})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter some text to analyze.")

    def test_api_analyze_json_with_emojis(self):
        response = self.client.post(
            reverse("api_analyze"),
            json.dumps({"text": "Antigravity pairing is simply wonderful! 🤩🚀"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["result"]["label"], "positive")
        self.assertEqual(data["result"]["mood"]["tier"], 5)
        self.assertGreaterEqual(len(data["result"]["detected_emojis"]), 1)

    def test_api_analyze_returns_complete_result_fragment_for_dynamic_rendering(self):
        response = self.client.post(
            reverse("api_analyze"),
            json.dumps({
                "text": "I absolutely love this product. It works perfectly!",
                "source_type": "image",
                "source_title": "Extracted Image Text",
            }),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        fragment = BeautifulSoup(data["result_html"], "html.parser")
        self.assertEqual(fragment.select_one("#verdict-label-text").get_text(strip=True), "Positive")
        self.assertEqual(fragment.select_one("#result-source-badge").get_text(" ", strip=True), "Source: Image OCR")
        self.assertIn("I absolutely love this product", fragment.select_one(".analyzed-text").get_text())
        self.assertIsNotNone(fragment.select_one("#btn-export-json"))

    @patch("analyzer.views.extract_url_content")
    def test_api_analyze_url(self, mock_extract):
        mock_extract.return_value = {
            "text": "The release went smoothly and customers are very happy.",
            "title": "Release Notes",
            "url": "https://example.com/release",
            "source_type": "url",
        }
        response = self.client.post(
            reverse("api_analyze_url"),
            json.dumps({"url": "https://example.com/release"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["result"]["label"], "positive")
        fragment = BeautifulSoup(data["result_html"], "html.parser")
        self.assertEqual(fragment.select_one("#result-source-badge").get_text(" ", strip=True), "Source: Web / URL")
        self.assertEqual(fragment.select_one(".source-title-display").get_text(strip=True), "Release Notes")

    @patch("requests.get")
    def test_api_blocked_social_url_returns_guidance_without_score(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>Facebook</title></head><body></body></html>"
        mock_get.return_value = mock_resp

        response = self.client.post(
            reverse("api_analyze_url"),
            json.dumps({"url": "https://www.facebook.com/posts/private"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Direct Text", data["message"])
        self.assertIn("Image OCR", data["message"])
        self.assertNotIn("result", data)

    @patch("requests.get")
    def test_form_blocked_social_url_shows_guidance_without_score(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>Facebook</title></head><body></body></html>"
        mock_get.return_value = mock_resp

        response = self.client.post(
            reverse("analyze"),
            {
                "input_type": "url",
                "url": "https://www.facebook.com/posts/private",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Direct Text")
        self.assertContains(response, "Image OCR")
        self.assertIsNone(response.context["result"])

    def test_api_analyze_document_txt(self):
        uploaded_file = SimpleUploadedFile("sample.txt", b"Everything is awesome and brilliant!")
        response = self.client.post(
            reverse("api_analyze_document"),
            {"file": uploaded_file},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["result"]["label"], "positive")
        fragment = BeautifulSoup(data["result_html"], "html.parser")
        self.assertEqual(fragment.select_one("#result-source-badge").get_text(" ", strip=True), "Source: Document")
        self.assertEqual(fragment.select_one(".source-title-display").get_text(strip=True), "sample.txt")

    def test_export_batch_csv(self):
        csv_bytes = b"text\nI adore this app\nI hate bugs"
        uploaded_file = SimpleUploadedFile("dataset.csv", csv_bytes)
        response = self.client.post(
            reverse("export_batch_csv"),
            {"file": uploaded_file},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn(b"vader_compound", response.content)
        self.assertIn(b"vader_rating_sentiment", response.content)
        self.assertIn(b"vader_rating_mismatch", response.content)
        self.assertIn(b"positive", response.content)
        self.assertIn(b"negative", response.content)

    def test_view_report_get(self):
        response = self.client.get(reverse("view_report"), {"text": "Super happy with the new features! 🤩🎉"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "VADER Sentiment Analysis Report")
        self.assertContains(response, "Positive")
        self.assertContains(response, "Radial Gauge")
        self.assertContains(response, "Emoji Sentiment Graph")

    def test_export_report_html(self):
        response = self.client.post(
            reverse("export_report_html"),
            json.dumps({"text": "Excellent speed and wonderful performance! 🚀"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/html; charset=utf-8")
        self.assertIn('attachment; filename="vader_sentiment_report.html"', response["Content-Disposition"])
        self.assertContains(response, "VADER Sentiment Analysis Report")

    def test_export_report_json(self):
        response = self.client.post(
            reverse("export_report_json"),
            json.dumps({"text": "Impressive release! 🎉", "source_type": "text"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertIn('attachment; filename="vader_sentiment_report.json"', response["Content-Disposition"])
        data = json.loads(response.content.decode("utf-8"))
        self.assertEqual(data["report_type"], "VADER_Sentiment_Analysis")
        self.assertEqual(data["data"]["label"], "positive")
        self.assertIn("needle_angle", data["data"])
        self.assertIn("mood", data["data"])
        self.assertIn("detected_emojis", data["data"])


class LanguageDetectionUnitTests(TestCase):
    def test_english_detection(self):
        self.assertEqual(detect_language("This is a great product and I really love using it every day.")["code"], "en")
        self.assertEqual(detect_language("Good morning, how are you?")["code"], "en")
        self.assertEqual(detect_language("Terrible customer service and broken item.")["code"], "en")
        self.assertEqual(detect_language("")["code"], "en")
        self.assertEqual(detect_language("   12345 !!! ???   ")["code"], "en")

    def test_spanish_detection(self):
        res = detect_language("El hotel es muy bueno y el servicio fue excelente.")
        self.assertEqual(res["code"], "es")
        self.assertEqual(res["name"], "Spanish")

    def test_french_detection(self):
        res = detect_language("Ce film est absolument magnifique et très captivant.")
        self.assertEqual(res["code"], "fr")
        self.assertEqual(res["name"], "French")

    def test_german_detection(self):
        res = detect_language("Das Wetter heute ist wirklich wunderbar und schön.")
        self.assertEqual(res["code"], "de")
        self.assertEqual(res["name"], "German")

    def test_japanese_detection(self):
        res = detect_language("この映画はとても面白くて感動しました。")
        self.assertEqual(res["code"], "ja")
        self.assertEqual(res["name"], "Japanese")

    def test_chinese_detection(self):
        res = detect_language("这个餐厅的服务非常周到，菜品很好吃。")
        self.assertEqual(res["code"], "zh")
        self.assertEqual(res["name"], "Chinese")

    def test_arabic_detection(self):
        res = detect_language("هذا المطعم رائع جدا والخدمة ممتازة والأسعار مناسبة")
        self.assertEqual(res["code"], "ar")
        self.assertEqual(res["name"], "Arabic")

    def test_russian_detection(self):
        res = detect_language("Это был великолепный вечер и замечательный концерт.")
        self.assertEqual(res["code"], "ru")
        self.assertEqual(res["name"], "Russian")


class TranslateToEnglishUnitTests(TestCase):
    def test_english_skips_translation(self):
        with patch("requests.get") as mock_get:
            res = translate_to_english("Hello world", "en")
            self.assertEqual(res["text"], "Hello world")
            self.assertFalse(res["used"])
            self.assertFalse(res["truncated"])
            self.assertIsNone(res["error"])
            mock_get.assert_not_called()

    @patch("requests.get")
    def test_successful_translation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "This is a wonderful product."},
        }
        mock_get.return_value = mock_resp

        res = translate_to_english("Este es un producto maravilloso.", "es")
        self.assertEqual(res["text"], "This is a wonderful product.")
        self.assertTrue(res["used"])
        self.assertFalse(res["truncated"])
        self.assertIsNone(res["error"])
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_long_text_truncated(self, mock_get):
        long_text = "magnifique " * 200  # > 2000 chars
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "magnificent"},
        }
        mock_get.return_value = mock_resp

        res = translate_to_english(long_text, "fr")
        self.assertTrue(res["truncated"])
        self.assertTrue(res["used"])
        self.assertIsNone(res["error"])

    @patch("requests.get")
    def test_translation_network_failure_fail_open(self, mock_get):
        mock_get.side_effect = requests.RequestException("Network unreachable")

        orig = "Este es un texto en español."
        res = translate_to_english(orig, "es")
        self.assertEqual(res["text"], orig)
        self.assertFalse(res["used"])
        self.assertFalse(res["truncated"])
        self.assertIsNotNone(res["error"])

    @patch("requests.get")
    def test_translation_api_error_status(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 429,
            "responseDetails": "MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS TODAY",
        }
        mock_get.return_value = mock_resp

        orig = "Bonjour tout le monde."
        res = translate_to_english(orig, "fr")
        self.assertEqual(res["text"], orig)
        self.assertFalse(res["used"])
        self.assertIsNotNone(res["error"])


class ScorePayloadTranslationTests(TestCase):
    def test_score_payload_english(self):
        with patch("requests.get") as mock_get:
            result = score_payload("I love this software! It is fantastic.", translate_non_english=True)
            self.assertEqual(result["language"]["code"], "en")
            self.assertEqual(result["language"]["name"], "English")
            self.assertTrue(result["language"]["is_english"])
            self.assertFalse(result["translation"]["used"])
            self.assertEqual(result["scored_text"], result["text"])
            self.assertEqual(result["label"], "positive")
            mock_get.assert_not_called()

    @patch("requests.get")
    def test_score_payload_spanish_translated_and_scored(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "The food was delicious and the service was wonderful."},
        }
        mock_get.return_value = mock_resp

        spanish_text = "La comida estuvo deliciosa y el servicio fue maravilloso. 😍"
        result = score_payload(spanish_text, translate_non_english=True)

        self.assertEqual(result["language"]["code"], "es")
        self.assertFalse(result["language"]["is_english"])
        self.assertTrue(result["translation"]["used"])
        self.assertEqual(result["translation"]["provider"], "mymemory")
        self.assertIn("delicious", result["scored_text"])
        self.assertEqual(result["label"], "positive")
        self.assertGreater(result["compound"], 0.5)
        # Original text preserved
        self.assertEqual(result["text"], spanish_text)
        # Emojis extracted from original text
        self.assertEqual(len(result["detected_emojis"]), 1)
        self.assertEqual(result["detected_emojis"][0]["emoji"], "😍")

    @patch("requests.get")
    def test_score_payload_translate_disabled(self, mock_get):
        spanish_text = "La comida estuvo deliciosa y el servicio fue maravilloso."
        result = score_payload(spanish_text, translate_non_english=False)

        self.assertFalse(result["translation"]["used"])
        self.assertEqual(result["scored_text"], spanish_text)
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_score_payload_translation_failure_fallback(self, mock_get):
        mock_get.side_effect = requests.RequestException("Timeout")
        spanish_text = "El servicio fue terrible y pésimo."
        result = score_payload(spanish_text, translate_non_english=True)

        self.assertFalse(result["translation"]["used"])
        self.assertIsNotNone(result["translation"]["warning"])
        self.assertEqual(result["scored_text"], spanish_text)
        self.assertIn("VADER scored the original text directly", result["plain_summary"])


class MeaningfulSocialTextTests(TestCase):
    def test_latin_letters(self):
        self.assertTrue(_is_meaningful_social_text("Great post about python!"))

    def test_japanese_letters(self):
        self.assertTrue(_is_meaningful_social_text("素晴らしいポストです！"))

    def test_arabic_letters(self):
        self.assertTrue(_is_meaningful_social_text("منشور رائع جدا"))

    def test_cyrillic_letters(self):
        self.assertTrue(_is_meaningful_social_text("Отличный пост"))

    def test_empty_and_noise_only(self):
        self.assertFalse(_is_meaningful_social_text(""))
        self.assertFalse(_is_meaningful_social_text("   --- !!! ??? ===   "))
        self.assertFalse(_is_meaningful_social_text("https://example.com/some/link"))


class ViewAndAPITranslationTests(TestCase):
    @patch("requests.get")
    def test_api_analyze_with_translation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "I absolutely love this experience!"},
        }
        mock_get.return_value = mock_resp

        response = self.client.post(
            reverse("api_analyze"),
            json.dumps({"text": "J'adore absolument cette expérience !", "translate_non_english": True}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["result"]["translation"]["used"])
        self.assertEqual(data["result"]["language"]["code"], "fr")
        self.assertEqual(data["result"]["label"], "positive")

    @patch("requests.get")
    def test_analyze_form_post_spanish_renders_banner(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "This place is truly amazing and great."},
        }
        mock_get.return_value = mock_resp

        response = self.client.post(
            reverse("analyze"),
            {"text": "Este lugar es realmente increíble y genial.", "translate_non_english": "1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Translated from Spanish")
        self.assertContains(response, "via MyMemory Translation")

    def test_batch_csv_non_english_detect_only(self):
        csv_bytes = b"review\nI love this product\nEl producto es muy bueno\nTerrible quality"
        uploaded_file = SimpleUploadedFile("reviews.csv", csv_bytes)

        with patch("requests.get") as mock_get:
            batch = analyze_batch_csv(uploaded_file, "review")
            self.assertEqual(batch["total_rows"], 3)
            self.assertEqual(batch["non_english_count"], 1)
            # Find the Spanish row
            non_en_rows = [r for r in batch["rows"] if r.get("is_non_english")]
            self.assertEqual(len(non_en_rows), 1)
            self.assertEqual(non_en_rows[0]["language"]["code"], "es")
            # Verify no network requests were made during batch CSV
            mock_get.assert_not_called()

    @patch("requests.get")
    def test_japanese_arabic_translation_scoring(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "This anime is awesome and fantastic!"},
        }
        mock_get.return_value = mock_resp

        result = score_payload("このアニメは本当に素晴らしいです！", translate_non_english=True)
        self.assertEqual(result["language"]["code"], "ja")
        self.assertEqual(result["language"]["name"], "Japanese")
        self.assertTrue(result["translation"]["used"])
        self.assertEqual(result["label"], "positive")

    @patch("requests.get")
    def test_long_text_truncation_in_score_payload(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "great content"},
        }
        mock_get.return_value = mock_resp

        long_spanish = "Excelente servicio y comida muy rica. " * 50  # ~1900 chars
        result = score_payload(long_spanish, translate_non_english=True)
        self.assertTrue(result["translation"]["used"])
        self.assertTrue(result["translation"]["truncated"])

    def test_toggle_off_form_post(self):
        with patch("requests.get") as mock_get:
            response = self.client.post(
                reverse("analyze"),
                {
                    "text": "El hotel fue pésimo y sucio.",
                    "has_translate_option": "1",
                    # translate_non_english omitted -> checkbox unchecked
                },
            )
            self.assertEqual(response.status_code, 200)
            mock_get.assert_not_called()

    @patch("requests.get")
    def test_view_report_with_translation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "The view is absolutely spectacular and breathtaking."},
        }
        mock_get.return_value = mock_resp

        response = self.client.post(
            reverse("view_report"),
            {
                "text": "La vista es absolutamente espectacular e impresionante.",
                "translate_non_english": "1",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Translated from Spanish")
        self.assertContains(response, "via MyMemory Translation")
        self.assertContains(response, "Scored English Translation")

    @patch("requests.get")
    def test_export_report_json_with_translation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "Everything was perfect and wonderful."},
        }
        mock_get.return_value = mock_resp

        response = self.client.post(
            reverse("export_report_json"),
            json.dumps({
                "text": "Tout était parfait et merveilleux.",
                "translate_non_english": True,
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content.decode("utf-8"))
        self.assertTrue(data["data"]["translation"]["used"])
        self.assertEqual(data["data"]["language"]["code"], "fr")
        self.assertEqual(data["data"]["label"], "positive")
