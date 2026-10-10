"""Offline regressions: no real credentials, provider requests or production data."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

with patch.dict(os.environ, {}, clear=True):
    import engine_core as engine
    from tools import _d4_utils as d4
    from tools import enrich_cards as cards


def completion(text="정상 번역입니다.", finish="stop", refusal=None):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": text, "refusal": refusal}}]}


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.object(engine, "COLLECT_PHASE", False))
        self.stack.enter_context(patch.object(socket.socket, "connect",
                                             side_effect=AssertionError("Live network forbidden")))
        self.post = self.stack.enter_context(patch.object(engine.requests, "post",
                                            side_effect=AssertionError("Unexpected API call")))
        self.urlopen = self.stack.enter_context(patch.object(d4.urllib.request, "urlopen",
                                               side_effect=AssertionError("Unexpected API call")))

    def provider_reply(self, obj=None, status=200):
        response = MagicMock(ok=status < 400, status_code=status)
        response.json.return_value = completion() if obj is None else obj
        self.post.side_effect = None
        self.post.return_value = response
        return response

    def d4_reply(self, obj=None):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            completion() if obj is None else obj).encode("utf-8")
        self.urlopen.side_effect = None
        self.urlopen.return_value = response


class UtilityTests(OfflineCase):
    def test_import_and_unicode_sentence_split(self):
        self.assertEqual(d4.split_sentences("One. Two! 세 문장。 마지막 문장！"),
                         ["One.", "Two!", "세 문장。", "마지막 문장！"])
        self.assertTrue(d4.extractive_summary("First sentence. Second sentence."))

    def test_completion_rejects_failure_shapes(self):
        for obj in (None, [], {}, {"choices": []}, {"choices": [None]},
                    completion(None), completion("English only"),
                    completion("요약: English only"), completion(finish="length"),
                    completion(finish="content_filter"), completion(refusal="refused")):
            with self.subTest(obj=obj):
                self.assertIsNone(d4.korean_completion(obj))
        self.assertEqual(d4.korean_completion(completion()), "정상 번역입니다.")

    def test_no_key_never_returns_english_as_korean(self):
        self.assertEqual(d4.to_ko("An English summary."), "")
        result = d4.enrich_item({"title": "News", "summary_en": "An English summary."})
        self.assertEqual(result["summary_en"], "An English summary.")
        self.assertEqual(result["summary_ko"], "")
        self.assertEqual(result["summary_ko_status"], "unavailable")
        self.urlopen.assert_not_called()

    def test_disabled_translation_never_calls_provider(self):
        os.environ.update(OPENAI_API_KEY="test-placeholder", QUALI_DISABLE_OPENAI="1")
        self.assertEqual(d4.to_ko("An English summary."), "")
        self.urlopen.assert_not_called()

    def test_korean_source_needs_no_call(self):
        self.assertEqual(d4.to_ko("이미 한국어 요약입니다."), "이미 한국어 요약입니다.")
        self.urlopen.assert_not_called()

    def test_success_and_existing_model_default(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        self.d4_reply()
        self.assertEqual(d4.to_ko(["An English summary."]), "정상 번역입니다.")
        request = self.urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data)["model"], "gpt-4o-mini")
        self.assertEqual(self.urlopen.call_count, 1)

    def test_d4_http_errors_do_not_retry_or_leak_response(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        for status in (401, 429, 500):
            with self.subTest(status=status):
                self.urlopen.reset_mock()
                self.urlopen.side_effect = urllib.error.HTTPError(
                    "https://api.openai.com/v1/chat/completions", status,
                    "secret response text", {}, io.BytesIO(b"private data"))
                with self.assertLogs("tools._d4_utils", level="WARNING") as logs:
                    self.assertEqual(d4.to_ko("English source"), "")
                self.assertEqual(self.urlopen.call_count, 1)
                self.assertNotIn("secret response text", str(logs.output))
                self.assertNotIn("test-placeholder", str(logs.output))

    def test_d4_timeout_and_malformed_json(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        for error in (TimeoutError("private"), urllib.error.URLError("private")):
            self.urlopen.side_effect = error
            with self.assertLogs("tools._d4_utils", level="WARNING"):
                self.assertEqual(d4.to_ko("English source"), "")
        self.d4_reply()
        self.urlopen.return_value.__enter__.return_value.read.return_value = b"not json"
        with self.assertLogs("tools._d4_utils", level="WARNING"):
            self.assertEqual(d4.to_ko("English source"), "")


class CardTests(OfflineCase):
    def test_default_does_not_enable_paid_translation(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        article = {"title": "한국어 제목", "summary_en": "An English summary."}
        self.assertTrue(cards._ensure_summaries(article))
        self.assertEqual(article["summary_ko"], "")
        self.assertEqual(article["summary_ko_status"], "not_requested")
        self.assertFalse(cards._ensure_summaries(article))
        self.urlopen.assert_not_called()

    def test_old_fake_summaries_retained_outside_korean_fields(self):
        article = {"summary_en": "English summary.", "summary_ko": "요약: English summary.",
                   "ko_summary": "English summary.", "summary_kr": "English summary.",
                   "custom": {"keep": True}}
        cards._ensure_summaries(article)
        for field in ("summary_ko", "ko_summary", "summary_kr"):
            self.assertEqual(article[field], "")
            self.assertIn(field, article["summary_ko_rejected"])
        self.assertEqual(article["custom"], {"keep": True})
        self.assertFalse(cards._ensure_summaries(article))

    def test_valid_existing_alias_is_preserved_without_api(self):
        article = {"summary_en": "English summary.", "summary_ko": "English summary.",
                   "summary_kr": "검토한 한국어 요약입니다."}
        cards._ensure_summaries(article, translate=True)
        self.assertEqual(article["summary_ko"], "검토한 한국어 요약입니다.")
        self.assertEqual(article["summary_ko_status"], "existing")
        self.urlopen.assert_not_called()

    def test_list_summary_and_blank_english_field(self):
        article = {"summary_en": ["First sentence.", "Second sentence."]}
        cards._ensure_summaries(article)
        self.assertEqual(article["summary_en"], ["First sentence.", "Second sentence."])
        self.assertEqual(article["summary_ko"], "")
        blank = {"summary_en": "", "description": "Source text."}
        cards._ensure_summaries(blank)
        self.assertEqual(blank["summary_en"], "Source text.")

    def test_source_korean_is_labeled_without_api(self):
        article = {"description": "한국어 원문입니다."}
        cards._ensure_summaries(article, translate=True)
        self.assertEqual(article["summary_ko_status"], "source_korean")
        self.assertEqual(article["summary_ko"], "한국어 원문입니다.")
        self.urlopen.assert_not_called()

    def test_missing_key_and_disabled_have_explicit_status(self):
        article = {"summary_en": "English summary."}
        cards._ensure_summaries(article, translate=True)
        self.assertEqual(article["summary_ko_status"], "missing_key")
        os.environ.update(OPENAI_API_KEY="test-placeholder", QUALI_DISABLE_OPENAI="1")
        cards._ensure_summaries(article, translate=True)
        self.assertEqual(article["summary_ko_status"], "disabled")
        self.urlopen.assert_not_called()

    def test_translation_success_is_reused(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        self.d4_reply()
        article = {"summary_en": "English summary."}
        self.assertTrue(cards._ensure_summaries(article, translate=True))
        self.assertEqual(article["summary_ko_status"], "translated")
        self.assertEqual(article["summary_ko"], "정상 번역입니다.")
        self.assertFalse(cards._ensure_summaries(article, translate=True))
        self.assertEqual(self.urlopen.call_count, 1)

    def test_bad_provider_output_keeps_original(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        for obj in (completion("English response"), completion(finish="length")):
            self.d4_reply(obj)
            article = {"summary_en": "English source."}
            cards._ensure_summaries(article, translate=True)
            self.assertEqual(article["summary_en"], "English source.")
            self.assertEqual(article["summary_ko"], "")
            self.assertEqual(article["summary_ko_status"], "unavailable")

    def test_cli_saves_source_and_reports_korean_separately(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "selected_articles.json"
            source.write_text(json.dumps({"articles": [{"summary_en": "Source."}],
                                          "custom": "preserve"}), encoding="utf-8")
            with patch.multiple(cards, ROOT=root, SRC_SELECTED=source,
                                SRC_KEYWORD=root / "absent.json"):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(cards.main(["--mode", "selection"]), 0)
                result = json.loads(out.getvalue())
            saved = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual(saved["custom"], "preserve")
            self.assertEqual(saved["articles"][0]["summary_ko"], "")
            self.assertEqual(result["korean_available"], 0)
            self.assertFalse(result["translation_requested"])
            self.assertFalse(source.with_suffix(".json.tmp").exists())
        self.urlopen.assert_not_called()

    def test_malformed_document_is_not_saved_as_success(self):
        with patch.object(cards, "_load_json", return_value=({"unrelated": []}, None)), \
                patch.object(cards, "_save_json") as save, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cards.main([]), 2)
            save.assert_not_called()


class EngineTests(OfflineCase):
    def test_missing_key_and_disabled_are_safe_for_body_and_note(self):
        self.assertIsNone(engine._translate_en_to_ko("English source."))
        self.assertTrue(engine._editor_note_from_text("Title", "English source."))
        os.environ.update(OPENAI_API_KEY="test-placeholder", QUALI_DISABLE_OPENAI="1")
        self.assertIsNone(engine._translate_en_to_ko("English source."))
        self.assertTrue(engine._editor_note_from_text("Title", "English source."))
        self.post.assert_not_called()

    def test_collect_mode_blocks_body_and_note(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        engine.COLLECT_PHASE = True
        self.assertIsNone(engine._translate_en_to_ko("English source."))
        self.assertTrue(engine._editor_note_from_text("Title", "English source."))
        self.post.assert_not_called()

    def test_publish_success_and_model_override(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        self.provider_reply()
        self.assertEqual(engine._translate_en_to_ko("English source."), "정상 번역입니다.")
        self.assertEqual(json.loads(self.post.call_args.kwargs["data"])["model"], "gpt-4o")
        self.assertEqual(engine._translate_en_to_ko("English source.", model="configured-model"),
                         "정상 번역입니다.")
        self.assertEqual(json.loads(self.post.call_args.kwargs["data"])["model"], "configured-model")

    def test_engine_http_errors_do_not_retry(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        for status in (401, 429, 500):
            with self.subTest(status=status):
                self.post.reset_mock()
                self.provider_reply(status=status)
                with self.assertLogs("quali", level="WARNING"):
                    self.assertIsNone(engine._translate_en_to_ko("English source."))
                self.assertEqual(self.post.call_count, 1)

    def test_engine_timeout_bad_json_and_incomplete_output(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        self.post.side_effect = engine.requests.Timeout("private details")
        with self.assertLogs("quali", level="WARNING") as logs:
            self.assertIsNone(engine._translate_en_to_ko("English source."))
        self.assertNotIn("private details", str(logs.output))
        response = self.provider_reply()
        response.json.side_effect = ValueError("bad response")
        with self.assertLogs("quali", level="WARNING"):
            self.assertIsNone(engine._translate_en_to_ko("English source."))
        for obj in (completion("English response"), completion(finish="length")):
            self.provider_reply(obj)
            self.assertIsNone(engine._translate_en_to_ko("English source."))

    def test_collect_after_publish_restores_no_ai_mode(self):
        cfg = {"paths": {"log": "unused"}}
        with patch.object(engine, "load_config", return_value=cfg), \
                patch.object(engine, "setup_rotating_logger"), \
                patch.object(engine, "fetch_articles", return_value=[]) as fetch, \
                patch.object(engine, "save_cards"), patch.object(engine, "write_selection_file"):
            engine.job_collect()
        self.assertTrue(engine.COLLECT_PHASE)
        fetch.assert_called_once()
        self.post.assert_not_called()

    def test_collector_does_not_store_english_in_korean_fields(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        engine.COLLECT_PHASE = True
        gate = {"status": "PASS", "score": 90, "evidence": [], "standards": []}
        with patch.object(engine, "_get", return_value="<html></html>"), \
                patch.object(engine, "_pick_top_articles_by_keywords", return_value=["https://example.com/article"]), \
                patch.object(engine, "_fetch_title_desc", return_value=("Title", "An English summary.")), \
                patch.object(engine, "_extract_main_text_from_html", return_value=("An English article.", 19, 1, 0)), \
                patch.object(engine, "quality_gate", return_value=gate), \
                patch.object(engine, "fact_check", return_value=gate):
            articles = engine.fetch_articles({})
        self.assertTrue(articles)
        for article in articles:
            self.assertEqual(article["summary_ko_text"], "")
            self.assertEqual(article["summary_ko"], ["", "", ""])
            self.assertEqual(article["summary_en"][0], "An English summary.")
        self.post.assert_not_called()

    def test_html_title_respects_collect_and_disable_guards(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        article_id = "a" * 32
        markdown = '''<!--ID:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa QG:PASS QS:90 FC:PASS FS:90-->
[퀄리뉴스] AI 뉴스 — 2026-10-10
📰 제목: "English headline"
출처: OpenAI (https://example.com/article)
- KO:
'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "news.md"
            path.write_text(markdown, encoding="utf-8")
            with patch.object(engine, "load_selection", return_value=({article_id}, {})), \
                    patch.object(engine, "_load_selmap_extra", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                engine.COLLECT_PHASE = True
                engine.export_html(str(path), {})
                engine.COLLECT_PHASE = False
                os.environ["QUALI_DISABLE_OPENAI"] = "1"
                engine.export_html(str(path), {})
            self.assertIn("example.com/article", path.with_suffix(".html").read_text())
        self.post.assert_not_called()

    def test_postprocess_failure_does_not_store_full_translation(self):
        os.environ["OPENAI_API_KEY"] = "test-placeholder"
        self.provider_reply(status=429)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "selected.json"
            path.write_text(json.dumps({"articles": [{"selected": True,
                            "title": "English title", "url": "https://example.com", "editor_note": ""}]}))
            with patch.object(engine, "_selection_path", return_value=str(path)), \
                    patch.object(engine, "_get", return_value="<html>Source</html>"), \
                    patch.object(engine, "_extract_main_text_from_html", return_value=("An English article.", 19, 1, 0)), \
                    self.assertLogs("quali", level="INFO"):
                engine.postprocess_selected_for_publish({})
            article = json.loads(path.read_text())["articles"][0]
            self.assertNotIn("full_ko", article)
            self.assertTrue(article["editor_note"])


if __name__ == "__main__":
    unittest.main()
