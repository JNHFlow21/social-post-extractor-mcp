import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from douyin_mcp_server.social_extractor import (
    ExtractionContext,
    OpenAICompatibleASRProvider,
    SocialExtractorService,
    SocialPost,
    build_volcengine_audio_request,
    build_volcengine_full_client_request,
    XHSStateParser,
    provider_asr_config,
    provider_volcengine_speech_config,
)


class FakePlatformAdapter:
    def __init__(self, post: SocialPost):
        self.post = post

    def can_handle(self, share_text: str) -> bool:
        return True

    def fetch_post(self, share_text: str) -> SocialPost:
        return self.post


class FakeAsrProvider:
    def __init__(self, transcript: str):
        self.transcript = transcript

    def transcribe(self, post: SocialPost, context: ExtractionContext) -> str:
        return self.transcript


class FakeCleanupProvider:
    def __init__(self, cleaned: str):
        self.cleaned = cleaned

    def cleanup(self, post: SocialPost, raw_transcript: str | None, image_texts: list[str], context: ExtractionContext) -> str:
        return self.cleaned


class FailingVisionProvider:
    def read_image_text(self, image_url: str, context: ExtractionContext, image_index: int) -> str:
        raise RuntimeError("vision failed")


class FakeOcrProvider:
    def read_image_text(self, image_url: str, context: ExtractionContext, image_index: int) -> str:
        return f"OCR:{image_index}:{image_url.rsplit('/', 1)[-1]}"


class FailingAsrProvider:
    def transcribe(self, post: SocialPost, context: ExtractionContext) -> str:
        raise RuntimeError("asr failed")


class SocialExtractorServiceTests(unittest.TestCase):
    def test_default_asr_registry_includes_bailian_and_doubao(self):
        service = SocialExtractorService(
            platform_adapters=[],
            cleanup_providers={},
            vision_providers={},
            ocr_provider=FakeOcrProvider(),
        )

        self.assertIn("bailian", service.asr_providers)
        self.assertIn("doubao", service.asr_providers)
        self.assertIn("volcengine_speech", service.asr_providers)

    def test_video_post_writes_script_and_info_files(self):
        post = SocialPost(
            platform="douyin",
            content_type="video",
            source_url="https://v.douyin.com/demo/",
            resolved_url="https://www.iesdouyin.com/share/video/123",
            post_id="123",
            title="测试抖音视频",
            body="视频简介",
            author_name="作者A",
            video_url="https://cdn.example.com/video.mp4",
            image_urls=["https://cdn.example.com/cover.jpg"],
        )
        service = SocialExtractorService(
            platform_adapters=[FakePlatformAdapter(post)],
            asr_providers={"siliconflow": FakeAsrProvider("原始转写内容")},
            cleanup_providers={"qwen": FakeCleanupProvider("整理后的脚本")},
            vision_providers={},
            ocr_provider=FakeOcrProvider(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = service.extract_social_post(
                "https://v.douyin.com/demo/",
                output_dir=tmpdir,
                asr_provider="siliconflow",
                clean_provider="qwen",
                clean_model="qwen-plus",
            )

            script_path = Path(result["script_path"])
            info_path = Path(result["info_path"])
            self.assertTrue(script_path.exists())
            self.assertTrue(info_path.exists())

            script_text = script_path.read_text(encoding="utf-8")
            self.assertIn("## 整理稿", script_text)
            self.assertIn("整理后的脚本", script_text)
            self.assertIn("## 原始转写", script_text)
            self.assertIn("原始转写内容", script_text)
            self.assertIn("## 原笔记正文", script_text)
            self.assertIn("视频简介", script_text)

            info = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(info["platform"], "douyin")
            self.assertEqual(info["content_type"], "video")
            self.assertEqual(info["post_id"], "123")
            self.assertEqual(info["asr_provider"], "siliconflow")
            self.assertEqual(info["clean_provider"], "qwen")
            self.assertEqual(info["clean_model"], "qwen-plus")
            self.assertEqual(result["script_preview"], "整理后的脚本")

    def test_image_note_falls_back_to_ocr(self):
        post = SocialPost(
            platform="xiaohongshu",
            content_type="image_note",
            source_url="https://www.xiaohongshu.com/explore/demo",
            resolved_url="https://www.xiaohongshu.com/explore/demo?xsec_token=abc",
            post_id="note-1",
            title="测试图文笔记",
            body="正文内容",
            author_name="作者B",
            image_urls=[
                "https://img.example.com/1.jpg",
                "https://img.example.com/2.jpg",
            ],
        )
        service = SocialExtractorService(
            platform_adapters=[FakePlatformAdapter(post)],
            asr_providers={},
            cleanup_providers={},
            vision_providers={"qwen-vl": FailingVisionProvider()},
            ocr_provider=FakeOcrProvider(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = service.extract_social_post(
                "https://www.xiaohongshu.com/explore/demo",
                output_dir=tmpdir,
                vision_provider="qwen-vl",
            )

            script_text = Path(result["script_path"]).read_text(encoding="utf-8")
            self.assertIn("## 图片文字提取", script_text)
            self.assertIn("OCR:1:1.jpg", script_text)
            self.assertIn("OCR:2:2.jpg", script_text)
            self.assertIn("## 整理稿", script_text)
            self.assertEqual(result["info"]["status"], "success")

    def test_rule_based_cleanup_is_used_when_no_cleanup_provider(self):
        post = SocialPost(
            platform="xiaohongshu",
            content_type="image_note",
            source_url="https://www.xiaohongshu.com/explore/demo",
            resolved_url="https://www.xiaohongshu.com/explore/demo",
            post_id="note-2",
            title="测试规则整理",
            body="正文段落",
            image_urls=["https://img.example.com/one.jpg"],
        )
        service = SocialExtractorService(
            platform_adapters=[FakePlatformAdapter(post)],
            asr_providers={},
            cleanup_providers={},
            vision_providers={"qwen-vl": FailingVisionProvider()},
            ocr_provider=FakeOcrProvider(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = service.extract_social_post(
                "https://www.xiaohongshu.com/explore/demo",
                output_dir=tmpdir,
                vision_provider="qwen-vl",
            )

            self.assertIn("正文段落", result["script_preview"])
            self.assertIn("OCR:1:one.jpg", result["script_preview"])

    def test_video_asr_failure_still_writes_partial_artifacts(self):
        post = SocialPost(
            platform="douyin",
            content_type="video",
            source_url="https://v.douyin.com/demo/",
            resolved_url="https://www.iesdouyin.com/share/video/999",
            post_id="999",
            title="测试 ASR 失败",
            body="只有正文也要落盘",
            author_name="作者C",
            video_url="https://cdn.example.com/video.mp4",
        )
        service = SocialExtractorService(
            platform_adapters=[FakePlatformAdapter(post)],
            asr_providers={"siliconflow": FailingAsrProvider()},
            cleanup_providers={},
            vision_providers={},
            ocr_provider=FakeOcrProvider(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = service.extract_social_post(
                "https://v.douyin.com/demo/",
                output_dir=tmpdir,
                asr_provider="siliconflow",
            )

            self.assertEqual(result["info"]["status"], "partial_success")
            self.assertIn("asr failed", result["info"]["error"])
            script_text = Path(result["script_path"]).read_text(encoding="utf-8")
            self.assertIn("只有正文也要落盘", script_text)

    def test_openai_compatible_asr_provider_uses_provider_specific_upload_endpoint(self):
        post = SocialPost(
            platform="douyin",
            content_type="video",
            source_url="https://v.douyin.com/demo/",
            resolved_url="https://www.iesdouyin.com/share/video/888",
            post_id="888",
            title="测试豆包 ASR",
            video_url="https://cdn.example.com/video.mp4",
        )
        context = ExtractionContext(
            asr_provider="doubao",
            asr_model="doubao-asr-demo",
        )
        provider = OpenAICompatibleASRProvider("doubao")

        with (
            patch.dict(
                "os.environ",
                {
                    "ARK_API_KEY": "ark-demo-key",
                    "ARK_ASR_URL": "https://ark.example.com/audio/transcriptions",
                },
                clear=False,
            ),
            patch("douyin_mcp_server.social_extractor.download_binary") as download_binary,
            patch("douyin_mcp_server.social_extractor.extract_audio") as extract_audio,
            patch("douyin_mcp_server.social_extractor.transcribe_audio_via_upload") as transcribe_audio_via_upload,
        ):
            download_binary.return_value = Path("/tmp/demo.mp4")
            extract_audio.return_value = Path("/tmp/demo.mp3")
            transcribe_audio_via_upload.return_value = "转写结果"

            transcript = provider.transcribe(post, context)

        self.assertEqual(transcript, "转写结果")
        transcribe_audio_via_upload.assert_called_once()
        kwargs = transcribe_audio_via_upload.call_args.kwargs
        self.assertEqual(kwargs["api_url"], "https://ark.example.com/audio/transcriptions")
        self.assertEqual(kwargs["api_key"], "ark-demo-key")
        self.assertEqual(kwargs["model"], "doubao-asr-demo")


class ProviderConfigTests(unittest.TestCase):
    def test_provider_asr_config_reads_bailian_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "BAILIAN_API_KEY": "bailian-demo-key",
            },
            clear=False,
        ):
            config = provider_asr_config("bailian")

        self.assertIsNotNone(config)
        self.assertEqual(config["api_key"], "bailian-demo-key")
        self.assertEqual(
            config["api_url"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions",
        )

    def test_provider_volcengine_speech_config_reads_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "VOLCENGINE_SPEECH_APP_ID": "123456789",
                "VOLCENGINE_SPEECH_ACCESS_TOKEN": "token-demo",
            },
            clear=False,
        ):
            config = provider_volcengine_speech_config()

        self.assertIsNotNone(config)
        self.assertEqual(config["app_id"], "123456789")
        self.assertEqual(config["access_token"], "token-demo")
        self.assertEqual(config["resource_id"], "volc.seedasr.sauc.duration")
        self.assertEqual(config["ws_url"], "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async")


class VolcengineSpeechProtocolTests(unittest.TestCase):
    def test_build_full_client_request_sets_expected_header(self):
        frame = build_volcengine_full_client_request(
            {
                "user": {"uid": "u-demo"},
                "audio": {"format": "wav", "rate": 16000, "bits": 16, "channel": 1},
                "request": {"model_name": "bigmodel"},
            }
        )

        self.assertEqual(frame[0], 0x11)
        self.assertEqual(frame[1], 0x10)
        self.assertEqual(frame[2], 0x11)
        self.assertEqual(frame[3], 0x00)

    def test_build_audio_request_marks_last_packet(self):
        non_final = build_volcengine_audio_request(b"abc", is_final=False)
        final = build_volcengine_audio_request(b"abc", is_final=True)

        self.assertEqual(non_final[0], 0x11)
        self.assertEqual(non_final[1], 0x20)
        self.assertEqual(non_final[2], 0x01)
        self.assertEqual(final[1], 0x22)


class XHSStateParserTests(unittest.TestCase):
    def test_parse_html_extracts_video_metadata(self):
        html = """
        <html><body><script>
        window.__INITIAL_STATE__={"global":{"pwaAddDesktopPrompt":undefined},"note":{"noteDetailMap":{"abc123":{"note":{
        "noteId":"abc123",
        "xsecToken":"tok",
        "title":"视频标题",
        "desc":"视频正文",
        "type":"video",
        "time":123456,
        "user":{"nickname":"作者","userId":"user-1"},
        "imageList":[{"urlDefault":"http://img.example.com/cover.jpg"}],
        "video":{"media":{"stream":{"h264":[{"masterUrl":"http://video.example.com/a.mp4"}]}},"capa":{"duration":88}}
        }}}}}
        </script></body></html>
        """

        post = XHSStateParser.parse_html(
            html=html,
            source_url="https://xhslink.com/demo",
            resolved_url="https://www.xiaohongshu.com/explore/abc123?xsec_token=tok",
        )

        self.assertEqual(post.platform, "xiaohongshu")
        self.assertEqual(post.content_type, "video")
        self.assertEqual(post.post_id, "abc123")
        self.assertEqual(post.video_url, "https://video.example.com/a.mp4")
        self.assertEqual(post.cover_url, "https://img.example.com/cover.jpg")
        self.assertEqual(post.duration_sec, 88)


if __name__ == "__main__":
    unittest.main()
