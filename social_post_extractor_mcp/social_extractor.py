#!/usr/bin/env python3
"""Unified social post extraction pipeline for Douyin and XiaoHongShu."""

from __future__ import annotations

import json
import os
import re
import tempfile
import asyncio
import gzip
import mimetypes
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}

DOUYIN_MOBILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 "
        "Version/17.0 Mobile/15E148 Safari/604.1"
    )
}

DEFAULT_ASR_PROVIDER = "bailian"
DEFAULT_ASR_MODEL = "paraformer-v2"
DEFAULT_VISION_PROVIDER = "bailian"
DEFAULT_CLEAN_PROVIDER = "bailian"
DEFAULT_DASHSCOPE_SHORT_ASR_MODEL = "qwen3-asr-flash"
DEFAULT_DASHSCOPE_LONG_ASR_MODEL = "qwen3-asr-flash-filetrans"
DEFAULT_DASHSCOPE_SHORT_MAX_DURATION_SEC = 300

VISION_PROMPT = (
    "请逐字提取这张图片里可见的所有文字。保持原有的段落、列表、标题和换行。"
    "不要总结，不要解释，不要补充图片里不存在的内容。"
)

OCR_FALLBACK_PROMPT = (
    "请只做 OCR 抽字。逐字输出图片中的所有可见文字，保留顺序与换行。"
    "不要总结，不要改写，不要补全。"
)

CLEANUP_PROMPT_TEMPLATE = """你要对短视频/图文内容做轻量整理，只提升可读性，不改写内容。

要求：
1. 不能丢失信息，不能新增信息，不能做摘要。
2. 只允许做这几类处理：补标点、分段、修正极明显的错别字、去除重复空行。
3. 不要改写句子意思，不要重组结构，不要润色成另一种文风。
4. 保留原文中的口头表达、术语、清单和操作步骤。
5. 输出纯文本，不要加解释。

标题：{title}
平台：{platform}
内容类型：{content_type}

原笔记正文：
{body}

原始转写：
{raw_transcript}

图片文字：
{image_texts}
"""


@dataclass
class SocialPost:
    platform: str
    content_type: str
    source_url: str
    resolved_url: str
    post_id: str
    title: str
    body: str = ""
    author_name: Optional[str] = None
    author_id: Optional[str] = None
    publish_time: Optional[int] = None
    cover_url: Optional[str] = None
    duration_sec: Optional[int] = None
    video_url: Optional[str] = None
    image_urls: list[str] = field(default_factory=list)
    page_url: Optional[str] = None
    xsec_token: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactPaths:
    script_path: Path
    info_path: Path


@dataclass
class ExtractionContext:
    asr_provider: Optional[str] = None
    asr_model: Optional[str] = None
    vision_provider: Optional[str] = None
    vision_model: Optional[str] = None
    clean_provider: Optional[str] = None
    clean_model: Optional[str] = None
    output_dir: Optional[Path] = None


class PlatformAdapter:
    def can_handle(self, share_text: str) -> bool:
        raise NotImplementedError

    def fetch_post(self, share_text: str) -> SocialPost:
        raise NotImplementedError


class XHSStateParser:
    """Parse XiaoHongShu note data from the page's initial state."""

    @staticmethod
    def parse_html(html: str, source_url: str, resolved_url: str) -> SocialPost:
        state_match = re.search(r"window\.__INITIAL_STATE__=(.*?)</script>", html, flags=re.DOTALL)
        if not state_match:
            raise ValueError("未找到小红书页面状态数据")

        state_blob = state_match.group(1)
        state_blob = re.sub(r":undefined([,}])", r":null\1", state_blob)
        state = json.loads(state_blob)

        note = None
        note_map = ((state.get("note") or {}).get("noteDetailMap") or {})
        if isinstance(note_map, dict) and note_map:
            first_entry = next(iter(note_map.values()))
            if isinstance(first_entry, dict):
                note = first_entry.get("note")
        if not isinstance(note, dict):
            raise ValueError("未找到小红书笔记详情")

        note_id = note.get("noteId") or XHSStateParser._extract_note_id_from_url(resolved_url)
        if not note_id:
            raise ValueError("未找到小红书笔记 ID")

        image_urls = []
        cover_url = None
        for image in note.get("imageList") or []:
            if not isinstance(image, dict):
                continue
            image_url = image.get("urlDefault") or image.get("urlPre") or image.get("url")
            if image_url:
                normalized = _normalize_media_url(image_url)
                image_urls.append(normalized)
                if not cover_url:
                    cover_url = normalized

        video_url = None
        duration_sec = None
        video = note.get("video") or {}
        stream = ((video.get("media") or {}).get("stream") or {})
        for codec in ("h264", "h265", "av1"):
            candidates = stream.get(codec) or []
            if candidates and isinstance(candidates[0], dict):
                master_url = candidates[0].get("masterUrl")
                if master_url:
                    video_url = _normalize_media_url(master_url)
                    break
        duration_sec = ((video.get("capa") or {}).get("duration")) or None

        user = note.get("user") or {}
        tags = []
        for tag in note.get("tagList") or []:
            if isinstance(tag, dict) and tag.get("name"):
                tags.append(tag["name"])

        note_type = note.get("type") or "normal"
        content_type = "video" if note_type == "video" or video_url else "image_note"
        return SocialPost(
            platform="xiaohongshu",
            content_type=content_type,
            source_url=source_url,
            resolved_url=resolved_url,
            post_id=note_id,
            title=(note.get("title") or "").strip() or note_id,
            body=(note.get("desc") or "").strip(),
            author_name=(user.get("nickname") or user.get("nickName")),
            author_id=user.get("userId"),
            publish_time=note.get("time"),
            cover_url=cover_url,
            duration_sec=duration_sec,
            video_url=video_url,
            image_urls=image_urls,
            page_url=resolved_url,
            xsec_token=note.get("xsecToken") or _extract_xsec_token(resolved_url),
            tags=tags,
            extra={"note_type": note_type},
        )

    @staticmethod
    def _extract_note_id_from_url(url: str) -> Optional[str]:
        match = re.search(r"/(?:explore|discovery/item)/([^/?]+)", url)
        return match.group(1) if match else None


class XiaoHongShuPlatformAdapter(PlatformAdapter):
    def can_handle(self, share_text: str) -> bool:
        lower = share_text.lower()
        return "xiaohongshu.com" in lower or "xhslink.com" in lower

    def fetch_post(self, share_text: str) -> SocialPost:
        source_url = _extract_first_url(share_text)
        response = requests.get(source_url, headers=HEADERS, timeout=30, allow_redirects=True)
        response.raise_for_status()
        return XHSStateParser.parse_html(
            html=response.text,
            source_url=source_url,
            resolved_url=response.url,
        )


class DouyinPlatformAdapter(PlatformAdapter):
    def can_handle(self, share_text: str) -> bool:
        lower = share_text.lower()
        return "douyin.com" in lower or "iesdouyin.com" in lower

    def fetch_post(self, share_text: str) -> SocialPost:
        source_url = _extract_first_url(share_text)
        share_response = requests.get(source_url, headers=DOUYIN_MOBILE_HEADERS, timeout=30, allow_redirects=True)
        share_response.raise_for_status()
        video_id = share_response.url.split("?")[0].strip("/").split("/")[-1]
        share_url = f"https://www.iesdouyin.com/share/video/{video_id}"

        response = requests.get(share_url, headers=DOUYIN_MOBILE_HEADERS, timeout=30)
        response.raise_for_status()
        pattern = re.compile(r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", flags=re.DOTALL)
        match = pattern.search(response.text)
        if not match:
            raise ValueError("从抖音 HTML 中解析视频信息失败")

        data = json.loads(match.group(1).strip())
        loader_data = data.get("loaderData") or {}
        video_info_res = None
        for key in ("video_(id)/page", "note_(id)/page"):
            page = loader_data.get(key) or {}
            if page.get("videoInfoRes"):
                video_info_res = page["videoInfoRes"]
                break
        if not video_info_res:
            raise ValueError("无法从抖音 JSON 中解析视频或图集信息")

        item = (video_info_res.get("item_list") or [{}])[0]
        video = item.get("video") or {}
        play_addr = video.get("play_addr") or {}
        url_list = play_addr.get("url_list") or []
        video_url = None
        if url_list:
            video_url = _normalize_media_url(url_list[0].replace("playwm", "play"))

        author = item.get("author") or {}
        cover = video.get("cover") or {}
        cover_urls = cover.get("url_list") or []
        duration_ms = video.get("duration")
        duration_sec = None
        if isinstance(duration_ms, (int, float)) and duration_ms:
            duration_sec = int(duration_ms / 1000) if duration_ms > 1000 else int(duration_ms)

        title = (item.get("desc") or "").strip() or f"douyin_{video_id}"
        title = re.sub(r'[\\/:*?"<>|]', "_", title)

        return SocialPost(
            platform="douyin",
            content_type="video",
            source_url=source_url,
            resolved_url=share_response.url,
            post_id=video_id,
            title=title,
            body=(item.get("desc") or "").strip(),
            author_name=author.get("nickname") or author.get("unique_id"),
            author_id=author.get("uid") or author.get("sec_uid"),
            publish_time=item.get("create_time"),
            cover_url=_normalize_media_url(cover_urls[0]) if cover_urls else None,
            duration_sec=duration_sec,
            video_url=video_url,
            image_urls=[_normalize_media_url(cover_urls[0])] if cover_urls else [],
            page_url=share_url,
            extra={"aweme_type": item.get("aweme_type")},
        )


class SocialExtractorService:
    def __init__(
        self,
        platform_adapters: Optional[list[PlatformAdapter]] = None,
        asr_providers: Optional[dict[str, Any]] = None,
        cleanup_providers: Optional[dict[str, Any]] = None,
        vision_providers: Optional[dict[str, Any]] = None,
        ocr_provider: Any = None,
    ):
        self.platform_adapters = platform_adapters or [
            DouyinPlatformAdapter(),
            XiaoHongShuPlatformAdapter(),
        ]
        self.asr_providers = asr_providers or {
            "siliconflow": SiliconFlowASRProvider(),
            "dashscope": DashScopeASRProvider(),
            "bailian": DashScopeASRProvider(),
            "doubao": OpenAICompatibleASRProvider("doubao"),
            "volcengine_speech": VolcengineSpeechASRProvider(),
        }
        self.cleanup_providers = cleanup_providers or build_llm_provider_registry()
        self.vision_providers = vision_providers or build_llm_provider_registry()
        self.ocr_provider = ocr_provider or LLMOcrProvider(self.vision_providers)

    def parse_social_post(self, share_text: str) -> SocialPost:
        adapter = self._resolve_platform_adapter(share_text)
        return adapter.fetch_post(share_text)

    def extract_social_post(
        self,
        share_text: str,
        *,
        output_dir: Optional[str] = None,
        asr_provider: Optional[str] = None,
        asr_model: Optional[str] = None,
        vision_provider: Optional[str] = None,
        vision_model: Optional[str] = None,
        clean_provider: Optional[str] = None,
        clean_model: Optional[str] = None,
        save_raw_segments: bool = False,
    ) -> dict[str, Any]:
        post = self.parse_social_post(share_text)
        resolved_asr_provider = asr_provider or os.getenv("ASR_PROVIDER") or DEFAULT_ASR_PROVIDER
        context = ExtractionContext(
            asr_provider=resolved_asr_provider,
            asr_model=(
                asr_model
                or os.getenv("ASR_MODEL")
                or default_model_for_provider(resolved_asr_provider, "asr")
                or DEFAULT_ASR_MODEL
            ),
            vision_provider=vision_provider or os.getenv("VISION_PROVIDER") or self._default_vision_provider(),
            vision_model=None,
            clean_provider=clean_provider or os.getenv("CLEAN_PROVIDER") or self._default_cleanup_provider(),
            clean_model=None,
        )
        context.vision_model = (
            vision_model
            or os.getenv("VISION_MODEL")
            or (
                default_model_for_provider(context.vision_provider, "vision")
                if context.vision_provider
                else None
            )
        )
        context.clean_model = (
            clean_model
            or os.getenv("CLEAN_MODEL")
            or (
                default_model_for_provider(context.clean_provider, "cleanup")
                if context.clean_provider
                else None
            )
        )

        raw_transcript = None
        image_texts: list[str] = []
        errors: list[str] = []

        if post.content_type == "video":
            try:
                raw_transcript = self._extract_video_text(post, context)
            except Exception as exc:
                raise RuntimeError(f"视频转写失败: {exc}") from exc
        else:
            image_texts, image_errors = self._extract_image_texts(post, context)
            errors.extend(image_errors)

        cleaned_script = self._cleanup_content(post, raw_transcript, image_texts, context, errors)
        status = "success"
        if errors:
            status = "partial_success"
        if not cleaned_script and not raw_transcript and not image_texts and not post.body:
            status = "failed"

        artifact_dir = self._prepare_output_dir(output_dir, post)
        paths = self._write_artifacts(
            post=post,
            context=context,
            artifact_dir=artifact_dir,
            cleaned_script=cleaned_script,
            raw_transcript=raw_transcript,
            image_texts=image_texts,
            status=status,
            errors=errors,
        )

        info = json.loads(paths.info_path.read_text(encoding="utf-8"))
        return {
            "platform": post.platform,
            "content_type": post.content_type,
            "post_id": post.post_id,
            "script_path": str(paths.script_path),
            "info_path": str(paths.info_path),
            "script_preview": cleaned_script or raw_transcript or "\n\n".join(image_texts) or post.body,
            "info": info,
            "raw_transcript": raw_transcript,
        }

    def _resolve_platform_adapter(self, share_text: str) -> PlatformAdapter:
        for adapter in self.platform_adapters:
            if adapter.can_handle(share_text):
                return adapter
        raise ValueError("unsupported_platform")

    def _extract_video_text(self, post: SocialPost, context: ExtractionContext) -> str:
        provider_name = context.asr_provider or DEFAULT_ASR_PROVIDER
        provider = self.asr_providers.get(provider_name)
        if not provider:
            raise ValueError(f"未配置 ASR provider: {provider_name}")
        return provider.transcribe(post, context)

    def _extract_image_texts(self, post: SocialPost, context: ExtractionContext) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        errors: list[str] = []
        primary = self.vision_providers.get(context.vision_provider or "") if context.vision_provider else None
        for index, image_url in enumerate(post.image_urls, start=1):
            try:
                if primary:
                    text = primary.read_image_text(image_url, context, index)
                elif self.ocr_provider:
                    text = self.ocr_provider.read_image_text(image_url, context, index)
                else:
                    raise RuntimeError("未配置 vision provider")
                texts.append(text.strip())
            except Exception as vision_error:
                try:
                    if not self.ocr_provider:
                        raise
                    fallback_text = self.ocr_provider.read_image_text(image_url, context, index)
                    texts.append(fallback_text.strip())
                except Exception as ocr_error:
                    errors.append(
                        f"图片 {index} 提取失败: vision={vision_error}; ocr={ocr_error}"
                    )
        return texts, errors

    def _cleanup_content(
        self,
        post: SocialPost,
        raw_transcript: Optional[str],
        image_texts: list[str],
        context: ExtractionContext,
        errors: list[str],
    ) -> str:
        if context.clean_provider:
            provider = self.cleanup_providers.get(context.clean_provider)
            if provider:
                try:
                    return provider.cleanup(post, raw_transcript, image_texts, context).strip()
                except Exception as exc:
                    errors.append(f"整理失败: {exc}")
        return rule_based_cleanup(post, raw_transcript, image_texts)

    def _prepare_output_dir(self, output_dir: Optional[str], post: SocialPost) -> Path:
        base_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "social-post-extract"
        artifact_dir = base_dir / f"{post.platform}_{post.content_type}_{post.post_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _write_artifacts(
        self,
        *,
        post: SocialPost,
        context: ExtractionContext,
        artifact_dir: Path,
        cleaned_script: str,
        raw_transcript: Optional[str],
        image_texts: list[str],
        status: str,
        errors: list[str],
    ) -> ArtifactPaths:
        script_path = artifact_dir / "script.md"
        info_path = artifact_dir / "info.json"

        script_text = build_script_markdown(
            post=post,
            cleaned_script=cleaned_script,
            raw_transcript=raw_transcript,
            image_texts=image_texts,
        )
        script_path.write_text(script_text, encoding="utf-8")

        info = build_info_dict(post, context, status=status, errors=errors)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return ArtifactPaths(script_path=script_path, info_path=info_path)

    def _default_vision_provider(self) -> Optional[str]:
        if os.getenv("VISION_PROVIDER"):
            return os.getenv("VISION_PROVIDER")
        for candidate in ("bailian", "qwen", "doubao", "minimax", "generic"):
            if provider_credentials(candidate):
                return candidate
        return None

    def _default_cleanup_provider(self) -> Optional[str]:
        if os.getenv("CLEAN_PROVIDER"):
            return os.getenv("CLEAN_PROVIDER")
        for candidate in ("bailian", "qwen", "doubao", "minimax", "generic"):
            if provider_credentials(candidate):
                return candidate
        return None


class SiliconFlowASRProvider:
    def __init__(self, api_base_url: Optional[str] = None):
        self.api_base_url = api_base_url or os.getenv(
            "SILICONFLOW_ASR_BASE_URL", "https://api.siliconflow.cn/v1/audio/transcriptions"
        )

    def transcribe(self, post: SocialPost, context: ExtractionContext) -> str:
        api_key = first_env("SILICONFLOW_API_KEY", "API_KEY")
        if not api_key:
            raise ValueError("未设置 SiliconFlow API Key")
        if not post.video_url:
            raise ValueError("视频内容缺少 video_url，无法进行 ASR")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_path = download_binary(post.video_url, tmp_path / f"{post.post_id}.mp4")
            audio_path = extract_audio(video_path, tmp_path / f"{post.post_id}.mp3")
            return transcribe_audio_via_upload(
                audio_path=audio_path,
                api_url=self.api_base_url,
                api_key=api_key,
                model=context.asr_model or DEFAULT_ASR_MODEL,
            )


class DashScopeASRProvider:
    def transcribe(self, post: SocialPost, context: ExtractionContext) -> str:
        api_key = first_env("DASHSCOPE_API_KEY", "BAILIAN_API_KEY")
        if not api_key:
            raise ValueError("未设置 DashScope/Bailian API Key")
        if not post.video_url:
            raise ValueError("视频内容缺少 video_url，无法调用 DashScope ASR")
        requested_model = _normalize_dashscope_requested_model(context.asr_model)
        preferred_model = requested_model or _select_dashscope_cloud_asr_model(post)

        try:
            transcript = self._transcribe_via_cloud_mirror(post, api_key=api_key, model=preferred_model)
            context.asr_model = preferred_model
            return transcript
        except Exception:
            # Short-form realtime ASR has tighter limits. If the auto path picked it,
            # retry once with the long-file async model for better robustness.
            if requested_model is None and preferred_model == DEFAULT_DASHSCOPE_SHORT_ASR_MODEL:
                fallback_model = DEFAULT_DASHSCOPE_LONG_ASR_MODEL
                transcript = self._transcribe_via_cloud_mirror(post, api_key=api_key, model=fallback_model)
                context.asr_model = fallback_model
                return transcript
            raise

    def _transcribe_via_cloud_mirror(self, post: SocialPost, *, api_key: str, model: str) -> str:
        oss_url = stream_remote_media_to_dashscope_oss(
            source_url=post.video_url,
            api_key=api_key,
            model_name=model,
            filename_hint=_default_dashscope_media_filename(post),
        )
        if model == DEFAULT_DASHSCOPE_SHORT_ASR_MODEL:
            return run_dashscope_multimodal_asr(
                oss_url=oss_url,
                api_key=api_key,
                model=model,
            )
        return run_dashscope_filetrans_task(
            oss_url=oss_url,
            api_key=api_key,
            model=model,
        )


class OpenAICompatibleASRProvider:
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def transcribe(self, post: SocialPost, context: ExtractionContext) -> str:
        config = provider_asr_config(self.provider_name)
        if not config:
            raise ValueError(f"provider {self.provider_name} 未配置 ASR 凭据")
        if not post.video_url:
            raise ValueError("视频内容缺少 video_url，无法进行 ASR")

        model = context.asr_model or default_model_for_provider(self.provider_name, "asr") or DEFAULT_ASR_MODEL

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                video_path = download_binary(post.video_url, tmp_path / f"{post.post_id}.mp4")
                audio_path = extract_audio(video_path, tmp_path / f"{post.post_id}.mp3")
                return transcribe_audio_via_upload(
                    audio_path=audio_path,
                    api_url=config["api_url"],
                    api_key=config["api_key"],
                    model=model,
                )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise RuntimeError(
                    f"{self.provider_name} ASR endpoint 不存在: {config['api_url']}。"
                    f"请配置 {provider_asr_url_env_name(self.provider_name)} 指向当前账号可用的官方转写接口。"
                ) from exc
            raise


class VolcengineSpeechASRProvider:
    def transcribe(self, post: SocialPost, context: ExtractionContext) -> str:
        config = provider_volcengine_speech_config()
        if not config:
            raise ValueError("未设置火山语音识别所需的 App ID / Access Token")
        if not post.video_url:
            raise ValueError("视频内容缺少 video_url，无法进行火山语音识别")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_path = download_binary(post.video_url, tmp_path / f"{post.post_id}.mp4")
            audio_path = extract_wav_audio(video_path, tmp_path / f"{post.post_id}.wav")
            return run_async(
                _transcribe_via_volcengine_websocket(
                    audio_path=audio_path,
                    config=config,
                    context=context,
                    user_id=post.author_id or post.post_id,
                )
            )


def run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("检测到已运行的事件循环，当前同步 provider 无法直接调用火山语音识别")


class OpenAICompatibleProvider:
    def __init__(self, provider_name: str, *, mode: str):
        self.provider_name = provider_name
        self.mode = mode

    def cleanup(self, post: SocialPost, raw_transcript: Optional[str], image_texts: list[str], context: ExtractionContext) -> str:
        model = context.clean_model or default_model_for_provider(self.provider_name, "cleanup")
        prompt = CLEANUP_PROMPT_TEMPLATE.format(
            title=post.title,
            platform=post.platform,
            content_type=post.content_type,
            body=post.body or "(无)",
            raw_transcript=raw_transcript or "(无)",
            image_texts="\n\n".join(image_texts) or "(无)",
        )
        return self._chat_text(prompt=prompt, model=model)

    def read_image_text(self, image_url: str, context: ExtractionContext, image_index: int) -> str:
        model = context.vision_model or default_model_for_provider(self.provider_name, "vision")
        prompt = VISION_PROMPT if self.mode == "vision" else OCR_FALLBACK_PROMPT
        return self._chat_vision(prompt=prompt, image_url=image_url, model=model)

    def _chat_text(self, prompt: str, model: str) -> str:
        config = provider_credentials(self.provider_name)
        if not config:
            raise ValueError(f"provider {self.provider_name} 未配置 API 凭据")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            f"{config['base_url'].rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return extract_message_text(response.json())

    def _chat_vision(self, prompt: str, image_url: str, model: str) -> str:
        config = provider_credentials(self.provider_name)
        if not config:
            raise ValueError(f"provider {self.provider_name} 未配置 API 凭据")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _normalize_media_url(image_url)}},
                    ],
                }
            ],
        }
        response = requests.post(
            f"{config['base_url'].rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return extract_message_text(response.json())


class LLMOcrProvider:
    def __init__(self, vision_providers: dict[str, Any]):
        self.vision_providers = vision_providers

    def read_image_text(self, image_url: str, context: ExtractionContext, image_index: int) -> str:
        provider_name = context.vision_provider or "qwen"
        provider = self.vision_providers.get(provider_name)
        if not provider:
            raise RuntimeError("未配置 OCR fallback provider")
        original_mode = getattr(provider, "mode", None)
        if original_mode is not None:
            provider.mode = "ocr"
        try:
            return provider.read_image_text(image_url, context, image_index)
        finally:
            if original_mode is not None:
                provider.mode = original_mode


def rule_based_cleanup(post: SocialPost, raw_transcript: Optional[str], image_texts: list[str]) -> str:
    parts = []
    if post.body:
        parts.append(post.body.strip())
    if raw_transcript:
        parts.append(raw_transcript.strip())
    if image_texts:
        parts.append("\n\n".join(text.strip() for text in image_texts if text.strip()))
    return "\n\n".join(part for part in parts if part).strip()


def build_script_markdown(
    *,
    post: SocialPost,
    cleaned_script: str,
    raw_transcript: Optional[str],
    image_texts: list[str],
) -> str:
    lines = [
        f"# {post.title}",
        "",
        f"- 平台: {post.platform}",
        f"- 类型: {post.content_type}",
        f"- ID: `{post.post_id}`",
        f"- 原链接: {post.source_url}",
    ]
    if post.author_name:
        lines.append(f"- 作者: {post.author_name}")
    if post.page_url:
        lines.append(f"- 页面: {post.page_url}")
    lines.append("")
    lines.append("## 整理稿")
    lines.append("")
    lines.append(cleaned_script or "（无整理稿）")
    if post.body:
        lines.extend(["", "## 原笔记正文", "", post.body])
    if raw_transcript:
        lines.extend(["", "## 原始转写", "", raw_transcript])
    if image_texts:
        lines.extend(["", "## 图片文字提取", ""])
        for index, text in enumerate(image_texts, start=1):
            lines.extend([f"### 图片 {index}", "", text or "（无内容）", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_info_dict(post: SocialPost, context: ExtractionContext, *, status: str, errors: list[str]) -> dict[str, Any]:
    return {
        "platform": post.platform,
        "content_type": post.content_type,
        "source_url": post.source_url,
        "resolved_url": post.resolved_url,
        "post_id": post.post_id,
        "title": post.title,
        "body": post.body,
        "author": {
            "name": post.author_name,
            "id": post.author_id,
        },
        "publish_time": post.publish_time,
        "cover_url": post.cover_url,
        "duration_sec": post.duration_sec,
        "video_url": post.video_url,
        "media_urls": post.image_urls if post.content_type == "image_note" else ([post.video_url] if post.video_url else []),
        "page_url": post.page_url,
        "xsec_token": post.xsec_token,
        "tags": post.tags,
        "asr_provider": context.asr_provider,
        "asr_model": context.asr_model,
        "vision_provider": context.vision_provider,
        "vision_model": context.vision_model,
        "clean_provider": context.clean_provider,
        "clean_model": context.clean_model,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": "\n".join(errors) if errors else None,
        "extra": post.extra,
    }


def _normalize_dashscope_requested_model(model: Optional[str]) -> Optional[str]:
    if model in (DEFAULT_DASHSCOPE_SHORT_ASR_MODEL, DEFAULT_DASHSCOPE_LONG_ASR_MODEL):
        return model
    return None


def _select_dashscope_cloud_asr_model(post: SocialPost) -> str:
    short_model = first_env("DASHSCOPE_SHORT_ASR_MODEL", "BAILIAN_SHORT_ASR_MODEL") or DEFAULT_DASHSCOPE_SHORT_ASR_MODEL
    long_model = first_env("DASHSCOPE_LONG_ASR_MODEL", "BAILIAN_LONG_ASR_MODEL") or DEFAULT_DASHSCOPE_LONG_ASR_MODEL
    max_short_duration = int(
        first_env("DASHSCOPE_SHORT_MAX_DURATION_SEC", "BAILIAN_SHORT_MAX_DURATION_SEC")
        or DEFAULT_DASHSCOPE_SHORT_MAX_DURATION_SEC
    )
    if post.duration_sec and post.duration_sec <= max_short_duration:
        return short_model
    return long_model


def _default_dashscope_media_filename(post: SocialPost) -> str:
    parsed = urlparse(post.video_url or "")
    basename = Path(parsed.path).name
    if basename and "." in basename:
        return basename
    if "video_id" in parse_qs(parsed.query):
        return f"{parse_qs(parsed.query)['video_id'][0]}.mp4"
    return f"{post.post_id}.mp4"


def build_llm_provider_registry() -> dict[str, OpenAICompatibleProvider]:
    registry: dict[str, OpenAICompatibleProvider] = {}
    for provider_name in ("minimax", "qwen", "bailian", "doubao", "generic"):
        registry[provider_name] = OpenAICompatibleProvider(provider_name, mode="vision")
    return registry


def provider_volcengine_speech_config() -> Optional[dict[str, str]]:
    app_id = first_env("VOLCENGINE_SPEECH_APP_ID", "VOLCENGINE_APP_ID")
    access_token = first_env("VOLCENGINE_SPEECH_ACCESS_TOKEN", "VOLCENGINE_ACCESS_TOKEN")
    if not app_id or not access_token:
        return None
    return {
        "app_id": app_id,
        "access_token": access_token,
        "resource_id": first_env("VOLCENGINE_SPEECH_RESOURCE_ID") or "volc.seedasr.sauc.duration",
        "ws_url": first_env("VOLCENGINE_SPEECH_WS_URL") or "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
    }


def provider_asr_config(provider_name: str) -> Optional[dict[str, str]]:
    env_map = {
        "bailian": {
            "keys": ["BAILIAN_API_KEY", "DASHSCOPE_API_KEY"],
            "api_urls": ["BAILIAN_ASR_URL", "DASHSCOPE_ASR_URL"],
            "base_urls": ["BAILIAN_ASR_BASE_URL", "DASHSCOPE_ASR_BASE_URL", "BAILIAN_BASE_URL", "DASHSCOPE_BASE_URL"],
            "default_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions",
        },
        "doubao": {
            "keys": ["DOUBAO_API_KEY", "ARK_API_KEY"],
            "api_urls": ["DOUBAO_ASR_URL", "ARK_ASR_URL"],
            "base_urls": ["DOUBAO_ASR_BASE_URL", "ARK_ASR_BASE_URL", "DOUBAO_BASE_URL", "ARK_BASE_URL"],
            "default_url": "https://ark.cn-beijing.volces.com/api/v3/audio/transcriptions",
        },
    }
    spec = env_map.get(provider_name)
    if not spec:
        return None
    api_key = first_env(*spec["keys"])
    if not api_key:
        return None
    api_url = first_env(*spec["api_urls"])
    if not api_url:
        base_url = first_env(*spec["base_urls"])
        api_url = _join_api_url(base_url, "/audio/transcriptions") if base_url else spec["default_url"]
    return {"api_key": api_key, "api_url": api_url}


def provider_credentials(provider_name: str) -> Optional[dict[str, str]]:
    env_map = {
        "minimax": {
            "keys": ["MINIMAX_API_KEY"],
            "base_urls": ["MINIMAX_BASE_URL"],
            "default": "https://api.minimax.chat/v1",
        },
        "qwen": {
            "keys": ["QWEN_API_KEY", "DASHSCOPE_API_KEY", "BAILIAN_API_KEY"],
            "base_urls": ["QWEN_BASE_URL", "DASHSCOPE_BASE_URL", "BAILIAN_BASE_URL"],
            "default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "bailian": {
            "keys": ["BAILIAN_API_KEY", "DASHSCOPE_API_KEY"],
            "base_urls": ["BAILIAN_BASE_URL", "DASHSCOPE_BASE_URL"],
            "default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "doubao": {
            "keys": ["DOUBAO_API_KEY", "ARK_API_KEY"],
            "base_urls": ["DOUBAO_BASE_URL", "ARK_BASE_URL"],
            "default": "https://ark.cn-beijing.volces.com/api/v3",
        },
        "generic": {
            "keys": ["OPENAI_API_KEY"],
            "base_urls": ["OPENAI_BASE_URL"],
            "default": "https://api.openai.com/v1",
        },
    }
    spec = env_map.get(provider_name)
    if not spec:
        return None
    api_key = first_env(*spec["keys"])
    if not api_key:
        return None
    base_url = first_env(*spec["base_urls"]) or spec["default"]
    return {"api_key": api_key, "base_url": base_url}


def default_model_for_provider(provider_name: str, purpose: str) -> Optional[str]:
    purpose_env = {
        ("bailian", "asr"): "ASR_MODEL",
        ("qwen", "vision"): "VISION_MODEL",
        ("qwen", "cleanup"): "CLEAN_MODEL",
        ("bailian", "vision"): "VISION_MODEL",
        ("bailian", "cleanup"): "CLEAN_MODEL",
        ("doubao", "asr"): "ASR_MODEL",
        ("volcengine_speech", "asr"): "ASR_MODEL",
        ("doubao", "vision"): "VISION_MODEL",
        ("doubao", "cleanup"): "CLEAN_MODEL",
        ("minimax", "vision"): "VISION_MODEL",
        ("minimax", "cleanup"): "CLEAN_MODEL",
        ("generic", "vision"): "VISION_MODEL",
        ("generic", "cleanup"): "CLEAN_MODEL",
    }
    env_name = purpose_env.get((provider_name, purpose))
    if env_name and os.getenv(env_name):
        return os.getenv(env_name)
    defaults = {
        ("bailian", "asr"): "paraformer-v2",
        ("qwen", "vision"): "qwen3-vl-flash",
        ("qwen", "cleanup"): "qwen-flash",
        ("bailian", "vision"): "qwen3-vl-flash",
        ("bailian", "cleanup"): "qwen-flash",
        ("doubao", "asr"): os.getenv("DOUBAO_ASR_MODEL") or os.getenv("ARK_ASR_MODEL"),
        ("volcengine_speech", "asr"): os.getenv("VOLCENGINE_SPEECH_MODEL_NAME", "bigmodel"),
        ("doubao", "vision"): "doubao-vision-pro-32k",
        ("doubao", "cleanup"): "doubao-pro-32k",
        ("minimax", "vision"): "MiniMax-Text-01",
        ("minimax", "cleanup"): "MiniMax-Text-01",
        ("generic", "vision"): os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        ("generic", "cleanup"): os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini"),
    }
    return defaults.get((provider_name, purpose))


def get_dashscope_upload_policy(api_key: str, model_name: str) -> dict[str, Any]:
    response = requests.get(
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        params={
            "action": "getPolicy",
            "model": model_name,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"DashScope 上传凭证响应异常: {payload}")
    return data


class StreamingMultipartForm:
    def __init__(
        self,
        *,
        boundary: str,
        fields: list[tuple[str, str]],
        file_field_name: str,
        file_name: str,
        file_content_type: str,
        file_chunks: Any,
        file_size: int,
    ):
        self.boundary = boundary
        self.file_chunks = file_chunks
        self.file_size = file_size
        self._prefix = self._build_prefix(
            fields=fields,
            file_field_name=file_field_name,
            file_name=file_name,
            file_content_type=file_content_type,
        )
        self._suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")

    def _build_prefix(
        self,
        *,
        fields: list[tuple[str, str]],
        file_field_name: str,
        file_name: str,
        file_content_type: str,
    ) -> bytes:
        parts: list[bytes] = []
        for name, value in fields:
            parts.append(
                (
                    f"--{self.boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode("utf-8")
            )
        parts.append(
            (
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{file_field_name}"; filename="{file_name}"\r\n'
                f"Content-Type: {file_content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        return b"".join(parts)

    def __iter__(self):
        yield self._prefix
        for chunk in self.file_chunks:
            if chunk:
                yield chunk
        yield self._suffix

    def __len__(self) -> int:
        return len(self._prefix) + self.file_size + len(self._suffix)


def _guess_media_content_type(source_url: str, response: requests.Response, filename: str) -> str:
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(filename or source_url)
    return guessed or "application/octet-stream"


def stream_remote_media_to_dashscope_oss(
    *,
    source_url: str,
    api_key: str,
    model_name: str,
    filename_hint: str,
) -> str:
    policy_data = get_dashscope_upload_policy(api_key, model_name)
    with requests.Session() as media_session:
        media_session.trust_env = False
        with media_session.get(
            _normalize_media_url(source_url),
            headers=HEADERS,
            timeout=120,
            stream=True,
            allow_redirects=True,
        ) as media_response:
            media_response.raise_for_status()
            content_length = int(media_response.headers.get("Content-Length") or "0")
            if content_length <= 0:
                raise RuntimeError("远程媒体缺少 Content-Length，无法稳定上传到 DashScope 临时存储")

            file_name = filename_hint or Path(urlparse(str(media_response.url)).path).name or f"{uuid.uuid4().hex}.bin"
            content_type = _guess_media_content_type(source_url, media_response, file_name)
            key = f"{policy_data['upload_dir'].rstrip('/')}/{file_name}"
            form = StreamingMultipartForm(
                boundary=f"dashscope-{uuid.uuid4().hex}",
                fields=[
                    ("OSSAccessKeyId", policy_data["oss_access_key_id"]),
                    ("Signature", policy_data["signature"]),
                    ("policy", policy_data["policy"]),
                    ("x-oss-object-acl", policy_data["x_oss_object_acl"]),
                    ("x-oss-forbid-overwrite", policy_data["x_oss_forbid_overwrite"]),
                    ("key", key),
                    ("success_action_status", "200"),
                ],
                file_field_name="file",
                file_name=file_name,
                file_content_type=content_type,
                file_chunks=media_response.iter_content(chunk_size=1024 * 1024),
                file_size=content_length,
            )
            with requests.Session() as upload_session:
                upload_session.trust_env = False
                upload_response = upload_session.post(
                    policy_data["upload_host"],
                    data=form,
                    headers={
                        "Content-Type": f"multipart/form-data; boundary={form.boundary}",
                        "Content-Length": str(len(form)),
                    },
                    timeout=(60, 1800),
                )
            upload_response.raise_for_status()
            return f"oss://{key}"


def run_dashscope_multimodal_asr(*, oss_url: str, api_key: str, model: str) -> str:
    import dashscope

    dashscope.base_http_api_url = first_env("DASHSCOPE_BASE_URL", "BAILIAN_BASE_URL") or "https://dashscope.aliyuncs.com/api/v1"
    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": [{"audio": oss_url}]}],
        result_format="message",
        asr_options={
            "language": "zh",
            "enable_itn": False,
        },
    )
    if getattr(response, "status_code", 200) != 200:
        raise RuntimeError(getattr(response, "message", "DashScope 实时 ASR 调用失败"))
    transcript = extract_dashscope_message_text(response)
    if transcript:
        return transcript
    raise RuntimeError("DashScope 实时 ASR 返回空文本")


def run_dashscope_filetrans_task(*, oss_url: str, api_key: str, model: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
        "X-DashScope-OssResourceResolve": "enable",
    }
    submit_response = requests.post(
        "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
        headers=headers,
        json={
            "model": model,
            "input": {"file_url": oss_url},
            "parameters": {
                "language": "zh",
                "enable_itn": False,
                "enable_words": True,
                "channel_id": [0],
            },
        },
        timeout=60,
    )
    submit_response.raise_for_status()
    submit_payload = submit_response.json()
    task_id = ((submit_payload.get("output") or {}).get("task_id"))
    if not task_id:
        raise RuntimeError(f"DashScope filetrans 未返回 task_id: {submit_payload}")

    poll_interval = float(first_env("DASHSCOPE_FILETRANS_POLL_INTERVAL_SEC", "BAILIAN_FILETRANS_POLL_INTERVAL_SEC") or "5")
    timeout_sec = float(first_env("DASHSCOPE_FILETRANS_TIMEOUT_SEC", "BAILIAN_FILETRANS_TIMEOUT_SEC") or "7200")
    deadline = time.monotonic() + timeout_sec
    last_payload: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        poll_response = requests.get(
            f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
            headers=headers,
            timeout=60,
        )
        poll_response.raise_for_status()
        last_payload = poll_response.json()
        output = last_payload.get("output") or {}
        task_status = output.get("task_status")
        if task_status == "SUCCEEDED":
            transcript = _extract_dashscope_filetrans_text(output)
            if transcript:
                return transcript
            raise RuntimeError(f"DashScope filetrans 成功但未返回文本: {last_payload}")
        if task_status == "FAILED":
            raise RuntimeError(f"DashScope filetrans 失败: {last_payload}")
        time.sleep(poll_interval)

    raise RuntimeError(f"DashScope filetrans 超时: {last_payload}")


def _extract_dashscope_filetrans_text(output: dict[str, Any]) -> str:
    result = output.get("result") or {}
    transcription_url = result.get("transcription_url")
    if transcription_url:
        transcript_response = requests.get(transcription_url, timeout=60)
        transcript_response.raise_for_status()
        transcript_payload = transcript_response.json()
        transcripts = transcript_payload.get("transcripts") or []
        parts = [item.get("text", "").strip() for item in transcripts if item.get("text")]
        return "\n".join(part for part in parts if part)

    results = output.get("results") or []
    for item in results:
        transcription_url = item.get("transcription_url")
        if transcription_url:
            transcript_response = requests.get(transcription_url, timeout=60)
            transcript_response.raise_for_status()
            transcript_payload = transcript_response.json()
            transcripts = transcript_payload.get("transcripts") or []
            parts = [entry.get("text", "").strip() for entry in transcripts if entry.get("text")]
            if parts:
                return "\n".join(parts)
    return ""


def transcribe_audio_via_upload(*, audio_path: Path, api_url: str, api_key: str, model: str) -> str:
    files = {
        "file": (audio_path.name, open(audio_path, "rb"), "audio/mpeg"),
        "model": (None, model),
    }
    try:
        response = requests.post(
            api_url,
            files=files,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text") or response.text
    finally:
        files["file"][1].close()


def build_volcengine_full_client_request(payload: dict[str, Any]) -> bytes:
    body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return bytes([0x11, 0x10, 0x11, 0x00]) + len(body).to_bytes(4, "big") + body


def build_volcengine_audio_request(audio_chunk: bytes, *, is_final: bool) -> bytes:
    body = gzip.compress(audio_chunk)
    message_type = 0x22 if is_final else 0x20
    return bytes([0x11, message_type, 0x01, 0x00]) + len(body).to_bytes(4, "big") + body


def parse_volcengine_server_message(message: bytes) -> dict[str, Any]:
    if len(message) < 4:
        raise ValueError("火山语音返回了无效响应帧")

    header_words = message[0] & 0x0F
    header_size = max(header_words, 1) * 4
    if len(message) < header_size:
        raise ValueError("火山语音响应头长度非法")

    message_code = message[1]
    serialization = message[2] >> 4
    compression = message[2] & 0x0F
    payload = message[header_size:]
    message_type = message_code & 0xF0
    is_final = (message_code & 0x0F) in (0x2, 0x3)

    if message_type == 0x90:
        payload_bytes, sequence = _extract_volcengine_full_response_payload(payload)
        decoded_payload = _decode_volcengine_payload(payload_bytes, serialization, compression)
        return {
            "message_type": message_type,
            "sequence": sequence,
            "payload": decoded_payload,
            "is_final": is_final or (sequence is not None and sequence < 0),
        }

    if message_type == 0xF0:
        if len(payload) < 8:
            raise ValueError("火山语音返回的错误响应体长度不足")
        error_code = int.from_bytes(payload[0:4], "big", signed=True)
        payload_size = int.from_bytes(payload[4:8], "big")
        payload_bytes = payload[8 : 8 + payload_size]
        decoded_payload = _decode_volcengine_payload(payload_bytes, serialization, compression)
        return {
            "message_type": message_type,
            "error_code": error_code,
            "payload": decoded_payload,
            "is_final": True,
        }

    return {
        "message_type": message_type,
        "payload": _decode_volcengine_payload(payload, serialization, compression),
        "is_final": is_final,
    }


# 抖音视频 CDN 域名列表，按优先级排序
# zjcdn 灰度节点不稳定，主 CDN aweme.snssdk.com 更可靠
_DOUYIN_CDN_REPLACEMENTS: list[tuple[str, str]] = [
    # (待替换域名, 替换为)
    ("v5-dy-o-abtest.zjcdn.com", "aweme.snssdk.com"),
    ("v3-dy-o-abtest.zjcdn.com", "aweme.snssdk.com"),
    ("v10-dy-o-abtest.zjcdn.com", "aweme.snssdk.com"),
    ("v11-dy-o-abtest.zjcdn.com", "aweme.snssdk.com"),
    ("v3.douyinvod.com", "aweme.snssdk.com"),
    ("v2.douyinvod.com", "aweme.snssdk.com"),
]


def _rewrite_douyin_cdn(url: str) -> str:
    """将抖音视频 URL 替换为更稳定的 CDN 节点"""
    for old_host, new_host in _DOUYIN_CDN_REPLACEMENTS:
        if old_host in url:
            # 替换域名，保留路径和参数
            parsed = urlparse(url)
            rewritten = f"{parsed.scheme}://{new_host}{parsed.path}"
            if parsed.query:
                rewritten += f"?{parsed.query}"
            return rewritten
    return url


def download_binary(url: str, destination: Path) -> Path:
    # 优先使用稳定的 aweme.snssdk.com CDN
    url = _rewrite_douyin_cdn(url)
    response = requests.get(_normalize_media_url(url), headers=HEADERS, timeout=60, stream=True)
    response.raise_for_status()
    with destination.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)
    return destination


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    import ffmpeg

    (
        ffmpeg.input(str(video_path))
        .output(str(audio_path), acodec="libmp3lame", q=0)
        .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
    )
    return audio_path


def extract_wav_audio(video_path: Path, audio_path: Path) -> Path:
    import ffmpeg

    (
        ffmpeg.input(str(video_path))
        .output(str(audio_path), format="wav", acodec="pcm_s16le", ac=1, ar=16000)
        .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
    )
    return audio_path


def build_volcengine_request_payload(audio_path: Path, context: ExtractionContext, user_id: str) -> dict[str, Any]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        bits = wav_file.getsampwidth() * 8

    return {
        "user": {"uid": user_id},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": sample_rate,
            "bits": bits,
            "channel": channels,
            "language": os.getenv("VOLCENGINE_SPEECH_LANGUAGE", "zh-CN"),
        },
        "request": {
            "model_name": (
                context.asr_model
                or os.getenv("VOLCENGINE_SPEECH_MODEL_NAME")
                or default_model_for_provider("volcengine_speech", "asr")
                or "bigmodel"
            ),
            "enable_itn": _env_flag("VOLCENGINE_SPEECH_ENABLE_ITN", True),
            "enable_ddc": _env_flag("VOLCENGINE_SPEECH_ENABLE_DDC", False),
            "enable_punc": _env_flag("VOLCENGINE_SPEECH_ENABLE_PUNC", True),
            "show_utterances": _env_flag("VOLCENGINE_SPEECH_SHOW_UTTERANCES", True),
            "result_type": os.getenv("VOLCENGINE_SPEECH_RESULT_TYPE", "full"),
        },
    }


def iter_wav_audio_chunks(audio_path: Path, *, chunk_ms: int = 200):
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames_per_chunk = max(int(sample_rate * chunk_ms / 1000), 1)
        while True:
            chunk = wav_file.readframes(frames_per_chunk)
            if not chunk:
                break
            yield chunk


async def _transcribe_via_volcengine_websocket(
    *,
    audio_path: Path,
    config: dict[str, str],
    context: ExtractionContext,
    user_id: str,
) -> str:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("未安装 websockets 依赖，无法调用火山语音识别") from exc

    payload = build_volcengine_request_payload(audio_path, context, user_id)
    connect_headers = {
        "X-Api-App-Key": config["app_id"],
        "X-Api-Access-Key": config["access_token"],
        "X-Api-Resource-Id": config["resource_id"],
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    connect_kwargs = {
        "open_timeout": 30,
        "close_timeout": 30,
        "max_size": 8 * 1024 * 1024,
    }
    last_error: Optional[Exception] = None
    for header_arg_name in ("additional_headers", "extra_headers"):
        try:
            async with websockets.connect(
                config["ws_url"],
                **connect_kwargs,
                **{header_arg_name: connect_headers},
            ) as websocket:
                await websocket.send(build_volcengine_full_client_request(payload))
                latest_text = await _drain_volcengine_messages(websocket, latest_text="", wait_final=False)

                pending_chunk = None
                for audio_chunk in iter_wav_audio_chunks(audio_path):
                    if pending_chunk is not None:
                        await websocket.send(build_volcengine_audio_request(pending_chunk, is_final=False))
                        latest_text = await _drain_volcengine_messages(
                            websocket,
                            latest_text=latest_text,
                            wait_final=False,
                        )
                    pending_chunk = audio_chunk

                if pending_chunk is None:
                    raise RuntimeError("提取出的音频为空，无法调用火山语音识别")

                await websocket.send(build_volcengine_audio_request(pending_chunk, is_final=True))
                latest_text = await _drain_volcengine_messages(websocket, latest_text=latest_text, wait_final=True)
                return latest_text.strip()
        except TypeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError("无法建立火山语音 websocket 连接")


async def _drain_volcengine_messages(websocket: Any, *, latest_text: str, wait_final: bool) -> str:
    timeout = 15.0 if wait_final else 0.3
    while True:
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return latest_text

        parsed = parse_volcengine_server_message(message)
        if parsed.get("error_code") is not None:
            error_payload = parsed.get("payload")
            raise RuntimeError(f"火山语音识别失败({parsed['error_code']}): {_stringify_volcengine_payload(error_payload)}")

        current_text = _extract_volcengine_result_text(parsed.get("payload"))
        if current_text:
            latest_text = current_text

        if parsed.get("is_final"):
            return latest_text
        timeout = 0.05 if not wait_final else 15.0


def extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and ("text" in item or item.get("type") == "text"):
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    return ""


def extract_dashscope_message_text(payload: Any) -> str:
    if hasattr(payload, "output"):
        payload = {"output": getattr(payload, "output")}
    output = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(output, dict):
        return extract_message_text(output)
    return ""


def provider_asr_url_env_name(provider_name: str) -> str:
    env_names = {
        "bailian": "BAILIAN_ASR_URL 或 DASHSCOPE_ASR_URL",
        "doubao": "DOUBAO_ASR_URL 或 ARK_ASR_URL",
    }
    return env_names.get(provider_name, "ASR_URL")


def _join_api_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def _normalize_media_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def _extract_first_url(text: str) -> str:
    urls = re.findall(r"http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", text)
    if not urls:
        raise ValueError("未找到有效链接")
    return urls[0]


def _extract_xsec_token(url: str) -> Optional[str]:
    parsed = urlparse(url)
    token = parse_qs(parsed.query).get("xsec_token")
    return token[0] if token else None


def first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _decode_volcengine_payload(payload: bytes, serialization: int, compression: int) -> Any:
    if compression == 1 and payload:
        payload = gzip.decompress(payload)
    if serialization == 1 and payload:
        return json.loads(payload.decode("utf-8"))
    if not payload:
        return ""
    return payload.decode("utf-8", errors="ignore")


def _extract_volcengine_full_response_payload(payload: bytes) -> tuple[bytes, Optional[int]]:
    if len(payload) < 4:
        raise ValueError("火山语音返回的响应体长度不足")

    if len(payload) >= 8:
        sequence = int.from_bytes(payload[0:4], "big", signed=True)
        payload_size = int.from_bytes(payload[4:8], "big")
        if payload_size <= len(payload) - 8:
            return payload[8 : 8 + payload_size], sequence

    size_only_payload_size = int.from_bytes(payload[0:4], "big")
    if size_only_payload_size <= len(payload) - 4:
        return payload[4 : 4 + size_only_payload_size], None

    raise ValueError("火山语音返回的响应体长度不足")


def _extract_volcengine_result_text(payload: Any) -> str:
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                return text
        text = payload.get("text")
        if isinstance(text, str):
            return text
    if isinstance(payload, str):
        return payload
    return ""


def _stringify_volcengine_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if payload is None:
        return ""
    return json.dumps(payload, ensure_ascii=False)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
