#!/usr/bin/env python3
"""MCP server for Douyin and XiaoHongShu extraction."""

from __future__ import annotations

import json
import os
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP

from .social_extractor import DEFAULT_ASR_PROVIDER, SocialExtractorService


mcp = FastMCP(
    "Social Post Extractor MCP Server",
    dependencies=["requests", "ffmpeg-python", "mcp"],
)

_SERVICE = SocialExtractorService()


def _detect_legacy_asr_provider(model: Optional[str]) -> str:
    if model and "paraformer" in model.lower():
        return "dashscope"
    return os.getenv("ASR_PROVIDER") or DEFAULT_ASR_PROVIDER


def parse_social_post_info_value(share_link: str) -> dict:
    post = _SERVICE.parse_social_post(share_link)
    return {
        "platform": post.platform,
        "content_type": post.content_type,
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
        "image_urls": post.image_urls,
        "page_url": post.page_url,
        "resolved_url": post.resolved_url,
        "status": "success",
    }


def extract_social_post_script_value(
    share_link: str,
    *,
    output_dir: Optional[str] = None,
    asr_provider: Optional[str] = None,
    asr_model: Optional[str] = None,
    vision_provider: Optional[str] = None,
    vision_model: Optional[str] = None,
    clean_provider: Optional[str] = None,
    clean_model: Optional[str] = None,
    save_raw_segments: bool = False,
) -> dict:
    return _SERVICE.extract_social_post(
        share_link,
        output_dir=output_dir,
        asr_provider=asr_provider,
        asr_model=asr_model,
        vision_provider=vision_provider,
        vision_model=vision_model,
        clean_provider=clean_provider,
        clean_model=clean_model,
        save_raw_segments=save_raw_segments,
    )


@mcp.tool()
def parse_social_post_info(share_link: str) -> str:
    """自动识别抖音或小红书链接并返回结构化信息。"""
    try:
        return json.dumps(parse_social_post_info_value(share_link), ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2)


@mcp.tool()
def extract_social_post_script(
    share_link: str,
    output_dir: Optional[str] = None,
    asr_provider: Optional[str] = None,
    asr_model: Optional[str] = None,
    vision_provider: Optional[str] = None,
    vision_model: Optional[str] = None,
    clean_provider: Optional[str] = None,
    clean_model: Optional[str] = None,
    save_raw_segments: bool = False,
) -> str:
    """
    自动识别抖音或小红书链接，并输出 script.md 与 info.json。

    默认行为：
    - 抖音/小红书视频：提取语音脚本
    - 小红书图文：提取正文和图片文字
    """
    try:
        result = extract_social_post_script_value(
            share_link,
            output_dir=output_dir,
            asr_provider=asr_provider,
            asr_model=asr_model,
            vision_provider=vision_provider,
            vision_model=vision_model,
            clean_provider=clean_provider,
            clean_model=clean_model,
            save_raw_segments=save_raw_segments,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2)


@mcp.tool()
def parse_douyin_video_info(share_link: str) -> str:
    """兼容旧接口：只处理抖音链接并返回视频信息。"""
    try:
        result = parse_social_post_info_value(share_link)
        if result["platform"] != "douyin":
            raise ValueError("该接口仅支持抖音链接")
        return json.dumps(
            {
                "video_id": result["post_id"],
                "title": result["title"],
                "download_url": result["video_url"],
                "status": "success",
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2)


@mcp.tool()
def get_douyin_download_link(share_link: str) -> str:
    """兼容旧接口：返回抖音无水印视频链接。"""
    try:
        result = parse_social_post_info_value(share_link)
        if result["platform"] != "douyin":
            raise ValueError("该接口仅支持抖音链接")
        return json.dumps(
            {
                "status": "success",
                "video_id": result["post_id"],
                "title": result["title"],
                "download_url": result["video_url"],
                "description": f"视频标题: {result['title']}",
                "usage_tip": "可以直接使用此链接下载无水印视频",
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2)


@mcp.tool()
def extract_douyin_text(
    share_link: str,
    model: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """兼容旧接口：返回抖音视频原始转写文本。"""
    try:
        if ctx:
            ctx.info("正在通过统一提取管线处理抖音视频...")
        result = extract_social_post_script_value(
            share_link,
            asr_provider=_detect_legacy_asr_provider(model),
            asr_model=model,
        )
        return result.get("raw_transcript") or result.get("script_preview") or ""
    except Exception as exc:
        if ctx:
            ctx.error(str(exc))
        raise


@mcp.prompt()
def social_post_extraction_guide() -> str:
    """统一提取工具使用说明。"""
    return """
# 社交内容提取使用指南

## 支持的平台
- 抖音视频
- 小红书视频笔记
- 小红书图文笔记

## 推荐工具
- `parse_social_post_info`: 只解析基础信息
- `extract_social_post_script`: 生成 script.md 和 info.json

## 兼容旧工具
- `parse_douyin_video_info`
- `get_douyin_download_link`
- `extract_douyin_text`

## 默认输出
- `script.md`: 整理稿 + 原始内容
- `info.json`: 结构化 metadata、模型信息和状态

## 模型切换
支持通过环境变量设置默认 provider/model，也支持在单次调用时覆盖：
- ASR: `asr_provider` / `asr_model`
- Vision: `vision_provider` / `vision_model`
- Cleanup: `clean_provider` / `clean_model`
"""


def main():
    mcp.run()


if __name__ == "__main__":
    main()
