# Social Post Extractor MCP

一个面向二次开发的 MCP 项目，用来从抖音和小红书链接中提取结构化信息、脚本文件和信息文件。

这个仓库基于原项目 `yzfly/douyin-mcp-server` 做了独立演进，保留 Apache 2.0 许可证，并扩展了以下能力：

- 同时支持抖音和小红书
- 支持小红书视频笔记和图文笔记
- 图文笔记支持正文提取和图片文字提取
- 统一输出 `script.md` 和 `info.json`
- 支持多 provider 切换
- 支持火山语音 `App ID + Access Token` 的 ASR 接法

## 当前能力

### 平台与内容类型

- 抖音视频
- 小红书视频笔记
- 小红书图文笔记

### 输出产物

- `script.md`
- `info.json`

### MCP 工具

- `parse_social_post_info`
- `extract_social_post_script`
- `parse_douyin_video_info`
- `get_douyin_download_link`
- `extract_douyin_text`

## Provider 说明

### ASR

- `siliconflow`
- `dashscope`
- `bailian`
- `doubao`
- `volcengine_speech`

### Vision / Cleanup

- `minimax`
- `qwen`
- `bailian`
- `doubao`
- `generic`

## 推荐配置

如果你希望整套配置尽量简单、直接、好用、快速，建议默认收敛到百炼：

- `ASR_PROVIDER=bailian`
- `ASR_MODEL=paraformer-v2`
- `VISION_PROVIDER=bailian`
- `VISION_MODEL=qwen3-vl-flash`
- `CLEAN_PROVIDER=bailian`
- `CLEAN_MODEL=qwen-flash`

其中：

- `paraformer-v2` 负责视频 ASR
- `qwen3-vl-flash` 负责小红书图文图片读字
- `qwen-flash` 只做轻整理：分段、标点、少量明显错字修正

## 快速开始

```bash
git clone <your-repo-url>
cd social-post-extractor-mcp
uv sync
```

复制环境变量模板：

```bash
cp .env.example .env
```

运行测试：

```bash
python3 -m unittest discover -s tests
```

启动 MCP：

```bash
uv run python -m social_post_extractor_mcp
```

## 环境变量

可参考 `.env.example`。

核心变量分三组：

- ASR
- Vision
- Cleanup

推荐直接使用百炼：

```bash
ASR_PROVIDER=bailian
ASR_MODEL=paraformer-v2
VISION_PROVIDER=bailian
VISION_MODEL=qwen3-vl-flash
CLEAN_PROVIDER=bailian
CLEAN_MODEL=qwen-flash
BAILIAN_API_KEY=your_api_key
```

## 许可证

本仓库继续采用 Apache 2.0 许可证，见 `LICENSE`。
