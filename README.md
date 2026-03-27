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

如果你希望视频转写和图文整理都走火山体系，建议这样拆：

- `ASR_PROVIDER=volcengine_speech`
- `VISION_PROVIDER=doubao`
- `CLEAN_PROVIDER=doubao`

其中：

- `volcengine_speech` 使用火山语音服务的 `App ID + Access Token`
- `doubao` 使用方舟 `API Key`

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

火山语音识别需要：

```bash
ASR_PROVIDER=volcengine_speech
VOLCENGINE_SPEECH_APP_ID=your_app_id
VOLCENGINE_SPEECH_ACCESS_TOKEN=your_access_token
VOLCENGINE_SPEECH_RESOURCE_ID=volc.seedasr.sauc.duration
VOLCENGINE_SPEECH_WS_URL=wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async
```

豆包视觉与清理可使用：

```bash
VISION_PROVIDER=doubao
VISION_MODEL=doubao-1-5-vision-pro-32k-250115
CLEAN_PROVIDER=doubao
CLEAN_MODEL=doubao-1-5-pro-32k-250115
ARK_API_KEY=your_ark_api_key
```

## 许可证

本仓库继续采用 Apache 2.0 许可证，见 `LICENSE`。
