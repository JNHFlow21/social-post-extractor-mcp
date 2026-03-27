# Social Post Extractor MCP

AI-first MCP for extracting structured content from:

- Douyin videos
- Xiaohongshu video notes
- Xiaohongshu image notes

Default outputs:

- `script.md`
- `info.json`

This repository is intended to be handed to an AI agent. Write prompts assuming the AI will read this file and perform the setup.

## What This MCP Does

Input:

- one Douyin share link
- one Xiaohongshu share link
- share text that contains one of those links

Output:

- parsed metadata
- raw transcript for videos
- OCR text for Xiaohongshu image notes
- lightly cleaned readable script
- artifacts written to disk

Supported MCP tools:

- `parse_social_post_info`
- `extract_social_post_script`
- `parse_douyin_video_info`
- `get_douyin_download_link`
- `extract_douyin_text`

## Default Decision

If you are an AI agent configuring this repo, use this default stack unless the user explicitly asks for something else:

- `ASR_PROVIDER=bailian`
- `ASR_MODEL=paraformer-v2`
- `VISION_PROVIDER=bailian`
- `VISION_MODEL=qwen3-vl-flash`
- `CLEAN_PROVIDER=bailian`
- `CLEAN_MODEL=qwen-flash`

Reason:

- `paraformer-v2` is the cheapest and fastest default for video ASR
- `qwen3-vl-flash` is a good default for Xiaohongshu image-note OCR
- `qwen-flash` is enough for light cleanup: punctuation, paragraphing, obvious typo fixes

Do not switch to heavier models by default.

## Install

If you are an AI agent, execute these steps in order:

```bash
git clone https://github.com/JNHFlow21/social-post-extractor-mcp.git
cd social-post-extractor-mcp
uv sync
```

Optional local verification:

```bash
python3 -m unittest discover -s tests
```

Run the MCP server:

```bash
uv run python -m social_post_extractor_mcp
```

## Configure

Copy the template:

```bash
cp .env.example .env
```

Use this exact default configuration:

```bash
ASR_PROVIDER=bailian
ASR_MODEL=paraformer-v2
VISION_PROVIDER=bailian
VISION_MODEL=qwen3-vl-flash
CLEAN_PROVIDER=bailian
CLEAN_MODEL=qwen-flash
BAILIAN_API_KEY=your_bailian_api_key
```

Do not commit `.env`.

## MCP Config Snippet

If you are wiring this repo into an MCP client, use a local config like this:

```json
{
  "mcpServers": {
    "douyin": {
      "command": "/bin/zsh",
      "args": [
        "-lc",
        "cd '/absolute/path/to/social-post-extractor-mcp' && exec '.venv/bin/python' -m social_post_extractor_mcp"
      ],
      "env": {
        "ASR_PROVIDER": "bailian",
        "ASR_MODEL": "paraformer-v2",
        "VISION_PROVIDER": "bailian",
        "VISION_MODEL": "qwen3-vl-flash",
        "CLEAN_PROVIDER": "bailian",
        "CLEAN_MODEL": "qwen-flash",
        "BAILIAN_API_KEY": "YOUR_BAILIAN_API_KEY"
      }
    }
  }
}
```

Use the shell wrapper form above if your MCP client launches stdio servers from a directory outside this repo. It avoids Python module resolution failures.

### Agent Reach / mcporter

If the user is using Agent Reach, the effective system config is usually:

- `~/.mcporter/mcporter.json`

Do not assume `~/.agent-reach/tools/.../config/mcporter.json` is the active config file.

Check the active source with:

```bash
mcporter config list
```

If `mcporter config list` shows the `douyin` server coming from `~/.mcporter/mcporter.json`, edit that file instead.

## How To Buy The API

This repo currently assumes Alibaba Cloud Bailian / DashScope for the default path.

Official product page:

- https://www.aliyun.com/product/bailian

Official pricing page:

- https://help.aliyun.com/zh/model-studio/model-pricing

Official first API call / API key doc:

- https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen

If you are an AI agent, the purchase flow is:

1. Ask the user to log in to Alibaba Cloud.
2. Tell the user to open the Bailian product page.
3. Tell the user to activate Bailian if it is not activated yet.
4. Tell the user to recharge their Alibaba Cloud account balance.
5. Tell the user to open Bailian console `密钥管理`.
6. Tell the user to create an API key.
7. Put that key into local config as `BAILIAN_API_KEY`.

Current official product page also advertises:

- a one-click API onboarding entry
- free token quota for new users

Important:

- Keep the API key in local config or local environment variables only.
- Do not commit the real key to Git.
- Do not paste the real key into tracked files.

## Current Official Pricing Reference

All numbers below are from Alibaba Cloud official pages and should be treated as the current default reference for this repo.

### Video ASR

`paraformer-v2`:

- `0.00008 元 / 秒`
- `36,000 秒` monthly free quota shown on the pricing page, equal to `10 小时`

`fun-asr`:

- `0.00022 元 / 秒`

Interpretation:

- `fun-asr` is about `2.75x` the price of `paraformer-v2`
- use `paraformer-v2` as the default unless the user explicitly needs a more expensive ASR model

### Vision OCR

`qwen3-vl-flash` under the current pricing page tier `0 < Token <= 32K`:

- input: `0.15 元 / 百万 Token`
- output: `1.5 元 / 百万 Token`

### Light Cleanup

`qwen-flash` under the current pricing page tier `0 < Token <= 128K`:

- input: `0.15 元 / 百万 Token`
- output: `1.5 元 / 百万 Token`

## Estimated Cost

These are rough operating estimates for the default stack.

### ASR Cost

Using `paraformer-v2` only:

- `1 minute video` ≈ `0.0048 元`
- `3 minute video` ≈ `0.0144 元`
- `5 minute video` ≈ `0.024 元`
- `10 minute video` ≈ `0.048 元`
- `100 videos x 1 minute` ≈ `0.48 元`
- `100 videos x 3 minutes` ≈ `1.44 元`
- `100 hours total` ≈ `28.8 元`

### Light Cleanup Cost

For `qwen-flash`, cleanup is usually negligible compared with ASR.

Reference estimate:

- assume `2,000` input tokens + `2,000` output tokens for one short transcript
- estimated cleanup cost ≈ `0.0033 元 / 条`

Formula:

- input cost = `input_tokens / 1,000,000 * 0.15`
- output cost = `output_tokens / 1,000,000 * 1.5`

### Xiaohongshu Image OCR Cost

For `qwen3-vl-flash`, OCR cost depends on:

- image count
- image size
- prompt tokens
- OCR output length

Use this as the operating rule:

- image-note OCR is still cheap for normal creator workflows
- if the user mainly processes videos, budget primarily by ASR
- if the user mainly processes long multi-image notes, monitor token usage from actual runs instead of guessing

## What The Code Handles vs What The Models Handle

Code handles:

- link parsing
- Douyin / Xiaohongshu detection
- note type detection
- metadata extraction
- artifact directory creation
- writing `script.md` and `info.json`

Cloud models handle:

- `paraformer-v2`: video speech-to-text
- `qwen3-vl-flash`: image text extraction
- `qwen-flash`: light readability cleanup

This is not "the LLM figures everything out by itself".
The MCP does the workflow orchestration; the models only handle recognition and light cleanup.

## Recommended Behavior For AI Agents

If you are an AI agent using this repo:

1. Prefer `extract_social_post_script` over platform-specific tools.
2. Keep `extract_douyin_text` only for backward compatibility.
3. Keep cleanup light. Do not summarize unless the user asks.
4. Preserve raw transcript and raw OCR text in artifacts.
5. Do not change the default model stack unless the user asks.
6. Do not store real API keys in tracked files.

## Minimal Verification Commands

Run these after setup:

```bash
python3 -m unittest discover -s tests
```

Example MCP smoke test:

```bash
mcporter call 'douyin.extract_social_post_script(share_link: "https://v.douyin.com/xxxxx/", output_dir: "/tmp/social-post-extract")'
```

## Related Docs

- [AGENT_REACH_INTEGRATION.md](./AGENT_REACH_INTEGRATION.md)

## License

Apache 2.0. See [LICENSE](./LICENSE).
