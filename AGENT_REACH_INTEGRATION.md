# Agent Reach Integration

## 目的

让 `Agent Reach` 通过 `mcporter` 直接调用这个 MCP，处理：

- 抖音视频链接
- 小红书视频链接
- 小红书图文笔记链接

默认输出：

- `script.md`
- `info.json`

## 实际调用链路

调用链路如下：

`Agent Reach skill` -> `mcporter` -> `douyin` MCP server alias -> `social_post_extractor_mcp`

其中：

- `Agent Reach` 负责让大模型知道“该调用哪个工具”
- `MCP` 负责平台解析、统一编排、写出产物
- 云端模型负责识别与轻整理

## 哪部分是代码处理，哪部分是模型处理

不是“全靠大模型自己想”，也不是“全靠本地代码硬处理”。

代码负责：

- 识别抖音 / 小红书链接
- 解析页面和元数据
- 判断是视频还是图文
- 组织输出目录
- 生成 `script.md` 和 `info.json`

云端模型负责：

- `paraformer-v2`：视频语音转文字
- `qwen3-vl-flash`：小红书图文图片读字
- `qwen-flash`：轻整理，只做分段、标点、少量明显错字修正

## 默认配置

推荐默认栈：

```json
{
  "mcpServers": {
    "douyin": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "social_post_extractor_mcp"],
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

## 兼容性说明

为了兼容 `Agent Reach` 里原来的调用方式，保留了旧工具名：

- `parse_douyin_video_info`
- `get_douyin_download_link`
- `extract_douyin_text`

同时新增统一工具：

- `parse_social_post_info`
- `extract_social_post_script`

这意味着原有 skill 不需要改提示词，也能继续工作；如果要利用小红书和统一产物，优先调用新工具。

## 已验证结果

已通过真实 `mcporter` / MCP 链路验证：

- 小红书图文：成功
- 抖音视频：成功
- 小红书视频：成功

验证重点：

- `Agent Reach` 调用的 `douyin` server alias 可正常列出 schema
- MCP 默认配置已切到百炼
- 视频 ASR 不再走本地下载/转码慢路径

## 安全约束

敏感信息不要放进仓库。

建议：

- API Key 只放在本地 `mcporter.json` 或本地环境变量
- 仓库只保留 `.env.example`
- 不提交 `.env`
- 不提交本地 `config/`

如果密钥曾经出现在聊天记录、终端历史或截图里，建议尽快轮换。
