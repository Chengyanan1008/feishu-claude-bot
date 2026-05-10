"""
飞书 × Claude 全功能机器人（lark-oapi SDK 长连接版）

原理：SDK 主动连接飞书 WebSocket 网关，飞书推送事件过来。
      全程出站连接，无需公网 IP，无需内网穿透，直接 python bot.py 跑起来即可。

功能：
  - 流式输出：实时更新飞书卡片（打字机效果）
  - 富文本卡片：代码块高亮、Markdown、Token 统计
  - 指令系统：/help /clear /model /persona /system /status
  - 工具调用：multi-turn tool_use 循环（默认空，按需添加）

依赖：pip install lark-oapi anthropic

配置：通过环境变量传入凭据
  export FEISHU_APP_ID=cli_xxx
  export FEISHU_APP_SECRET=xxx
  export ANTHROPIC_API_KEY=sk-ant-xxx
  # 可选：企业自签名证书环境
  export INSECURE_SSL=1
"""

import concurrent.futures
import json
import os
import re
import ssl
import threading
import time
import warnings
from collections import defaultdict
from datetime import datetime

# ============================================================
# 企业代理自签名证书处理（仅当 INSECURE_SSL=1 时启用）
# 必须在 import anthropic / lark_oapi 之前执行
# ============================================================
if os.environ.get("INSECURE_SSL") == "1":
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    _orig_ssl_ctx = ssl.create_default_context

    def _unverified_ssl_context(*args, **kwargs):
        ctx = _orig_ssl_ctx(*args, **kwargs)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ssl.create_default_context = _unverified_ssl_context
    ssl._create_default_https_context = ssl._create_unverified_context

    import requests as _requests
    _orig_merge = _requests.Session.merge_environment_settings

    def _patched_merge(self, url, proxies, stream, verify, cert):
        settings = _orig_merge(self, url, proxies, stream, verify, cert)
        settings["verify"] = False
        return settings

    _requests.Session.merge_environment_settings = _patched_merge

import requests
import anthropic
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

# ============================================================
# 配置区 ← 通过环境变量注入，避免凭据进入仓库
# ============================================================
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
    raise SystemExit(
        "请设置环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET\n"
        "例如：export FEISHU_APP_ID=cli_xxx FEISHU_APP_SECRET=xxx"
    )
if not ANTHROPIC_API_KEY:
    raise SystemExit("请设置环境变量 ANTHROPIC_API_KEY")

STREAM_UPDATE_INTERVAL = 0.8   # 流式更新卡片的间隔（秒），过短会触发飞书限流
MAX_HISTORY_TURNS = 10          # 每个用户保留的最大对话轮数（1轮 = 用户+助手各1条）
# ============================================================

# 可用模型
MODELS: dict[str, tuple[str, str, str]] = {
    "haiku":  ("claude-haiku-4-5",  "Haiku 4.5",  "💚"),
    "sonnet": ("claude-sonnet-4-6", "Sonnet 4.6", "💙"),
    "opus":   ("claude-opus-4-7",   "Opus 4.7",   "💜"),
}
DEFAULT_MODEL = "opus"

# 预设角色
PERSONAS: dict[str, dict] = {
    "default": {
        "name": "智能助手", "icon": "🤖", "color": "blue",
        "prompt": "你是一个智能助手，请用中文回复，语言简洁清晰。",
    },
    "coder": {
        "name": "编程专家", "icon": "👨‍💻", "color": "indigo",
        "prompt": (
            "你是一个资深程序员助手，擅长代码分析、调试和架构设计。"
            "提供清晰的代码示例，并解释关键逻辑和设计决策。"
        ),
    },
    "translator": {
        "name": "翻译助手", "icon": "🌐", "color": "turquoise",
        "prompt": (
            "你是一个专业翻译助手。"
            "用户发中文时翻译成英文，发英文时翻译成中文，"
            "并简要说明语言特点或翻译难点。"
        ),
    },
    "analyst": {
        "name": "数据分析师", "icon": "📊", "color": "orange",
        "prompt": (
            "你是一名数据分析师助手，擅长数据分析、SQL、Python 数据处理。"
            "回答时提供结构化分析思路和可执行代码。"
        ),
    },
    "writer": {
        "name": "写作助手", "icon": "✍️", "color": "violet",
        "prompt": (
            "你是一个专业写作助手，擅长文案撰写、内容优化、语言润色。"
            "风格简洁有力，注重表达的准确性和美感。"
        ),
    },
}

# ============================================================
# 初始化
# ============================================================
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

feishu_client = (
    lark.Client.builder()
    .app_id(FEISHU_APP_ID)
    .app_secret(FEISHU_APP_SECRET)
    .build()
)


def _fetch_bot_open_id() -> str:
    """启动时取一次 bot 自己的 open_id，用于群聊 @ 识别。"""
    try:
        token_resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
            timeout=5,
        )
        token = token_resp.json().get("tenant_access_token", "")
        info_resp = requests.get(
            "https://open.feishu.cn/open-apis/bot/v3/info",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        return info_resp.json().get("bot", {}).get("open_id", "")
    except Exception as e:
        print(f"[warn] failed to get bot open_id: {e}", flush=True)
        return ""


BOT_OPEN_ID: str = _fetch_bot_open_id()
print(f"[init] BOT_OPEN_ID={BOT_OPEN_ID!r}", flush=True)

_user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

# 消息去重：飞书有时会重复推送同一事件
_seen_msg_ids: set[str] = set()
_seen_msg_lock = threading.Lock()
_SEEN_MSG_MAX = 500

# 有界线程池：防止并发请求过多导致资源耗尽
_chat_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=20, thread_name_prefix="chat"
)


def _check_and_mark_seen(msg_id: str) -> bool:
    """已处理过返回 True（重复），首次见到返回 False。"""
    with _seen_msg_lock:
        if msg_id in _seen_msg_ids:
            return True
        if len(_seen_msg_ids) > _SEEN_MSG_MAX:
            _seen_msg_ids.clear()
        _seen_msg_ids.add(msg_id)
        return False


def _default_state() -> dict:
    return {"history": [], "model": DEFAULT_MODEL, "persona": "default", "custom_system": None}


user_state: dict[str, dict] = defaultdict(_default_state)


# ============================================================
# 飞书消息 API
# ============================================================

def feishu_send_card(receive_id: str, receive_id_type: str, card: dict) -> str | None:
    """发送卡片消息，返回 message_id"""
    req = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type("interactive")
            .content(json.dumps(card))
            .build()
        )
        .build()
    )
    resp = feishu_client.im.v1.message.create(req)
    if resp.success():
        return resp.data.message_id
    return None


def feishu_reply_card(parent_message_id: str, card: dict) -> str | None:
    """在消息下方回复卡片，返回 message_id"""
    req = (
        ReplyMessageRequest.builder()
        .message_id(parent_message_id)
        .request_body(
            ReplyMessageRequestBody.builder()
            .msg_type("interactive")
            .content(json.dumps(card))
            .build()
        )
        .build()
    )
    resp = feishu_client.im.v1.message.reply(req)
    if resp.success():
        return resp.data.message_id
    return None


def feishu_update_card(message_id: str, card: dict) -> None:
    """更新已有卡片（实现流式打字效果）。注意是 patch 不是 update。"""
    req = (
        PatchMessageRequest.builder()
        .message_id(message_id)
        .request_body(
            PatchMessageRequestBody.builder()
            .content(json.dumps(card))
            .build()
        )
        .build()
    )
    resp = feishu_client.im.v1.message.patch(req)
    if not resp.success():
        print(f"[update_card ERROR] code={resp.code} msg={resp.msg}")


# ============================================================
# 卡片构建器（飞书卡片 2.0）
# ============================================================

def _text_to_elements(text: str) -> list:
    """
    将文本解析为卡片 2.0 markdown elements。
    代码块保留 ``` 格式由 markdown 渲染，patch 接口不支持独立 code_block 标签。
    单元素 > 3000 字符会被截断，所以按代码块边界切片。
    """
    MAX_CHUNK = 3000
    elements: list = []

    parts = re.split(r"(```[\s\S]*?```)", text)
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        while len(chunk) > MAX_CHUNK:
            elements.append({"tag": "markdown", "content": chunk[:MAX_CHUNK]})
            chunk = chunk[MAX_CHUNK:]
        elements.append({"tag": "markdown", "content": chunk})

    if not elements:
        elements.append({"tag": "markdown", "content": text or "（空响应）"})
    return elements


def _card(header: dict, elements: list) -> dict:
    """卡片 2.0 骨架"""
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": header,
        "body": {"elements": elements},
    }


def card_thinking() -> dict:
    return _card(
        {"title": {"tag": "plain_text", "content": "⏳ 正在思考..."}, "template": "grey"},
        [{"tag": "markdown", "content": "Claude 正在思考，请稍候..."}],
    )


def card_streaming(text: str, model_key: str) -> dict:
    """流式输出中（末尾附打字光标 ▌）"""
    _, model_name, model_icon = MODELS[model_key]
    elements = _text_to_elements(text + " ▌")
    elements.append({"tag": "markdown", "content": f"---\n{model_icon} {model_name}  ·  输出中..."})
    return _card(
        {"title": {"tag": "plain_text", "content": "🤖 Claude"}, "template": "wathet"},
        elements,
    )


def card_response(
    text: str,
    model_key: str,
    persona_key: str = "default",
    input_tokens: int = 0,
    output_tokens: int = 0,
    elapsed: float = 0.0,
) -> dict:
    """最终完整回复卡片"""
    _, model_name, model_icon = MODELS[model_key]
    persona = PERSONAS.get(persona_key, PERSONAS["default"])
    elements = _text_to_elements(text)
    footer = "  ·  ".join([
        f"{model_icon} {model_name}",
        f"📥 {input_tokens}  📤 {output_tokens} tokens",
        f"⏱ {elapsed:.1f}s",
        datetime.now().strftime("%H:%M"),
    ])
    elements.append({"tag": "markdown", "content": f"---\n{footer}"})
    return _card(
        {"title": {"tag": "plain_text", "content": f"{persona['icon']} {persona['name']}"}, "template": persona["color"]},
        elements,
    )


def card_error(error_msg: str) -> dict:
    return _card(
        {"title": {"tag": "plain_text", "content": "❌ 出错了"}, "template": "red"},
        [{"tag": "markdown", "content": f"```\n{error_msg}\n```"}],
    )


def card_info(title: str, content: str, color: str = "green") -> dict:
    """通用信息卡片（指令响应用）"""
    return _card(
        {"title": {"tag": "plain_text", "content": title}, "template": color},
        [{"tag": "markdown", "content": content}],
    )


def card_tool_call(tool_name: str, tool_input: dict, pre_text: str = "") -> dict:
    """工具调用过渡卡片：显示正在执行的工具名称和参数"""
    input_str = json.dumps(tool_input, ensure_ascii=False, indent=2)
    elements = []
    if pre_text.strip():
        elements += _text_to_elements(pre_text)
    elements.append({
        "tag": "markdown",
        "content": f"---\n⚙️ **调用工具：** `{tool_name}`\n```json\n{input_str}\n```",
    })
    return _card(
        {"title": {"tag": "plain_text", "content": "🔧 工具调用中..."}, "template": "yellow"},
        elements,
    )


# ============================================================
# 工具系统
# ============================================================
# 在这里声明你想让 Claude 调用的工具。默认为空——bot 退化成纯聊天机器人。
# 每个工具需要：
#   1) 在 TOOL_DEFINITIONS 里声明 schema
#   2) 在 TOOL_HANDLERS 里注册 Python 实现
#
# 示例（取消注释即启用）：
#
# def _tool_run_sql(sql: str, timeout_seconds: int = 60) -> str:
#     # TODO: 接入你自己的查询引擎，返回字符串结果
#     return f"模拟执行: {sql}"
#
# TOOL_DEFINITIONS = [
#     {
#         "name": "run_sql",
#         "description": "执行 SQL 查询，返回结果",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "sql": {"type": "string", "description": "SQL 语句"},
#                 "timeout_seconds": {"type": "integer", "description": "超时秒数，默认 60"},
#             },
#             "required": ["sql"],
#         },
#     },
# ]
# TOOL_HANDLERS = {"run_sql": _tool_run_sql}

TOOL_DEFINITIONS: list[dict] = []
TOOL_HANDLERS: dict[str, callable] = {}


def _execute_tool(name: str, tool_input: dict) -> str:
    fn = TOOL_HANDLERS.get(name)
    if fn is None:
        return f"未知工具: {name}"
    try:
        return str(fn(**tool_input))
    except Exception as e:
        return f"工具执行出错: {e}"


# ============================================================
# 指令系统
# ============================================================

def _cmd_help(args: str, uid: str) -> dict:
    content = """**📌 所有可用指令：**

| 指令 | 说明 |
| --- | --- |
| `/help` | 显示此帮助 |
| `/clear` | 清除对话历史，开始新对话 |
| `/status` | 查看当前配置和对话状态 |
| `/model <名称>` | 切换模型 |
| `/persona <名称>` | 切换预设角色（同时清除历史） |
| `/system <提示词>` | 自定义系统提示词（覆盖角色） |
| `/system reset` | 恢复角色默认提示词 |

**🤖 可用模型：**
- `haiku` — 💚 速度最快，适合简单问答
- `sonnet` — 💙 速度与智能均衡（推荐）
- `opus` — 💜 最强智能，适合复杂任务

**🎭 可用角色：**
- `default` 🤖 通用智能助手
- `coder` 👨‍💻 编程专家
- `translator` 🌐 翻译助手
- `analyst` 📊 数据分析师
- `writer` ✍️ 写作助手"""
    return card_info("📋 指令帮助", content, "green")


def _cmd_clear(args: str, uid: str) -> dict:
    with _user_locks[uid]:
        user_state[uid]["history"] = []
    return card_info("✅ 对话已清除", "历史记录已清空，开始全新对话！", "green")


def _cmd_status(args: str, uid: str) -> dict:
    state = user_state[uid]
    _, model_name, model_icon = MODELS[state["model"]]
    persona = PERSONAS.get(state["persona"], PERSONAS["default"])
    turns = len(state["history"]) // 2
    sys_prompt = state["custom_system"] or persona["prompt"]
    preview = sys_prompt[:120] + "..." if len(sys_prompt) > 120 else sys_prompt
    content = f"""**当前配置：**

- **模型：** {model_icon} {model_name}
- **角色：** {persona['icon']} {persona['name']}
- **对话轮数：** {turns} / {MAX_HISTORY_TURNS}
- **系统提示词：**
  {preview}"""
    return card_info("📊 当前状态", content, "blue")


def _cmd_model(args: str, uid: str) -> dict:
    key = args.strip().lower()
    if key not in MODELS:
        opts = "\n".join([f"- `{k}` — {v[2]} {v[1]}" for k, v in MODELS.items()])
        return card_info("❌ 未知模型", f"可用模型：\n{opts}", "red")
    with _user_locks[uid]:
        user_state[uid]["model"] = key
    _, name, icon = MODELS[key]
    return card_info("✅ 模型已切换", f"当前模型：{icon} **{name}**", "green")


def _cmd_persona(args: str, uid: str) -> dict:
    key = args.strip().lower()
    if key not in PERSONAS:
        opts = "\n".join([f"- `{k}` — {v['icon']} {v['name']}" for k, v in PERSONAS.items()])
        return card_info("❌ 未知角色", f"可用角色：\n{opts}", "red")
    with _user_locks[uid]:
        state = user_state[uid]
        state["persona"] = key
        state["custom_system"] = None
        state["history"] = []
    p = PERSONAS[key]
    return card_info("✅ 角色已切换", f"当前角色：{p['icon']} **{p['name']}**\n\n对话历史已自动清除。", p["color"])


def _cmd_system(args: str, uid: str) -> dict:
    args = args.strip()
    if args.lower() == "reset":
        with _user_locks[uid]:
            user_state[uid]["custom_system"] = None
        return card_info("✅ 已恢复", "系统提示词已恢复为当前角色默认值。", "green")
    if not args:
        return card_info("❌ 参数缺失", "用法：\n- `/system <自定义提示词>`\n- `/system reset` — 恢复默认", "red")
    with _user_locks[uid]:
        user_state[uid]["custom_system"] = args
    preview = args[:200] + "..." if len(args) > 200 else args
    return card_info("✅ 系统提示词已设置", f"**当前提示词：**\n{preview}", "green")


COMMAND_MAP = {
    "/help":    _cmd_help,
    "/clear":   _cmd_clear,
    "/status":  _cmd_status,
    "/model":   _cmd_model,
    "/persona": _cmd_persona,
    "/system":  _cmd_system,
}


def try_handle_command(text: str, uid: str) -> dict | None:
    """若文本是 /指令 则执行并返回卡片，否则返回 None。"""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    handler = COMMAND_MAP.get(cmd)
    if not handler:
        return card_info("❓ 未知指令", f"指令 `{cmd}` 不存在。\n发送 `/help` 查看所有可用指令。", "yellow")
    return handler(args, uid)


# ============================================================
# 流式对话（含 multi-turn tool_use 循环）
# ============================================================

def _block_to_dict(b):
    """SDK 返回的 content block 不能直接塞回 messages，必须剥成纯字典。"""
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {"type": b.type}


def _stream_chat(uid: str, user_text: str, reply_fn, update_fn) -> None:
    """
    流式调用 Claude，实时更新飞书卡片。支持 tool_use 多轮循环。
    reply_fn(card) → message_id   首次发送占位卡片
    update_fn(msg_id, card)        后续更新同一条消息
    """
    state = user_state[uid]
    model_key = state["model"]
    model_id = MODELS[model_key][0]
    persona_key = state["persona"]
    system_prompt = state["custom_system"] or PERSONAS[persona_key]["prompt"]

    with _user_locks[uid]:
        state["history"].append({"role": "user", "content": user_text})
        if len(state["history"]) > MAX_HISTORY_TURNS * 2:
            state["history"] = state["history"][-(MAX_HISTORY_TURNS * 2):]
        history_snapshot = list(state["history"])

    reply_msg_id = reply_fn(card_thinking())
    if not reply_msg_id:
        print("[stream_chat] reply_msg_id is None, abort")
        return

    start_time = time.time()
    api_messages = list(history_snapshot)
    total_input_tokens = 0
    total_output_tokens = 0
    tools = TOOL_DEFINITIONS

    try:
        while True:
            accumulated = ""
            last_update_time = time.time()

            stream_kwargs = dict(
                model=model_id,
                max_tokens=4096,
                system=system_prompt,
                messages=api_messages,
            )
            if tools:
                stream_kwargs["tools"] = tools

            with claude_client.messages.stream(**stream_kwargs) as stream:
                for chunk in stream.text_stream:
                    accumulated += chunk
                    now = time.time()
                    if now - last_update_time >= STREAM_UPDATE_INTERVAL:
                        update_fn(reply_msg_id, card_streaming(accumulated, model_key))
                        last_update_time = now
                final_msg = stream.get_final_message()

            total_input_tokens += final_msg.usage.input_tokens
            total_output_tokens += final_msg.usage.output_tokens

            if final_msg.stop_reason == "tool_use":
                assistant_content = [_block_to_dict(b) for b in final_msg.content]
                api_messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in final_msg.content:
                    if block.type == "tool_use":
                        update_fn(reply_msg_id, card_tool_call(block.name, block.input, accumulated))
                        result = _execute_tool(block.name, block.input)
                        print(f"[tool] {block.name} → {str(result)[:300]}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })

                api_messages.append({"role": "user", "content": tool_results})
                # 继续下一轮，让 Claude 根据工具结果生成回复

            else:
                final_text = "".join(
                    b.text for b in final_msg.content if hasattr(b, "text")
                ) or accumulated

                elapsed = time.time() - start_time
                update_fn(
                    reply_msg_id,
                    card_response(final_text, model_key, persona_key,
                                  input_tokens=total_input_tokens,
                                  output_tokens=total_output_tokens,
                                  elapsed=elapsed),
                )
                with _user_locks[uid]:
                    state["history"].append({"role": "assistant", "content": final_text})
                break

    except Exception as e:
        import traceback
        print(f"[stream_chat ERROR] {e}")
        traceback.print_exc()
        update_fn(reply_msg_id, card_error(str(e)))
        with _user_locks[uid]:
            if state["history"] and state["history"][-1]["role"] == "user":
                state["history"].pop()


# ============================================================
# 飞书事件处理
# ============================================================

def on_message_receive(data: lark.P2ImMessageReceiveV1) -> None:
    """SDK 长连接收到消息时触发，丢到线程池避免阻塞事件循环。"""
    _chat_executor.submit(_handle_message, data)


def _handle_message(data: lark.P2ImMessageReceiveV1) -> None:
    msg = data.event.message
    sender = data.event.sender

    if msg.message_type != "text":
        return

    message_id = msg.message_id
    chat_type = msg.chat_type        # "p2p" 或 "group"
    is_group = chat_type != "p2p"
    uid = sender.sender_id.open_id

    # 飞书有时重复推送同一事件，丢弃重复
    if _check_and_mark_seen(message_id):
        return

    try:
        raw = json.loads(msg.content)
        text = raw.get("text", "").strip()
    except Exception as e:
        print(f"[handle] parse content failed: {e}", flush=True)
        return

    # 群聊中只响应 @机器人 的消息
    if is_group:
        mentions = msg.mentions or []
        if not BOT_OPEN_ID or not any(
            getattr(m.id, "open_id", None) == BOT_OPEN_ID for m in mentions
        ):
            return

    # 去除 @机器人 占位符，保留正文
    text = re.sub(r"@_user_\d+|<at[^>]*>[^<]*</at>", "", text).strip()

    if not text or not uid:
        return

    # ---- 指令处理 ----
    result_card = try_handle_command(text, uid)
    if result_card is not None:
        if is_group:
            feishu_reply_card(message_id, result_card)
        else:
            feishu_send_card(uid, "open_id", result_card)
        return

    # ---- 流式对话 ----
    if is_group:
        def reply_fn(card): return feishu_reply_card(message_id, card)
    else:
        def reply_fn(card): return feishu_send_card(uid, "open_id", card)

    _stream_chat(uid, text, reply_fn, feishu_update_card)


# ============================================================
# 启动
# ============================================================

event_dispatcher = (
    lark.EventDispatcherHandler.builder("", "")   # 长连接模式无需 encrypt_key / token
    .register_p2_im_message_receive_v1(on_message_receive)
    .build()
)

if __name__ == "__main__":
    print("=" * 55)
    print("  飞书 × Claude 全功能机器人（WebSocket 长连接）")
    print("  无需公网 IP，无需内网穿透，直接运行即可")
    print("=" * 55)
    print("  发送 /help 查看所有指令")
    print()

    ws_client = lark.ws.Client(
        FEISHU_APP_ID,
        FEISHU_APP_SECRET,
        event_handler=event_dispatcher,
        log_level=lark.LogLevel.INFO,
    )
    ws_client.start()   # 阻塞，保持长连接，自动重连
