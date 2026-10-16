"""Per-user prompts scoped to one AstrBot conversation."""

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from time import monotonic

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register

PLUGIN_NAME = "astrbot_plugin_user_prompts"
PLUGIN_VERSION = "1.0.0"
PLUGIN_REPO = "https://github.com/xmlans/astrbot_plugin_user_prompts"
LEGACY_PLUGIN_NAME = "astrbot_plugin_user_prompt"
DEFAULT_MAX_LENGTH = 1000
INPUT_TIMEOUT = 300
COMMAND_HANDLED_KEY = "_user_prompt_command_handled"
PROMPT_COMMAND = re.compile(r"\A/?prompt(?:@(?P<bot>[A-Za-z0-9_]+))?(?=\s|\Z)")


def message_text(event: AstrMessageEvent) -> str:
    """Prefer the original message so wake-prefix processing cannot lose lines."""
    message_obj = getattr(event, "message_obj", None)
    raw = getattr(message_obj, "raw_message", None)
    original = getattr(raw, "effective_message", None) or getattr(raw, "message", None)
    text = getattr(original, "text", None)
    if not isinstance(text, str):
        text = getattr(message_obj, "message_str", None)
    if not isinstance(text, str):
        text = event.get_message_str()
    return text.strip()


class PendingPromptFilter(filter.CustomFilter):
    """Activate the early listener only for users who explicitly entered input mode."""

    pending: dict[tuple[str, str], tuple[str, float]] = {}
    last_prune = 0.0

    @staticmethod
    def identity(event: AstrMessageEvent) -> tuple[str, str]:
        return event.unified_msg_origin, str(event.get_sender_id())

    @classmethod
    def matches(cls, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        now = monotonic()
        if now - cls.last_prune >= 60:
            for key, (_, deadline) in tuple(cls.pending.items()):
                if now >= deadline:
                    cls.pending.pop(key, None)
            cls.last_prune = now
        key = cls.identity(event)
        waiting = cls.pending.get(key)
        if waiting is None:
            return False
        if now >= waiting[1]:
            cls.pending.pop(key, None)
            return False
        # Commands remain commands while waiting; in particular show may inspect
        # the old prompt and clear may cancel input without being saved as text.
        if UserPromptPlugin._is_prompt_command(event):
            return False
        prefixes = cfg.get("wake_prefix", ["/"])
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        source = message_text(event)
        if any(isinstance(p, str) and p and source.startswith(p) for p in prefixes):
            return False
        return True

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        return self.matches(event, cfg)


@register(PLUGIN_NAME, "xmlans", "用户为当前对话设置专属提示词", PLUGIN_VERSION, PLUGIN_REPO)
class UserPromptPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        # Keep AstrBot's live configuration object, including an initially empty dict.
        self.config = config
        self.data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.storage_file = self.data_dir / "prompts.json"
        self._prompts: dict[str, str] = {}
        self._pending: dict[tuple[str, str], tuple[str, float]] = {}
        # AstrBot creates custom filters when registering handlers, before the
        # plugin instance. Both entry points therefore share this module's map.
        PendingPromptFilter.pending = self._pending
        PendingPromptFilter.last_prune = 0.0
        self._storage_error = False
        self._write_lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            source_file = self.storage_file
            if not source_file.exists():
                source_file = self.data_dir.with_name(LEGACY_PLUGIN_NAME) / "prompts.json"
            if not source_file.exists():
                return
            payload = json.loads(source_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                raise ValueError("Unsupported prompt storage format")
            prompts = payload.get("prompts")
            if not isinstance(prompts, dict):
                raise ValueError("Invalid prompt records")
            for key, value in prompts.items():
                identity = json.loads(key)
                if (
                    not isinstance(identity, list)
                    or len(identity) != 3
                    or not all(isinstance(part, str) and part for part in identity)
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    raise ValueError("Invalid prompt record")
            if source_file != self.storage_file:
                # Keep the old file intact; subsequent loads use the new directory.
                self._write_snapshot(prompts)
            self._prompts = prompts
        except (OSError, ValueError, TypeError) as exc:
            # Preserve an unreadable file instead of overwriting users' existing data.
            self._storage_error = True
            logger.error(
                "用户提示词数据读取失败，已暂停读写，请修复后重载插件 (%s)", type(exc).__name__
            )

    def _write_snapshot(self, prompts: dict[str, str]) -> None:
        """Write beside the destination and replace atomically on the same filesystem."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.data_dir, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump({"version": 1, "prompts": prompts}, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.storage_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def _save(self, prompts: dict[str, str]) -> None:
        # The caller holds _write_lock. On cancellation, finish the thread before
        # releasing that lock so a delayed write cannot replace a newer snapshot.
        task = asyncio.create_task(asyncio.to_thread(self._write_snapshot, prompts))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            self._prompts = prompts
            raise
        self._prompts = prompts

    @staticmethod
    def _key(event: AstrMessageEvent, cid: str) -> str:
        # JSON encoding keeps separators inside platform/user identifiers unambiguous.
        return json.dumps([event.unified_msg_origin, str(event.get_sender_id()), cid])

    @staticmethod
    def _user_label(event: AstrMessageEvent) -> str:
        return f"{event.get_platform_id()}:{event.get_sender_id()}"

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        if not self.config.get("enable_whitelist", False):
            return True
        whitelist = self.config.get("whitelist", [])
        if not isinstance(whitelist, list):
            return False
        entries = {str(item).strip() for item in whitelist if isinstance(item, (str, int))}
        return str(event.get_sender_id()) in entries or self._user_label(event) in entries

    def _max_length(self) -> int | None:
        if not self.config.get("enable_length_limit", True):
            return None
        value = self.config.get("max_prompt_length", DEFAULT_MAX_LENGTH)
        if isinstance(value, bool):
            return DEFAULT_MAX_LENGTH
        try:
            limit = int(value)
        except (TypeError, ValueError, OverflowError):
            return DEFAULT_MAX_LENGTH
        return limit if limit > 0 else DEFAULT_MAX_LENGTH

    def _inactive_reason(self, event: AstrMessageEvent, prompt: str = "") -> str:
        if not self.config.get("enable", True):
            return "管理员已关闭自定义提示词功能。"
        if not self._is_allowed(event):
            return f"你不在提示词白名单中，请联系管理员添加：{self._user_label(event)}"
        limit = self._max_length()
        if limit is not None and len(prompt) > limit:
            return f"提示词共 {len(prompt)} 字，超过当前 {limit} 字限制，请缩短后重新设置。"
        return ""

    async def _current_cid(self, event: AstrMessageEvent, create: bool = False) -> str | None:
        manager = self.context.conversation_manager
        cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
        if not cid and create:
            cid = await manager.new_conversation(event.unified_msg_origin)
        return cid

    @staticmethod
    def _arguments(event: AstrMessageEvent) -> str:
        # AstrBot removes the wake prefix before dispatch. Read the full message
        # ourselves because its parameter parser collapses whitespace and newlines.
        parts = event.get_message_str().strip().split(maxsplit=1)
        return parts[1].strip() if len(parts) == 2 else ""

    @staticmethod
    def _is_prompt_command(event: AstrMessageEvent) -> bool:
        """Recognize only a complete command token, including Telegram bot suffixes."""
        if not event.is_at_or_wake_command:
            return False
        match = PROMPT_COMMAND.match(event.get_message_str().lstrip())
        if match is None:
            return False
        bot = match.group("bot")
        if bot is None:
            return True
        return (
            event.get_platform_name() == "telegram"
            and bot.casefold() == str(event.get_self_id()).lstrip("@").casefold()
        )

    async def _handle_prompt(self, event: AstrMessageEvent) -> str:
        if not event.get_sender_id() or not event.unified_msg_origin:
            return "无法识别你的用户或会话信息，请在正常聊天会话中使用 /prompt。"
        if self._storage_error:
            return "提示词存储暂不可用，请联系管理员检查数据文件并重载插件。"

        arguments = self._arguments(event)
        if not arguments:
            reason = self._inactive_reason(event)
            if reason:
                return reason
            async with self._write_lock:
                cid = await self._current_cid(event, create=True)
                if not cid:
                    return "无法获取当前对话，请稍后重试。"
                self._pending[PendingPromptFilter.identity(event)] = (
                    cid,
                    monotonic() + INPUT_TIMEOUT,
                )
            limit = self._max_length()
            length_label = f"{limit} 字" if limit is not None else "不限制"
            return (
                "请发送你想要添加/修改的提示词。\n"
                f"支持多行，字数上限：{length_label}。请在 5 分钟内发送。\n"
                "/prompt show：查看\n"
                "/prompt clear：清除并退出输入\n"
                "仅当前对话生效；开启新会话后不会继承。\n"
                f"用户标识：{self._user_label(event)}"
            )

        if arguments == "show":
            cid = await self._current_cid(event)
            prompt = self._prompts.get(self._key(event, cid), "") if cid else ""
            if not prompt:
                return "你尚未为当前对话设置提示词。发送 /prompt 开始设置。"
            status = self._inactive_reason(event, prompt) or "正在生效"
            return f"你的当前对话提示词（{len(prompt)} 字）：\n{prompt}\n状态：{status}"

        # Even if access was revoked, users may remove their own saved prompt.
        if arguments == "clear":
            async with self._write_lock:
                was_waiting = self._pending.pop(PendingPromptFilter.identity(event), None)
                cid = await self._current_cid(event)
                key = self._key(event, cid) if cid else None
                if key not in self._prompts:
                    if was_waiting:
                        return "已退出提示词输入，当前对话没有已保存的个人提示词。"
                    return "你尚未为当前对话设置提示词，无需清除。"
                snapshot = self._prompts.copy()
                snapshot.pop(key)
                await self._save(snapshot)
            return "已清除你在当前对话中的提示词。"

        reason = self._inactive_reason(event, arguments)
        if reason:
            return reason

        async with self._write_lock:
            cid = await self._current_cid(event, create=True)
            if not cid:
                return "无法获取当前对话，请稍后重试。"
            snapshot = self._prompts.copy()
            snapshot[self._key(event, cid)] = arguments
            await self._save(snapshot)
            self._pending.pop(PendingPromptFilter.identity(event), None)
        return (
            f"已保存你的提示词（{len(arguments)} 字），从下一条 AI 对话起持续生效。\n"
            "仅作用于你在当前对话中的请求；开启新会话后不会继承。"
        )

    async def _command_reply(self, event: AstrMessageEvent) -> str:
        try:
            return await self._handle_prompt(event)
        except Exception as exc:
            # Do not expose or log the prompt itself when storage/framework calls fail.
            logger.error("用户提示词操作失败 (%s)", type(exc).__name__)
            return "提示词操作失败，未确认保存成功，请稍后重试或联系管理员检查日志。"

    @staticmethod
    def _stop_input_event(event: AstrMessageEvent) -> None:
        event.should_call_llm(False)
        event.set_extra(COMMAND_HANDLED_KEY, True)
        event.stop_event()

    async def _pending_reply(self, event: AstrMessageEvent) -> str | None:
        key = PendingPromptFilter.identity(event)
        try:
            async with self._write_lock:
                waiting = self._pending.get(key)
                if waiting is None:
                    return None
                if monotonic() >= waiting[1]:
                    self._pending.pop(key, None)
                    return None
                self._stop_input_event(event)
                cid = await self._current_cid(event)
                if cid != waiting[0]:
                    self._pending.pop(key, None)
                    return "当前对话已切换，已退出提示词输入。请重新发送 /prompt。"
                if self._storage_error:
                    self._pending.pop(key, None)
                    return "提示词存储暂不可用，请联系管理员检查数据文件并重载插件。"
                reason = self._inactive_reason(event)
                if reason:
                    self._pending.pop(key, None)
                    return reason
                text = message_text(event)
                if not text:
                    return "请发送文字提示词，支持多行；仍在等待输入。"
                reason = self._inactive_reason(event, text)
                if reason:
                    return reason + "\n请缩短后重新发送；仍在等待输入。"
                snapshot = self._prompts.copy()
                snapshot[self._key(event, cid)] = text
                await self._save(snapshot)
                self._pending.pop(key, None)
            return (
                f"已保存你的提示词（{len(text)} 字），从下一条 AI 对话起持续生效。\n"
                "仅作用于你在当前对话中的请求；开启新会话后不会继承。"
            )
        except Exception as exc:
            self._stop_input_event(event)
            logger.error("提示词输入保存失败 (%s)", type(exc).__name__)
            return "提示词保存失败，仍在等待输入。请重新发送，或联系管理员检查日志。"

    @filter.custom_filter(PendingPromptFilter, priority=10000)
    async def receive_prompt(self, event: AstrMessageEvent):
        """Receive the next whole message before ordinary chat handlers run."""
        if event.get_extra(COMMAND_HANDLED_KEY, False):
            event.stop_event()
            return
        reply = await self._pending_reply(event)
        if reply is not None:
            yield event.plain_result(reply).stop_event()

    @filter.command("prompt", priority=10000)
    async def prompt_command(self, event: AstrMessageEvent):
        """发送 /prompt 后，在下一条消息输入提示词；show 查看，clear 清除并退出。"""
        event.should_call_llm(False)
        if event.get_extra(COMMAND_HANDLED_KEY, False):
            event.stop_event()
            return
        event.set_extra(COMMAND_HANDLED_KEY, True)
        reply = await self._command_reply(event)
        yield event.plain_result(reply).stop_event()
        event.stop_event()

    @filter.on_llm_request(priority=100)
    async def apply_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        """Append this sender's prompt without changing the shared persona or history."""
        # Some message/command routes can let a setting message reach this hook.
        # Stop before awaiting storage or sending a reply, so errors cannot turn
        # the setting into a model request. The per-event marker avoids repeats.
        if event.get_extra(COMMAND_HANDLED_KEY, False) or self._is_prompt_command(event):
            event.should_call_llm(False)
            event.stop_event()
            if not event.get_extra(COMMAND_HANDLED_KEY, False):
                event.set_extra(COMMAND_HANDLED_KEY, True)
                reply = await self._command_reply(event)
                await event.send(event.plain_result(reply))
            return

        get_config = getattr(self.context, "get_config", None)
        cfg = get_config(umo=event.unified_msg_origin) if callable(get_config) else {}
        if PendingPromptFilter.matches(event, cfg):
            reply = await self._pending_reply(event)
            if reply is not None:
                await event.send(event.plain_result(reply))
                return

        if self._storage_error or not event.get_sender_id() or self._inactive_reason(event):
            return
        try:
            # Prefer the request's actual conversation: it may have been selected
            # before another concurrent message switched the active conversation.
            if req.conversation is not None:
                cid = req.conversation.cid
            else:
                cid = await self._current_cid(event)
            prompt = self._prompts.get(self._key(event, cid), "") if cid else ""
            if not prompt or self._inactive_reason(event, prompt):
                return
            suffix = f"\n\n[当前用户在本次对话中的自定义提示词]\n{prompt}"
            system_prompt = req.system_prompt or ""
            if not system_prompt.endswith(suffix):
                req.system_prompt = system_prompt + suffix
        except Exception as exc:
            logger.error("个人提示词注入失败，本次使用原提示词 (%s)", type(exc).__name__)

    async def terminate(self) -> None:
        self._pending.clear()
        if PendingPromptFilter.pending is self._pending:
            PendingPromptFilter.pending = {}
