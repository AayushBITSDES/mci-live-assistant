"""Tool schemas for voice-command handling.

Same six tools across both Grok and Gemini. Provider clients translate
this canonical list into their native tool format. The names map 1:1 to
ContextManager state mutations — the dispatcher in `app/main.py`
(coming in a later phase) wires them up.
"""
from __future__ import annotations

from typing import Any

# Snooze duration when dismissTemporarily is called (Design Decision: 5 min)
SNOOZE_DURATION_SECONDS = 5 * 60

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "markDone",
        "description": (
            "Mark the current task or risk as completed. The user has confirmed "
            "they finished it (e.g., 'I already took my pills', 'tea is done'). "
            "Closes any active risk window related to this task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Short label for what was completed (e.g. 'medication', 'tea').",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "dismissTemporarily",
        "description": (
            "Snooze the current nudge for 5 minutes. The user said 'remind me later' "
            "or similar. The same nudge will fire again if conditions persist."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "flagWrong",
        "description": (
            "User indicated this nudge was incorrect or unwelcome. Append the full "
            "context to the feedback log so the system can learn. Does NOT silence "
            "future nudges of the same type — only logs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "If the user gave a reason, capture it verbatim.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "closeForever",
        "description": (
            "User explicitly asked to never be reminded about this again. "
            "Permanent suppression for this nudge category. Use sparingly — only "
            "when the user is unambiguous (e.g., 'never tell me about tea again')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category being suppressed (e.g. 'tea_reminder').",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "toggleMic",
        "description": (
            "Toggle the microphone kill switch. User said 'mic off' / 'mic on'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["on", "off"]},
            },
            "required": ["state"],
        },
    },
    {
        "name": "toggleCamera",
        "description": (
            "Toggle the camera kill switch. User said 'camera off' / 'camera on'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["on", "off"]},
            },
            "required": ["state"],
        },
    },
    {
        "name": "assistantReply",
        "description": (
            "Reply naturally to a short user question or comment that is not a command. "
            "Use for questions like 'who is that?', 'what was I doing?', or 'why are you reminding me?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sentence": {
                    "type": "string",
                    "description": "One short sentence to show and speak back to the user.",
                },
            },
            "required": ["sentence"],
        },
    },
]


def to_openai_format() -> list[dict[str, Any]]:
    """Translate canonical tools to OpenAI/xAI `tools` parameter format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOL_DEFINITIONS
    ]


def tool_names() -> list[str]:
    return [t["name"] for t in TOOL_DEFINITIONS]
