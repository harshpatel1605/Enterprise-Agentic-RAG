import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from groq import Groq

from app.config import settings


# Determine config: If PORTKEY_CONFIG_ID is set (e.g. pc-...), use it.
# Otherwise, construct the dynamic gateway config with current active models.
if settings.PORTKEY_CONFIG_ID:
    GATEWAY_CONFIG = settings.PORTKEY_CONFIG_ID
else:
    GATEWAY_CONFIG = {
        "strategy": {"mode": "fallback"},
        "cache": {"mode": "simple"},
        "retry": {
            "attempts": 2,
            "on_status_codes": [429, 503]
        },
        "targets": [
            {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"}},
            {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.GROQ_FALLBACK_MODEL}"}},
        ]
    }


class ResilientPortkeyCompletions:
    def __init__(self, raw_portkey=None, direct_groq=None):
        self._raw_portkey = raw_portkey
        self._direct_groq = direct_groq

    def create(self, *args, **kwargs):
        # 1. Try Portkey if available and configured
        if self._raw_portkey and settings.PORTKEY_API_KEY:
            try:
                call_kwargs = dict(kwargs)
                if "model" not in call_kwargs:
                    call_kwargs["model"] = f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"
                return self._raw_portkey.chat.completions.create(*args, **call_kwargs)
            except Exception as e:
                logfire.warning(f"⚠️ Portkey gateway request failed ({e}). Falling back directly to Groq.")

        # 2. Resilient fallback to direct Groq client
        groq_client = self._direct_groq or Groq(api_key=settings.GROQ_API_KEY)
        call_kwargs = dict(kwargs)
        call_kwargs.pop("model", None)
        return groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            *args,
            **call_kwargs
        )


class ResilientPortkeyClient:
    def __init__(self):
        try:
            if settings.PORTKEY_API_KEY:
                self._pk = Portkey(api_key=settings.PORTKEY_API_KEY, config=GATEWAY_CONFIG)
            else:
                self._pk = None
        except Exception as e:
            logfire.warning(f"Failed to initialize Portkey client: {e}")
            self._pk = None

        try:
            self._groq = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        except Exception:
            self._groq = None

        self.chat = type("Chat", (), {"completions": ResilientPortkeyCompletions(self._pk, self._groq)})()


portkey_client = ResilientPortkeyClient()


def get_langchain_llm(feature: str = "rag"):
    """
    Returns a resilient LLM: attempts Portkey proxy first, but automatically falls back
    to ChatGroq if Portkey is not configured or fails.
    """
    groq_backup = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )

    if not settings.PORTKEY_API_KEY:
        return groq_backup

    try:
        portkey_llm = ChatOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
            temperature=0,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=GATEWAY_CONFIG,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production"
                }
            )
        )
        return portkey_llm.with_fallbacks([groq_backup])
    except Exception as e:
        logfire.warning(f"Could not build Portkey ChatOpenAI ({e}); defaulting to ChatGroq.")
        return groq_backup


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"