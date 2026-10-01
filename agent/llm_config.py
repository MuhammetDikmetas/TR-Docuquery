"""
Projedeki tüm LLM ayarlarının tek merkezi.
Model veya sağlayıcı değişecekse sadece bu dosya / .env değişir.

.env içinde:
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_MODEL=claude-sonnet-5-5      (opsiyonel)
"""
import os
import logging

from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from anthropic import (
    RateLimitError,
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

logger = logging.getLogger("LLMConfig")
load_dotenv(find_dotenv(), override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

if ANTHROPIC_API_KEY:
    logger.info(f"Anthropic API anahtarı yüklendi (Sonu: ...{ANTHROPIC_API_KEY[-4:]}), model: {ANTHROPIC_MODEL}")
else:
    logger.warning("ANTHROPIC_API_KEY bulunamadı! .env dosyanızı kontrol edin.")

# Tekrar denemeye değer (geçici) hatalar
GECICI_HATALAR = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class ClaudeChat(ChatAnthropic):
    """
    ChatAnthropic'in structured output varsayılanını 'json_schema' yapar.
    Yeni Claude modellerinde 'thinking' açıkken zorunlu tool calling desteklenmez
    (langchain uyarısı); json_schema yöntemi her iki durumda da güvenle çalışır.
    Tüm ajanlar get_llm() kullandığı için bu değişiklik tek yerden hepsine uygulanır.
    """

    def with_structured_output(self, schema, *, method: str = "json_schema", **kwargs):
        return super().with_structured_output(schema, method=method, **kwargs)


def get_llm(max_tokens: int = 4096, **kwargs) -> ChatAnthropic:
    """Projede kullanılan Claude modelini döner."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY tanımlı değil. .env dosyasına ekleyin.")
    return ClaudeChat(
        model=ANTHROPIC_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=2,
        **kwargs,
    )