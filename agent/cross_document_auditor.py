import logging
from typing import TypedDict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.llm_config import (
    get_llm, GECICI_HATALAR,
    RateLimitError, AuthenticationError, APITimeoutError,
)

logger = logging.getLogger("CrossDocumentAuditor")


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class AuditorState(TypedDict):
    doc1_text: str
    doc2_text: str
    comparison_results: dict
    executive_summary: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------
class ComparisonResult(BaseModel):
    changed_clauses: List[str] = Field(
        default_factory=list,
        description="Belge 1 ile Belge 2 arasında değiştirilmiş maddeler (Türkçe)."
    )
    discrepancies: List[str] = Field(
        default_factory=list,
        description="Fiyat, oran veya tarihlerdeki tutarsızlıklar (Türkçe)."
    )
    missing_sections: List[str] = Field(
        default_factory=list,
        description="Belge 1'de olup Belge 2'de bulunmayan bölümler (Türkçe)."
    )
    risk_score: int = Field(
        ge=1, le=10,
        description="Farklılıkların risk seviyesini gösteren 1-10 arası skor."
    )


def _metin_icerik(mesaj) -> str:
    """LangChain mesajının içeriğini düz metne çevirir (str veya blok listesi olabilir)."""
    icerik = getattr(mesaj, "content", mesaj)
    if isinstance(icerik, str):
        return icerik
    if isinstance(icerik, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in icerik
        )
    return str(icerik)


# ---------------------------------------------------------------------------
# Prompt'lar
# ---------------------------------------------------------------------------
KARSILASTIRMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Sen uzman bir analitik denetçisin. Görevin, bir belgenin iki versiyonunu sıkı bir şekilde karşılaştırmak; "
        "değişen maddeleri, tutarsızlıkları (fiyatlar, oranlar veya tarihler gibi) ve eksik bölümleri tespit etmektir. "
        "Farklılıkların ciddiyetine bağlı olarak 1 ile 10 arasında bir risk skoru belirle. "
        "Sadece belgelerde gerçekten yazan bilgilere dayan, tahmin yürütme. "
        "Tüm analizlerini, bulgularını ve açıklamalarını kesinlikle Türkçe olarak yapmalısın."
    )),
    ("user", "Lütfen aşağıdaki belgeleri karşılaştır.\n\n<belge_1>\n{doc1_text}\n</belge_1>\n\n<belge_2>\n{doc2_text}\n</belge_2>")
])

OZET_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Sen üst düzey bir yönetici asistanısın. Görevin, belge karşılaştırma sonuçlarına dayanarak "
        "üst yönetim için 2-3 cümlelik öz ve net bir yönetici özeti yazmaktır. "
        "Risk skorunu ve önemli tutarsızlıkları/değişiklikleri net bir şekilde belirttiğinden emin ol. "
        "Bu özeti kesinlikle Türkçe olarak yazmalısın."
    )),
    ("user", "Karşılaştırma Sonuçları:\n{comparison_results}")
])


# ---------------------------------------------------------------------------
# LLM Çağrıları (geçici hatalarda üstel bekleme ile 3 deneme)
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(GECICI_HATALAR),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _compare_with_llm(doc1_text: str, doc2_text: str) -> ComparisonResult:
    """Retry decorator'lı doküman karşılaştırma çağrısı."""
    logger.debug("LLM doküman karşılaştırması başlatılıyor.")
    chain = KARSILASTIRMA_PROMPT | get_llm().with_structured_output(ComparisonResult)
    return chain.invoke({"doc1_text": doc1_text, "doc2_text": doc2_text})


@retry(
    retry=retry_if_exception_type(GECICI_HATALAR),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _summarize_with_llm(comparison_results: dict) -> str:
    """Retry decorator'lı özet üretme çağrısı."""
    logger.debug("LLM özet üretimi başlatılıyor.")
    chain = OZET_PROMPT | get_llm(max_tokens=512)
    return _metin_icerik(chain.invoke({"comparison_results": comparison_results}))


# ---------------------------------------------------------------------------
# Node Function 1: Compare Documents
# ---------------------------------------------------------------------------
def compare_documents(state: AuditorState) -> dict:
    """
    İki belge metnini Claude + structured output ile karşılaştırır.
    Hata olursa çökmez; state'e error yazar ve boş sonuç döner.
    """
    logger.info("[compare_documents] Doküman karşılaştırması başlatılıyor.")
    try:
        result = _compare_with_llm(state["doc1_text"], state["doc2_text"])

        if hasattr(result, "model_dump"):
            comp_res = result.model_dump()
        elif isinstance(result, dict):
            comp_res = result
        else:
            comp_res = dict(result)

        logger.info(f"[compare_documents] Karşılaştırma tamamlandı. Risk skoru: {comp_res.get('risk_score', 'N/A')}")
        return {"comparison_results": comp_res, "error": None}

    except RateLimitError:
        logger.error("[compare_documents] Anthropic rate limit aşıldı.")
        return {"comparison_results": {}, "error": "Rate limit aşıldı — lütfen daha sonra tekrar deneyin."}
    except AuthenticationError:
        logger.error("[compare_documents] Anthropic kimlik doğrulama hatası.")
        return {"comparison_results": {}, "error": "API kimlik doğrulama hatası."}
    except APITimeoutError:
        logger.error("[compare_documents] Anthropic API zaman aşımı.")
        return {"comparison_results": {}, "error": "API zaman aşımı — lütfen tekrar deneyin."}
    except Exception as e:
        logger.error(f"[compare_documents] Beklenmeyen hata: {e}", exc_info=True)
        return {"comparison_results": {}, "error": f"Karşılaştırma hatası: {str(e)}"}


# ---------------------------------------------------------------------------
# Node Function 2: Generate Summary
# ---------------------------------------------------------------------------
def generate_summary(state: AuditorState) -> dict:
    """
    Karşılaştırma sonuçlarını okuyup yönetici özeti üretir.
    Önceki adımda hata varsa veya sonuç boşsa erken döner.
    """
    logger.info("[generate_summary] Özet üretimi başlatılıyor.")

    if state.get("error"):
        logger.warning(f"[generate_summary] Önceki hata nedeniyle özet atlanıyor: {state['error']}")
        return {"executive_summary": f"Özet üretilemedi: {state['error']}"}

    if not state.get("comparison_results"):
        logger.warning("[generate_summary] Karşılaştırma sonucu boş — özet atlanıyor.")
        return {"executive_summary": "Karşılaştırma sonucu bulunamadı, özet üretilemedi."}

    try:
        summary = _summarize_with_llm(state["comparison_results"])
        logger.info("[generate_summary] Özet başarıyla üretildi.")
        return {"executive_summary": summary}

    except RateLimitError:
        logger.error("[generate_summary] Anthropic rate limit aşıldı.")
        return {"executive_summary": "Özet üretilemedi: Rate limit aşıldı."}
    except AuthenticationError:
        logger.error("[generate_summary] Anthropic kimlik doğrulama hatası.")
        return {"executive_summary": "Özet üretilemedi: API kimlik hatası."}
    except APITimeoutError:
        logger.error("[generate_summary] Anthropic zaman aşımı.")
        return {"executive_summary": "Özet üretilemedi: Zaman aşımı."}
    except Exception as e:
        logger.error(f"[generate_summary] Beklenmeyen hata: {e}", exc_info=True)
        return {"executive_summary": f"Özet üretilemedi: {str(e)}"}


# ---------------------------------------------------------------------------
# Build the StateGraph:  START -> compare_documents -> generate_summary -> END
# ---------------------------------------------------------------------------
workflow = StateGraph(AuditorState)

workflow.add_node("compare_documents", compare_documents)
workflow.add_node("generate_summary", generate_summary)

workflow.add_edge(START, "compare_documents")
workflow.add_edge("compare_documents", "generate_summary")
workflow.add_edge("generate_summary", END)

auditor_app = workflow.compile()