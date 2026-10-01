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

logger = logging.getLogger("MissingInfoAgent")


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class ValidationState(TypedDict):
    extracted_text: str
    validation_results: dict
    error: Optional[str]


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------
class ValidationResult(BaseModel):
    is_complete: bool = Field(
        description="Belge, türü için gerekli tüm standart alanları içeriyorsa True, aksi halde False."
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Eksik alanların listesi; Türkçe, okunabilir ve kısa alan adları olarak "
            "(ör. 'TC Kimlik No', 'IBAN', 'Ad Soyad', 'Öğrenci No', 'Bölüm', 'Tarih', 'İmza'). "
            "is_complete True ise boş liste."
        )
    )
    document_type: str = Field(
        description="Belgenin tahmin edilen türü, Türkçe (ör. 'Dilekçe', 'Fatura', 'Kira Sözleşmesi', 'Başvuru Formu')."
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
DOGRULAMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Sen akıllı bir belge doğrulayıcısın. Görevin, OCR ile çıkarılmış metni okumak, "
        "belgenin türünü tahmin etmek ve o belge türü için standart olarak bulunması gereken "
        "alanlardan hangilerinin eksik olduğunu tespit etmektir.\n"
        "- Sadece gerçekten eksik olan alanları listele; metinde varsa eksik sayma.\n"
        "- OCR kaynaklı küçük yazım hatalarını eksiklik olarak değerlendirme.\n"
        "- Metin bir belgenin sadece bir bölümü gibi görünüyorsa (ör. fotoğrafı çekilmiş tek bir kısım), "
        "yalnızca o bölümde olması gereken alanları değerlendir; belgenin başka bölümlerinde olması "
        "beklenen alanları eksik sayma.\n"
        "- Alan adlarını Türkçe karakterlerle ve okunabilir yaz (ör. 'Öğrenci No', 'Bölüm').\n"
        "- Kesin ve tutarlı ol."
    )),
    ("user", "Çıkarılan Belge Metni:\n<belge>\n{extracted_text}\n</belge>")
])


# ---------------------------------------------------------------------------
# LLM Çağrısı (geçici hatalarda üstel bekleme ile 3 deneme)
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(GECICI_HATALAR),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _validate_with_llm(extracted_text: str) -> ValidationResult:
    """Retry decorator'lı belge doğrulama çağrısı."""
    logger.debug(f"LLM belge doğrulaması başlatılıyor ({len(extracted_text)} karakter).")
    chain = DOGRULAMA_PROMPT | get_llm(max_tokens=1024).with_structured_output(ValidationResult)
    return chain.invoke({"extracted_text": extracted_text})


# ---------------------------------------------------------------------------
# Node Function: Validate Document
# ---------------------------------------------------------------------------
def validate_document(state: ValidationState) -> dict:
    """
    Metinden belge türünü çıkarır ve eksik standart alanları bulur.
    Hata olursa çökmez; state'e error yazar ve boş sonuç döner.
    """
    logger.info("[validate_document] Belge doğrulaması başlatılıyor.")

    extracted_text = state.get("extracted_text", "")
    if not extracted_text.strip():
        logger.warning("[validate_document] Boş metin geldi — doğrulama atlanıyor.")
        return {
            "validation_results": {},
            "error": "Doğrulanacak metin boş."
        }

    try:
        result = _validate_with_llm(extracted_text)

        if hasattr(result, "model_dump"):
            res_dict = result.model_dump()
        elif isinstance(result, dict):
            res_dict = result
        else:
            res_dict = dict(result)

        logger.info(
            f"[validate_document] Doğrulama tamamlandı. "
            f"Tür: {res_dict.get('document_type', 'Bilinmiyor')}, "
            f"Tam: {res_dict.get('is_complete', False)}, "
            f"Eksik alanlar: {res_dict.get('missing_fields', [])}"
        )
        return {"validation_results": res_dict, "error": None}

    except RateLimitError:
        logger.error("[validate_document] Anthropic rate limit aşıldı.")
        return {"validation_results": {}, "error": "Rate limit aşıldı — lütfen daha sonra tekrar deneyin."}
    except AuthenticationError:
        logger.error("[validate_document] Anthropic kimlik doğrulama hatası.")
        return {"validation_results": {}, "error": "API kimlik doğrulama hatası."}
    except APITimeoutError:
        logger.error("[validate_document] Anthropic API zaman aşımı.")
        return {"validation_results": {}, "error": "API zaman aşımı — lütfen tekrar deneyin."}
    except Exception as e:
        logger.error(f"[validate_document] Beklenmeyen hata: {e}", exc_info=True)
        return {"validation_results": {}, "error": f"Doğrulama hatası: {str(e)}"}


# ---------------------------------------------------------------------------
# Build the StateGraph:  START -> validate_document -> END
# ---------------------------------------------------------------------------
workflow = StateGraph(ValidationState)
workflow.add_node("validate_document", validate_document)
workflow.add_edge(START, "validate_document")
workflow.add_edge("validate_document", END)

validator_app = workflow.compile()