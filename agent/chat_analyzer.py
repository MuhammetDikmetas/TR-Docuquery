import logging
from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent.llm_config import get_llm, GECICI_HATALAR, AuthenticationError

logger = logging.getLogger("ChatAnalyzer")


class KullaniciProfili(BaseModel):
    ilgi_alanlari: List[str] = Field(
        default_factory=list,
        description=(
            "Kullanıcının güncel ilgi alanları, hedefleri veya uğraştığı sorunlar. "
            "Her biri web'de aranabilecek kısa bir ifade. Belirgin bir konu yoksa boş liste."
        )
    )


SISTEM_PROMPT = (
    "Sen bir profil analiz uzmanısın. Sana verilen kullanıcı sohbet geçmişini incele "
    "ve kullanıcının güncel ilgi alanlarını, hazırlandığı sınavları, kariyer hedeflerini "
    "veya günlük hayatta çözmeye çalıştığı sorunları listele.\n"
    "- Her maddeyi web'de aranabilecek kısa bir ifade olarak yaz (ör. 'KPSS 2026 sınav tarihi').\n"
    "- En fazla 5 madde döndür, en önemliden başla.\n"
    "- Selamlaşma, sohbet veya belirgin bir konu yoksa boş liste döndür; madde uydurma."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SISTEM_PROMPT),
    ("human", "Aşağıdaki sohbet geçmişini analiz et:\n{gecmis_metni}")
])

_chain = None


def _get_chain():
    """Chain ilk kullanımda bir kez kurulur (her istekte yeniden oluşturulmaz)."""
    global _chain
    if _chain is None:
        _chain = PROMPT | get_llm(max_tokens=512).with_structured_output(KullaniciProfili)
    return _chain


@retry(
    retry=retry_if_exception_type(GECICI_HATALAR),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _llm_cagir(gecmis_metni: str) -> KullaniciProfili:
    """Retry decorator'lı LLM çağrısı."""
    logger.debug("LLM çağrısı yapılıyor (ilgi alanı çıkarımı).")
    return _get_chain().invoke({"gecmis_metni": gecmis_metni})


def ilgi_alanlarini_cikar(sohbet_gecmisi: List[str]) -> KullaniciProfili:
    """
    Sohbet geçmişini analiz ederek kullanıcının ilgi alanlarını çıkarır.
    Hata durumunda boş profil döner (crash etmez).
    """
    logger.info(f"İlgi alanı çıkarımı başlatılıyor ({len(sohbet_gecmisi)} mesaj).")

    if not sohbet_gecmisi:
        logger.warning("Boş sohbet geçmişi geldi — boş profil dönülüyor.")
        return KullaniciProfili()

    gecmis_metni = "\n".join(f"- {mesaj}" for mesaj in sohbet_gecmisi)

    try:
        sonuc = _llm_cagir(gecmis_metni) or KullaniciProfili()
        logger.info(f"İlgi alanları başarıyla çıkarıldı: {sonuc.ilgi_alanlari}")
        return sonuc
    except AuthenticationError:
        logger.error("Anthropic kimlik doğrulama hatası — ANTHROPIC_API_KEY'i kontrol edin.")
        return KullaniciProfili()
    except GECICI_HATALAR as e:
        logger.error(f"Anthropic API geçici hata (3 deneme sonrası): {type(e).__name__} — boş profil dönülüyor.")
        return KullaniciProfili()
    except Exception as e:
        logger.error(f"İlgi alanı çıkarımında beklenmeyen hata: {e}", exc_info=True)
        return KullaniciProfili()


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')
    load_dotenv(find_dotenv(), override=True)

    test_gecmisi = [
        'İngilizce B2 seviyesine gelmek için çalışıyorum, bir yandan da Vodafone ve Garanti BBVA sınavlarına hazırlanıyorum.',
        'Son zamanlarda sağlıklı yulaf tarifleri deniyorum. Bir de çelik termosumdaki kahve lekelerini nasıl temizlerim?'
    ]

    profil = ilgi_alanlarini_cikar(test_gecmisi)

    print("\n💡 Tespit Edilen İlgi Alanları ve Hedefler:")
    print("-" * 45)
    for ilgi_alani in profil.ilgi_alanlari:
        print(f"• {ilgi_alani}")
    print("-" * 45 + "\n")