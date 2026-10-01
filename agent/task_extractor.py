import logging
from typing import List, Optional

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.prompts import ChatPromptTemplate

from agent.llm_config import get_llm, GECICI_HATALAR, AuthenticationError

logger = logging.getLogger("TaskExtractor")


# --- ÇIKTI FORMATI (Structured Output) ---
class Gorev(BaseModel):
    baslik: str = Field(description="Görevin kısa başlığı")
    sorumlu: Optional[str] = Field(default=None, description="Sorumlu kişi veya kurum, yoksa null")
    son_tarih: Optional[str] = Field(default=None, description="Son tarih, yoksa null")


class CikartmaSonucu(BaseModel):
    gorevler: List[Gorev] = Field(default_factory=list, description="Metinden çıkarılan görevler listesi")
    onemli_tarihler: List[str] = Field(default_factory=list, description="Metinde geçen önemli tarihler")
    konular: List[str] = Field(default_factory=list, description="Metnin ana konuları, ilgi alanları")


# --- PROMPT ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen bir belge analiz uzmanısın.
Sana verilen metinden şunları çıkar:
- Yapılması gereken görevler ve sorumlular
- Önemli tarihler (başvuru, teslim, toplantı vb.)
- Metnin ana konuları (örn: KOSGEB, ihale, kira, proje)

Metinde olmayan bilgiyi uydurma; yoksa null bırak."""),
    ("human", "Aşağıdaki metni analiz et:\n\n{metin}")
])

# --- CHAIN (ilk kullanımda kurulur; API key yoksa import sırasında çökmez) ---
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        structured_llm = get_llm().with_structured_output(CikartmaSonucu)
        _chain = prompt | structured_llm
    return _chain


# --- RETRY DECORATOR ---
@retry(
    retry=retry_if_exception_type(GECICI_HATALAR),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _llm_cagir(metin: str) -> CikartmaSonucu:
    """Retry decorator'lı LLM çağrısı."""
    logger.debug(f"LLM çağrısı yapılıyor (metin uzunluğu: {len(metin)} karakter).")
    return _get_chain().invoke({"metin": metin})


# --- ANA FONKSİYON ---
def metinden_cikar(metin: str) -> CikartmaSonucu:
    """OCR'dan gelen metni alır, görev ve tarihleri çıkarır."""
    logger.info(f"Metin analizi başlatılıyor ({len(metin)} karakter).")
    bos = CikartmaSonucu()
    try:
        sonuc = _llm_cagir(metin)
        if sonuc is None:
            return bos
        logger.info(f"Analiz tamamlandı: {len(sonuc.gorevler)} görev, {len(sonuc.onemli_tarihler)} tarih bulundu.")
        return sonuc
    except AuthenticationError:
        logger.error("Anthropic kimlik doğrulama hatası — ANTHROPIC_API_KEY'i kontrol edin.")
        return bos
    except GECICI_HATALAR as e:
        logger.error(f"Anthropic API geçici hata (3 deneme sonrası): {type(e).__name__} — boş sonuç dönülüyor.")
        return bos
    except Exception as e:
        logger.error(f"Görev çıkarımında beklenmeyen hata: {e}", exc_info=True)
        return bos


# --- CLASS YAPISI ---
class TaskExtractor:
    def extract(self, text: str) -> list:
        logger.info("TaskExtractor.extract() çağrıldı.")
        try:
            cikarilan_veri = metinden_cikar(text)
            findings = []

            for gorev in cikarilan_veri.gorevler:
                findings.append({
                    "title": gorev.baslik,
                    "date": gorev.son_tarih if gorev.son_tarih else "Belirtilmemiş"
                })

            if not findings and cikarilan_veri.onemli_tarihler:
                for tarih in cikarilan_veri.onemli_tarihler:
                    findings.append({
                        "title": "İlgili Etkinlik/Tarih",
                        "date": tarih
                    })

            logger.info(f"TaskExtractor: {len(findings)} bulgu çıkarıldı.")
            return findings

        except Exception as e:
            logger.error(f"TaskExtractor.extract() hatası: {e}", exc_info=True)
            return []


# --- TEST ---
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')

    test_metni = """
    KOSGEB Girişimcilik Destek Programı Başvuru Duyurusu

    Başvuru tarihi: 15 Mart 2025
    Son başvuru tarihi: 30 Nisan 2025

    Başvuru sahiplerinin aşağıdaki belgeleri teslim etmesi gerekmektedir:
    - İş planı hazırlanacak (Sorumlu: Başvuru sahibi)
    - Nüfus cüzdanı fotokopisi eklenecek
    - Mali tablolar muhasebeci onaylı olacak (Sorumlu: Mali müşavir)

    Değerlendirme toplantısı: 15 Mayıs 2025
    Sonuç açıklaması: 1 Haziran 2025
    """

    print("🔍 Metin analiz ediliyor...\n")
    sonuc = metinden_cikar(test_metni)

    print("📋 GÖREVLER:")
    for g in sonuc.gorevler:
        print(f"  - {g.baslik} | Sorumlu: {g.sorumlu} | Tarih: {g.son_tarih}")

    print("\n📅 ÖNEMLİ TARİHLER:")
    for t in sonuc.onemli_tarihler:
        print(f"  - {t}")

    print("\n🏷️ KONULAR:")
    for k in sonuc.konular:
        print(f"  - {k}")