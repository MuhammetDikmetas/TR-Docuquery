import logging
import re
import threading
import time
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Kendi modüllerimizden içe aktarma
from agent.chat_analyzer import ilgi_alanlarini_cikar
from agent.web_searcher import ilgi_alanlarini_arastir

logger = logging.getLogger("ProactiveGraph")

# ---------------------------------------------------------------------------
# Cooldown Mekanizması: Aynı konu 10 dakika içinde aranmışsa tekrar arama
# ---------------------------------------------------------------------------
_arama_gecmisi: Dict[str, float] = {}   # {"konu": timestamp_epoch}
_kilit = threading.Lock()               # API aynı anda birden çok istek işleyebilir
COOLDOWN_SURESI_SANIYE = 600            # 10 dakika

# LLM'in "ilgi alanı yok" anlamında döndürebileceği ifadeler
_GECERSIZ_IFADELER = ("belirlenemedi", "bulunamadı", "bulunamadi")

# Konu benzerliği için anlamsız kelimeler
_DOLGU_KELIMELER = {"ve", "ile", "icin", "bir", "bu", "hakkinda", "nasil", "nedir", "en", "da", "de"}
BENZERLIK_ESIGI = 0.75


def _kokler(metin: str) -> set:
    """
    Konuyu karşılaştırılabilir köklere indirger:
    Türkçe karakterleri sadeleştirir, dolgu kelimeleri atar, her kelimenin ilk 4 harfini alır.
    Böylece 'hibeleri' / 'hibe', 'paneli' / 'panel' aynı kök sayılır.
    """
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    metin = metin.translate(str.maketrans("çğıöşü", "cgiosu"))
    return {k[:4] for k in re.findall(r"\w+", metin) if k not in _DOLGU_KELIMELER}


def _ayni_konu_mu(a: str, b: str) -> bool:
    """
    LLM aynı konuyu her seferinde biraz farklı yazabilir
    ('Güneş enerjisi paneli hibeleri' / 'Güneş Enerjisi Panel Hibe Programları').
    Kök örtüşmesi eşiği geçerse aynı konu kabul edilir.
    """
    ka, kb = _kokler(a), _kokler(b)
    if not ka or not kb:
        return False
    return len(ka & kb) / min(len(ka), len(kb)) >= BENZERLIK_ESIGI


def _gecerli_alanlar(ilgi_alanlari: List[str]) -> List[str]:
    """Boş / 'belirlenemedi' / 'yok' gibi geçersiz ilgi alanlarını ayıklar."""
    sonuc = []
    for alan in ilgi_alanlari or []:
        alan_str = str(alan).strip()
        alan_lower = alan_str.lower()
        if not alan_str or alan_lower == "yok":
            continue
        if any(ifade in alan_lower for ifade in _GECERSIZ_IFADELER):
            continue
        sonuc.append(alan_str)
    return sonuc


def _cooldown_kontrol(ilgi_alanlari: List[str]) -> List[str]:
    """
    İlgi alanlarını cooldown filtresinden geçirir.
    Son 10 dakika içinde aranan konuları listeden çıkarır.

    Returns:
        Aranabilir (cooldown dışı) ilgi alanları listesi
    """
    simdi = time.time()
    aranabilir = []
    with _kilit:
        # Süresi dolmuş kayıtları temizle (sözlük sonsuza kadar büyümesin)
        for eski in [k for k, t in _arama_gecmisi.items() if simdi - t >= COOLDOWN_SURESI_SANIYE]:
            del _arama_gecmisi[eski]

        for alan in ilgi_alanlari:
            # Birebir aynı değil, BENZER bir konu yakın zamanda arandıysa da atla
            son_arama = next(
                (t for k, t in _arama_gecmisi.items() if _ayni_konu_mu(alan, k)),
                None
            )
            if son_arama is not None:
                gecen_sure = simdi - son_arama
                kalan = int(COOLDOWN_SURESI_SANIYE - gecen_sure)
                logger.info(f"[Cooldown] '{alan}' son {int(gecen_sure)}s önce arandı. Kalan bekleme: {kalan}s — atlanıyor.")
            else:
                aranabilir.append(alan)
    return aranabilir


def _cooldown_guncelle(ilgi_alanlari: List[str]) -> None:
    """Aranan konuların zaman damgasını günceller."""
    simdi = time.time()
    with _kilit:
        for alan in ilgi_alanlari:
            _arama_gecmisi[alan.strip().lower()] = simdi
    logger.debug(f"Cooldown zaman damgası güncellendi: {ilgi_alanlari}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ProactiveState(TypedDict):
    sohbet_gecmisi: List[str]
    ilgi_alanlari: List[str]
    arama_sonuclari: Dict[str, Any]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node: analiz
# ---------------------------------------------------------------------------
def analiz_node(state: ProactiveState) -> ProactiveState:
    sohbet_gecmisi = state.get("sohbet_gecmisi", [])
    logger.info(f"[analiz_node] Sohbet geçmişi analiz ediliyor ({len(sohbet_gecmisi)} mesaj).")

    try:
        sonuc_objesi = ilgi_alanlarini_cikar(sohbet_gecmisi)
        ham_alanlar = getattr(sonuc_objesi, 'ilgi_alanlari', []) or []
        # Geçersiz ifadeler state'e hiç yazılmaz -> arama düğümü de onları görmez
        ilgi_alanlari = _gecerli_alanlar(ham_alanlar)
        logger.info(f"[analiz_node] Bulunan ilgi alanları: {ilgi_alanlari}")
        return {"ilgi_alanlari": ilgi_alanlari, "error": None}
    except Exception as e:
        logger.error(f"[analiz_node] İlgi alanı çıkarımı başarısız: {e}", exc_info=True)
        return {"ilgi_alanlari": [], "error": f"Analiz hatası: {str(e)}"}


# ---------------------------------------------------------------------------
# Conditional Edge: arama_gerekli_mi
# ---------------------------------------------------------------------------
def arama_gerekli_mi(state: ProactiveState) -> str:
    # Önceki bir hata varsa direkt bitir
    if state.get("error"):
        logger.warning(f"[Routing] Önceki node'da hata var, arama yapılmıyor: {state['error']}")
        return "bitir"

    gecerli_alanlar = _gecerli_alanlar(state.get("ilgi_alanlari", []))
    if not gecerli_alanlar:
        logger.info("[Routing] Geçerli ilgi alanı bulunamadı — arama yapılmıyor.")
        return "bitir"

    # Cooldown kontrolü
    aranabilir = _cooldown_kontrol(gecerli_alanlar)
    if not aranabilir:
        logger.info("[Routing] Tüm konular cooldown süresi içinde — arama yapılmıyor.")
        return "bitir"

    logger.info(f"[Routing] {len(aranabilir)} konu için arama yapılacak.")
    return "arama_yap"


# ---------------------------------------------------------------------------
# Node: arama
# ---------------------------------------------------------------------------
def arama_node(state: ProactiveState) -> ProactiveState:
    ilgi_alanlari = _gecerli_alanlar(state.get("ilgi_alanlari", []))
    logger.info(f"[arama_node] Web araması başlatılıyor: {ilgi_alanlari}")

    # Cooldown filtresinden geç (routing'de yapıldı ama node içinde de uygula)
    aranabilir = _cooldown_kontrol(ilgi_alanlari)
    if not aranabilir:
        logger.info("[arama_node] Cooldown nedeniyle aranacak konu yok.")
        return {"arama_sonuclari": {}, "error": None}

    try:
        sonuclar = ilgi_alanlarini_arastir(aranabilir)

        if not sonuclar:
            logger.warning("[arama_node] DuckDuckGo arama sonuç döndürmedi.")
            return {"arama_sonuclari": {}, "error": None}

        # Başarılı arama — cooldown damgasını güncelle
        _cooldown_guncelle(aranabilir)
        logger.info(f"[arama_node] Arama tamamlandı. {len(sonuclar)} konu için sonuç bulundu.")
        return {"arama_sonuclari": sonuclar, "error": None}

    except Exception as e:
        logger.error(f"[arama_node] DuckDuckGo araması başarısız: {e}", exc_info=True)
        # Hata durumunda graceful fallback — boş sonuç dön, crash etme
        return {
            "arama_sonuclari": {},
            "error": f"Web araması başarısız: {str(e)}"
        }


# ---------------------------------------------------------------------------
# Node: bildirim
# ---------------------------------------------------------------------------
def bildirim_node(state: ProactiveState) -> ProactiveState:
    arama_sonuclari = state.get("arama_sonuclari", {})
    error = state.get("error")

    if error:
        logger.warning(f"[bildirim_node] Hata durumu tespit edildi: {error}")
        return state

    if arama_sonuclari:
        logger.info("[bildirim_node] Proaktif sonuçlar hazır — bildirim oluşturuluyor.")
        print("\n🤖 PROAKTİF ASİSTAN BİLDİRİMİ 🤖")
        print("Sohbet geçmişinden odaklandığın konuları fark ettim ve senin için şu güncel bilgileri topladım:")
        for konu, ozet in arama_sonuclari.items():
            print(f"🎯 [{konu}]: {ozet}")
            print("-" * 50)
    else:
        logger.info("[bildirim_node] Bildirim için sonuç yok.")

    return state


# ---------------------------------------------------------------------------
# Graf İnşası
#   START -> analiz --(ilgi alanı var + cooldown dışı)--> arama -> bildirim -> END
#                   \--(yok / hata / cooldown)---------------------------------> END
# ---------------------------------------------------------------------------
builder = StateGraph(ProactiveState)

builder.add_node("analiz", analiz_node)
builder.add_node("arama", arama_node)
builder.add_node("bildirim", bildirim_node)

builder.add_edge(START, "analiz")

builder.add_conditional_edges(
    "analiz",
    arama_gerekli_mi,
    {
        "arama_yap": "arama",
        "bitir": END
    }
)

builder.add_edge("arama", "bildirim")
builder.add_edge("bildirim", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# Test Bloğu
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')
    load_dotenv(override=True)

    senaryo_1 = {"sohbet_gecmisi": ["Yarın KPSS sınavım var, bana dua et."], "ilgi_alanlari": [], "arama_sonuclari": {}, "error": None}
    senaryo_2 = {"sohbet_gecmisi": ["Merhaba, nasılsın?"], "ilgi_alanlari": [], "arama_sonuclari": {}, "error": None}

    print("=" * 60)
    print("SENARYO 1 (Dolu): Sistem analiz edip arama düğümüne gitmeli")
    try:
        sonuc_1 = graph.invoke(senaryo_1)
        print("ÇIKTI DURUMU (State):", sonuc_1)
    except Exception as e:
        print(f"Senaryo 1 hatası: {e}")

    print("=" * 60)
    print("\nSENARYO 2 (Boş): Sistem ilgi alanı bulamadığı için arama yapmadan bitirmeli")
    try:
        sonuc_2 = graph.invoke(senaryo_2)
        print("ÇIKTI DURUMU (State):", sonuc_2)
    except Exception as e:
        print(f"Senaryo 2 hatası: {e}")

    print("=" * 60)
    print("\nSENARYO 3 (Cooldown): Aynı konu 10 dakika içinde tekrar aranmamalı")
    try:
        sonuc_3 = graph.invoke(senaryo_1)
        print("ÇIKTI DURUMU (State):", sonuc_3)
    except Exception as e:
        print(f"Senaryo 3 hatası: {e}")
    print("=" * 60)