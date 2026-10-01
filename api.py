import os
import sys
from dotenv import load_dotenv

# .env dosyasını en başta yüklüyoruz
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, cast
import logging
import json
import re
import uuid
import datetime
import math
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# --- AGENT ---
from agent.web_searcher import WebSearcher
from agent.task_extractor import TaskExtractor, CikartmaSonucu
from agent.graph_builder import graph_invoke_with_timeout, AgentState
from agent.proactive_graph import graph as proactive_graph, ProactiveState
from agent.cross_document_auditor import auditor_app
from agent.missing_info_agent import validator_app
from agent.llm_config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

# --- ANTHROPIC (CLAUDE) ENTEGRASYONU ---
from anthropic import AsyncAnthropic, RateLimitError, AuthenticationError, APITimeoutError

# --- DİĞER KÜTÜPHANELER ---
import aiofiles
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# --- PROJE MODÜLLERİ (OCR + PDF + Chunking tek kaynaktan) ---
from pdf_reader import PDFProcessor
from process_documents import DocumentProcessor

# --- LLM İSTEMCİSİ ---
if not ANTHROPIC_API_KEY:
    print("⚠️ UYARI: ANTHROPIC_API_KEY bulunamadı! .env dosyanızı kontrol edin.")
    client = None
else:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=60, max_retries=2)

# --- LOGLAMA ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("OCR_API")


async def claude_cevap(system: str, user: str, max_tokens: int = 2048) -> str:
    """Claude'a tek turlu istek atar ve metin cevabını döner."""
    resp = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# --- EMBEDDING ---
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

def metadata_temizle(metadata: dict) -> dict:
    if not metadata: return {}
    temiz = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) and value is not None:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)): continue
            if isinstance(value, str) and not value.strip(): continue
            temiz[key] = value
    return temiz

# --- GLOBAL STATE ---
class AppState:
    def __init__(self):
        self.ocr_reader = None
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.documents_folder = "documents"
        self.db_path = "./chroma_db_store"

state = AppState()

# PDF okuyucu (metin katmanı + taranmış sayfalar için OCR fallback)
pdf_isleyici = PDFProcessor()
# Madde bazlı chunking metodları. Sadece chunking fonksiyonları kullanıldığı için
# __init__ çağrılmaz (embedding modeli / veritabanı ikinci kez yüklenmez).
chunker = object.__new__(DocumentProcessor)

app = FastAPI(
    title="OCR RAG API",
    version="4.1",
    description="OCR + RAG + LangGraph Agent (Claude) — Tek sunucu (port 8000)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global Exception Handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Yakalanmamış hata [{request.method} {request.url}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": str(exc), "code": 500}
    )

## mobilden gelen ping istekleri için endpoint --
@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.on_event("startup")
async def startup_event():
    logger.info("🎬 SİSTEM BAŞLATILIYOR...")
    os.makedirs(state.documents_folder, exist_ok=True)

    try:
        # ocr_engine import edilince EasyOCR bir kez yüklenir; tüm proje aynı okuyucuyu kullanır
        import ocr_engine
        state.ocr_reader = ocr_engine.reader
        logger.info("✅ EasyOCR yüklendi.")
    except Exception as e:
        logger.critical(f"EasyOCR yüklenemedi: {e}")
        sys.exit(1)

    try:
        state.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("✅ Embedding modeli yüklendi.")
    except Exception as e:
        logger.critical(f"Embedding modeli yüklenemedi: {e}")
        sys.exit(1)

    try:
        state.chroma_client = chromadb.PersistentClient(path=state.db_path)
        state.collection = state.chroma_client.get_or_create_collection(
            name="hukuk_dokumanlari",
            embedding_function=MyEmbeddingFunction(state.embedding_model),
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✅ SİSTEM HAZIR. Koleksiyondaki belge sayısı: {state.collection.count()}")
    except Exception as e:
        logger.critical(f"ChromaDB başlatılamadı: {e} — Sunucu kapatılıyor.")
        sys.exit(1)

# ==========================================
# 1. OCR GÖRÜNTÜ İŞLEME MANTIĞI
# ==========================================
def goruntu_isleyerek_oku(image_bytes) -> dict:
    """
    ocr_engine.py'deki mantığı kullanarak görüntüyü okur.
    Returns: {"text": str, "confidence": float, "warning": str|None}
    """
    from ocr_engine import ocr_ile_oku as _ocr_ile_oku
    logger.info("Görüntü OCR işlemi başlatılıyor (goruntu_isleyerek_oku).")
    try:
        result = _ocr_ile_oku(image_bytes)
        logger.info(f"OCR tamamlandı — güven: {result.get('confidence', 0):.2f}, uyarı: {result.get('warning')}")
        return result
    except Exception as e:
        logger.error(f"OCR İşleme Hatası: {e}")
        return {"text": "", "confidence": 0.0, "warning": f"OCR hatası: {str(e)}"}

# ==========================================
# 2. PDF OKUMA + MADDE BAZLI CHUNKING
# ==========================================
def pdf_ocr_yap_advanced(pdf_path: str) -> Dict[int, str]:
    """
    Geriye dönük uyumluluk: {sayfa_no: metin} döner.
    Dijital metin yetersizse pdf_reader otomatik olarak ocr_engine'e düşer.
    """
    logger.info(f"📄 PDF İşleniyor: {pdf_path}")
    return {s["page"]: s["text"] for s in pdf_isleyici.extract_text_with_metadata(pdf_path)}

def chunking_logic(metin: str, sayfa_no: int, dosya_adi: str, baslangic_index: int,
                   okuma_yontemi: Optional[str] = None) -> List[Dict]:
    """
    process_documents.DocumentProcessor ile aynı madde bazlı chunking'i kullanır
    ve sonucu API'nin metadata şemasına (dosya / sayfa / madde_no) çevirir.
    """
    chunks = []
    for c in chunker.madde_bazli_chunk(metin, dosya_adi, sayfa_no, okuma_yontemi):
        m = c["metadata"]
        madde = m.get("section_number")
        chunks.append({
            "metin": c["text"],
            "metadata": {
                "dosya": dosya_adi,
                "sayfa": sayfa_no,
                "madde_no": str(madde) if madde is not None else "genel",
                "madde_basligi": m.get("section_title"),
                "chunk_type": m.get("chunk_type"),
                "okuma_yontemi": okuma_yontemi,
                "chunk_index": baslangic_index + len(chunks),
            }
        })
    return chunks

def worker_process(pdf_path: str, dosya_adi: str):
    logger.info(f"⚙️ İşlemci çalışıyor: {dosya_adi}")
    sayfalar = pdf_isleyici.extract_text_with_metadata(pdf_path)
    tum_chunks = []
    for s in sayfalar:
        if len(s["text"].strip()) < 20:
            continue
        yeni = chunking_logic(s["text"], s["page"], dosya_adi, len(tum_chunks), s.get("method"))
        tum_chunks.extend(yeni)
    return tum_chunks

def get_full_text_by_filename(filename: str) -> str:
    """ChromaDB'den belirli bir dosyanın tüm parçalarını çekip birleştirir."""
    if state.collection is None: return ""
    try:
        results = state.collection.get(where={"dosya": filename})
        if not results or not results.get('documents'):
            return ""

        # Parçaları chunk_index'e göre sırala (metadata içinden)
        docs_with_meta = []
        for i in range(len(results['documents'])):
            docs_with_meta.append({
                "text": results['documents'][i],
                "index": results['metadatas'][i].get('chunk_index', 0) if results['metadatas'] else 0
            })

        docs_with_meta.sort(key=lambda x: x['index'])
        return "\n".join([d['text'] for d in docs_with_meta])
    except Exception as e:
        logger.error(f"Dosya metni çekilirken hata ({filename}): {e}")
        return ""

async def get_missing_info(text: str) -> List[str]:
    """
    LangGraph validator_app kullanarak metindeki eksik alanları bulur.
    """
    try:
        if not text or len(text.strip()) < 10:
            return []

        initial_state = {
            "extracted_text": text,
            "validation_results": {}
        }
        final_state = await validator_app.ainvoke(initial_state)
        results = final_state.get("validation_results", {})
        return results.get("missing_fields", [])
    except Exception as e:
        logger.error(f"Eksik bilgi tespiti hatası: {e}")
        return []

# ==========================================
# 3. ENDPOINTS
# ==========================================

# Belgenin tamamını ilgilendiren (özet / genel bakış) soruları yakalayan ifadeler
OZET_TETIKLEYICILER = [
    "ne hakkında", "ne hakkinda", "özetle", "ozetle", "özeti", "ozeti", "özet çıkar",
    "hangi maddeler", "maddelerden", "tüm maddeler", "tum maddeler",
    "genel olarak", "konusu ne", "neyi kapsıyor", "neyi kapsiyor", "içeriği ne", "icerigi ne",
]
OZET_MAKS_PARCA = 30  # Bundan büyük belgelerde tüm parçalar yerine normal arama yapılır
GENEL_SORU_ESIGI = 0.45  # Belge seçili değilken bu benzerliğin altındaki sorular sohbet olarak cevaplanır

class SoruModel(BaseModel):
    soru: str
    top_k: Optional[int] = 10
    dosya_adi: Optional[str] = None
    dosya_adlari: Optional[List[str]] = None
    session_id: Optional[str] = None

class ValidationCompleteModel(BaseModel):
    dosya_adi: str
    alanlar: Dict[str, str]
    session_id: Optional[str] = None

async def dosya_isle_ve_kaydet(file: UploadFile):
    """Tek bir dosyayı işler, OCR yapar ve ChromaDB'ye kaydeder."""
    # Güvenlik: "../../x.pdf" gibi dosya adlarıyla klasör dışına yazmayı engelle
    dosya_adi = os.path.basename(file.filename or "")
    if not dosya_adi.lower().endswith('.pdf'):
        raise HTTPException(400, f"Sadece PDF kabul edilir: {file.filename}")

    path = os.path.join(state.documents_folder, dosya_adi)
    try:
        if state.collection is not None:
            try:
                old_data = state.collection.get(where={"dosya": dosya_adi})
                if old_data and old_data.get('ids'):
                    state.collection.delete(ids=old_data['ids'])
            except Exception:
                pass

        async with aiofiles.open(path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        chunks = await run_in_threadpool(worker_process, path, dosya_adi)
        logger.info(f"✅ İşleme Tamamlandı: {dosya_adi} -> {len(chunks)} parça oluşturuldu.")

        if chunks and state.collection is not None:
            ids = [f"{dosya_adi}_{c['metadata']['chunk_index']}_{uuid.uuid4().hex[:6]}" for c in chunks]
            docs = [c["metin"] for c in chunks]
            metas = [metadata_temizle(c["metadata"]) for c in chunks]

            for m in metas:
                if 'dosya' not in m: m['dosya'] = dosya_adi

            state.collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"📥 {len(chunks)} parça ChromaDB'ye eklendi.")

        full_text = "\n".join([c["metin"] for c in chunks])
        missing_info = await get_missing_info(full_text)

        return {
            "filename": dosya_adi,
            "chunk_sayisi": len(chunks),
            "missing_info": missing_info
        }
    except Exception as e:
        logger.error(f"Dosya İşleme Hatası ({dosya_adi}): {e}")
        raise e

@app.post("/yukle")
async def yukle(files: List[UploadFile] = File(...)):
    processed_files = []
    total_chunks = 0

    try:
        for file in files:
            res = await dosya_isle_ve_kaydet(file)
            processed_files.append(res)
            total_chunks += res["chunk_sayisi"]

        return {
            "success": True,
            "mesaj": f"{len(processed_files)} dosya yüklendi.",
            "chunk_sayisi": total_chunks,
            "files": processed_files,
            "missing_info": processed_files[0]["missing_info"] if processed_files else []
        }
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/validator/complete")
async def validator_complete(req: ValidationCompleteModel):
    try:
        logger.info(f"✅ Bilgiler güncellendi: {req.dosya_adi} -> {req.alanlar}")
        return {"success": True, "mesaj": "Bilgiler başarıyla güncellendi."}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/sor")
async def soru_sor(req: SoruModel):
    if not client: raise HTTPException(500, "Anthropic API Key yok!")

    soru = req.soru.strip()
    if not soru or len(soru) < 3:
        return {"cevap": "Geçersiz soru.", "kaynaklar": [], "missing_info": []}

    logger.info(f"/sor isteği: '{soru[:80]}...' (top_k={req.top_k})")

    # ── Intent Detection 1: Cross-Document Audit ───────────────────────────────
    AUDIT_TRIGGERS = [
        "kıyasla", "kiyasla", "karşılaştır", "karsilastir",
        "fark nedir", "değişen maddeler", "degisen maddeler",
        "çapraz denetle", "capraz denetle", "iki belge", "belgeler arasında",
        "belgeler arasinda"
    ]
    soru_lower = soru.lower()
    if any(trigger in soru_lower for trigger in AUDIT_TRIGGERS):
        logger.info("🔍 Audit intent algılandı — belge karşılaştırması yönlendiriliyor.")
        try:
            # Hangi dosyalar var? dosya_adlari öncelikli, yoksa ChromaDB'den bul
            audit_filenames: List[str] = []
            if req.dosya_adlari and len(req.dosya_adlari) >= 2:
                audit_filenames = req.dosya_adlari[:2]
            elif state.collection is not None:
                # En son eklenen benzersiz dosya adlarından ilk 2'yi al
                try:
                    all_meta = state.collection.get()
                    seen: list = []
                    for m in (all_meta.get("metadatas") or []):
                        fn = (m or {}).get("dosya")
                        if fn and fn not in seen:
                            seen.append(fn)
                    audit_filenames = seen[:2]
                except Exception as _e:
                    logger.warning(f"ChromaDB dosya listesi alınamadı: {_e}")

            if len(audit_filenames) < 2:
                return {
                    "cevap": "Karşılaştırma için iki belge yükleyin.",
                    "kaynaklar": [],
                    "missing_info": [],
                    "intent": "audit"
                }

            # Dosyaların tam metinlerini çek
            doc1_text = get_full_text_by_filename(audit_filenames[0])
            doc2_text = get_full_text_by_filename(audit_filenames[1])

            if not doc1_text or not doc2_text:
                return {
                    "cevap": "Karşılaştırma için her iki belgenin de içeriği gerekli. Belgeleri yeniden yükleyin.",
                    "kaynaklar": [],
                    "missing_info": [],
                    "intent": "audit"
                }

            audit_initial = {
                "doc1_text": doc1_text,
                "doc2_text": doc2_text,
                "comparison_results": {},
                "executive_summary": "",
                "error": None
            }
            audit_final = await auditor_app.ainvoke(audit_initial)

            if audit_final.get("error"):
                return {
                    "cevap": f"Belge karşılaştırması sırasında hata oluştu: {audit_final['error']}",
                    "kaynaklar": [],
                    "missing_info": [],
                    "intent": "audit"
                }

            comp = audit_final.get("comparison_results", {})
            summary = audit_final.get("executive_summary", "")
            comp["timestamp"] = datetime.datetime.now().isoformat()

            logger.info(f"✅ Audit tamamlandı ({audit_filenames[0]} vs {audit_filenames[1]})")
            return {
                "cevap": summary or "Karşılaştırma tamamlandı.",
                "kaynaklar": [
                    {"source": audit_filenames[0], "page": 0, "type": "audit", "score": 1.0},
                    {"source": audit_filenames[1], "page": 0, "type": "audit", "score": 1.0},
                ],
                "missing_info": [],
                "intent": "audit",
                "audit_result": {
                    "success": True,
                    "comparison_results": comp,
                    "executive_summary": summary,
                    "doc1": audit_filenames[0],
                    "doc2": audit_filenames[1]
                }
            }
        except Exception as _ae:
            logger.error(f"Audit intent işleme hatası: {_ae}", exc_info=True)
            # Hata durumunda RAG'a düş
    # ── Intent Detection 2: Missing Info Validator ─────────────────────────────
    VALIDATOR_TRIGGERS = [
        "eksik bilgi", "eksik alan", "belgeyi doğrula", "belgeyi dogrula",
        "belge tam mı", "belge tam mi", "eksik", "doğrula", "dogrula",
        "neyi eksik", "ne eksik"
    ]
    if any(trigger in soru_lower for trigger in VALIDATOR_TRIGGERS):
        logger.info("🔍 Validator intent algılandı — eksik bilgi kontrolü yönlendiriliyor.")
        try:
            # Hangi belge metni kullanılacak?
            val_text = ""
            val_filenames = req.dosya_adlari or ([req.dosya_adi] if req.dosya_adi else [])
            if val_filenames and state.collection is not None:
                val_text = get_full_text_by_filename(val_filenames[0])
            elif state.collection is not None:
                # Hiç dosya belirtilmediyse koleksiyondaki ilk dosyayı al
                try:
                    all_meta = state.collection.get()
                    seen_val: list = []
                    for m in (all_meta.get("metadatas") or []):
                        fn = (m or {}).get("dosya")
                        if fn and fn not in seen_val:
                            seen_val.append(fn)
                    if seen_val:
                        val_text = get_full_text_by_filename(seen_val[0])
                except Exception as _e:
                    logger.warning(f"ChromaDB dosya listesi alınamadı (validator): {_e}")

            if not val_text.strip():
                return {
                    "cevap": "Doğrulanacak belge bulunamadı. Lütfen önce bir belge yükleyin.",
                    "kaynaklar": [],
                    "missing_info": [],
                    "intent": "validator"
                }

            val_initial = {
                "extracted_text": val_text,
                "validation_results": {},
                "error": None
            }
            val_final = await validator_app.ainvoke(val_initial)

            if val_final.get("error"):
                return {
                    "cevap": f"Belge doğrulama sırasında hata: {val_final['error']}",
                    "kaynaklar": [],
                    "missing_info": [],
                    "intent": "validator"
                }

            vres = val_final.get("validation_results", {})
            missing = vres.get("missing_fields", [])
            is_complete = vres.get("is_complete", False)
            doc_type = vres.get("document_type", "Bilinmiyor")

            if is_complete:
                cevap_val = f"✅ Belge tam görünüyor ({doc_type}). Eksik alan tespit edilmedi."
            else:
                cevap_val = (
                    f"⚠️ Belge türü: {doc_type}. "
                    f"Eksik alanlar: {', '.join(missing) if missing else 'Yok'}."
                )

            logger.info(f"✅ Validator tamamlandı. Eksik: {missing}")
            return {
                "cevap": cevap_val,
                "kaynaklar": [],
                "missing_info": missing,
                "intent": "validator",
                "validation_result": {
                    "success": True,
                    "validation_results": vres
                }
            }
        except Exception as _ve:
            logger.error(f"Validator intent işleme hatası: {_ve}", exc_info=True)
            # Hata durumunda RAG'a düş
    # ── End Intent Detection ────────────────────────────────────────────────────

    try:
        if state.collection is None:
            return {"cevap": "Veritabanı hazır değil.", "kaynaklar": [], "missing_info": []}

        # Vektör Arama (dosya_adi veya dosya_adlari varsa filtrele)
        query_kwargs: Dict[str, Any] = {
            "query_texts": [soru],
            "n_results": req.top_k
        }
        if req.dosya_adlari and len(req.dosya_adlari) > 0:
            logger.info(f"🔍 Çoklu Dosya Filtresi Uygulanıyor: {req.dosya_adlari}")
            query_kwargs["where"] = {"dosya": {"$in": req.dosya_adlari}}
        elif req.dosya_adi:
            logger.info(f"🔍 Sorgu Filtresi Uygulanıyor: {req.dosya_adi}")
            query_kwargs["where"] = {"dosya": req.dosya_adi}
        else:
            logger.info("🌐 Genel Sorgu Yapılıyor (Filtre Yok)")

        # ── Özet / genel bakış sorusu mu? ────────────────────────────────
        # "Belge ne hakkında?", "Hangi maddeler var?" gibi sorular belgenin TAMAMINI
        # ilgilendirir; tek bir parçayı getirmek eksik cevaba yol açar. Seçili belge
        # küçükse (<= OZET_MAKS_PARCA) tüm parçaları bağlama al.
        dosya_secili = bool(req.dosya_adi or req.dosya_adlari)
        ozet_modu = False
        if dosya_secili and any(t in soru_lower for t in OZET_TETIKLEYICILER):
            toplam_parca = len(state.collection.get(where=query_kwargs["where"]).get("ids") or [])
            if 0 < toplam_parca <= OZET_MAKS_PARCA:
                ozet_modu = True
                query_kwargs["n_results"] = toplam_parca
                logger.info(f"📋 Özet sorusu algılandı — belgenin tüm {toplam_parca} parçası kullanılacak.")

        results = state.collection.query(**query_kwargs)
        if not results.get('documents') or not results['documents'][0]:
            return {"cevap": "Bu soruyla ilgili yeterli bilgi bulunamadı.", "kaynaklar": [], "missing_info": []}

        raw_docs: List[str] = list(results['documents'][0])
        raw_metas: List[Dict[str, Any]] = list(results['metadatas'][0])  # type: ignore
        raw_dists: List[float] = list(results['distances'][0])  # type: ignore

        if not raw_dists:
            return {"cevap": "Bu soruyla ilgili yeterli bilgi bulunamadı.", "kaynaklar": [], "missing_info": []}

        en_iyi_skor = 1 - raw_dists[0]
        dinamik_esik = en_iyi_skor * 0.85 if en_iyi_skor > 0.75 else max(en_iyi_skor * 0.70, 0.30)

        # ── Reranking: dinamik eşik kullanan filtreleme ─────────────────
        combined = []
        for i in range(len(raw_docs)):
            score = 1.0 - (raw_dists[i] if i < len(raw_dists) else 1.0)
            if ozet_modu or score >= dinamik_esik:
                combined.append({
                    "text": raw_docs[i],
                    "meta": raw_metas[i] if i < len(raw_metas) else {},
                    "score": score
                })

        logger.info(f"Reranking sonucu: {len(combined)}/{len(raw_docs)} chunk eşiği geçti "
                    f"(en_iyi_skor={en_iyi_skor:.4f}, dinamik_esik={dinamik_esik:.4f}).")

        # Genel sorular ("Bu formda ne isteniyor?", "Belge ne hakkında?") belgedeki
        # herhangi bir parçaya düşük benzerlik verir. Kullanıcı belirli bir belge
        # seçtiyse en iyi parçalar yine de LLM'e verilir; LLM sadece bağlamdan cevap
        # vermeye ve bilgi yoksa "bulunamadı" demeye yönlendirildiği için uydurma riski düşüktür.
        if not combined and dosya_secili:
            combined = [
                {"text": raw_docs[i], "meta": raw_metas[i] if i < len(raw_metas) else {},
                 "score": 1.0 - raw_dists[i]}
                for i in range(min(3, len(raw_docs)))
            ]
            logger.info(f"Eşik geçilemedi ama belge seçili — en iyi {len(combined)} parça bağlam olarak kullanılıyor.")

        # Belge seçili değilse ve soru hiçbir belgeyle yeterince ilgili değilse
        # ("Merhaba", "Ne yapabilirsin?" gibi) belge aramadan normal asistan gibi cevap ver.
        if not dosya_secili and (not combined or en_iyi_skor < GENEL_SORU_ESIGI):
            logger.info(f"Belgeyle ilgili olmayan genel soru (en_iyi_skor={en_iyi_skor:.4f}) — sohbet modunda cevaplanıyor.")
            genel_cevap = await claude_cevap(
                "Sen TR-DocuQuery adlı Türkçe belge asistanısın. Kullanıcı şu an belirli bir belge seçmedi. "
                "Kısa ve samimi bir şekilde cevap ver. Belgeyle ilgili bir şey sorarsa, önce ataç simgesiyle "
                "PDF ya da fotoğraf yüklemesini öner. Neler yapabildiğin sorulursa şunları say: belgeye soru sorma "
                "(kaynak ve madde numarasıyla), eksik bilgi tespiti, iki belgeyi karşılaştırma ve risk analizi, "
                "fotoğraftan metin okuma (OCR). Uydurma bilgi verme.",
                soru,
                max_tokens=600,
            )
            return {"cevap": genel_cevap, "kaynaklar": [], "missing_info": []}

        if not combined:
            return {"cevap": "Bu soruyla ilgili yeterli bilgi bulunamadı.", "kaynaklar": [], "missing_info": []}

        # Seçilen parçaları belgedeki doğal sıralarına (chunk_index) göre diz
        combined.sort(key=lambda x: int(cast(dict, x["meta"]).get("chunk_index", 9999)) if isinstance(x["meta"], dict) else 9999)
        ai_baglam = "\n---\n".join([str(item['text']) for item in combined])
        kaynaklar = [
            {
                "source": str(cast(dict, c["meta"]).get("dosya", "")),
                "page": int(cast(dict, c["meta"]).get("sayfa", 0)),
                "type": str(cast(dict, c["meta"]).get("madde_no", "")),
                "score": round(c.get("score", 0.0), 4)
            }
            for c in combined if isinstance(c["meta"], dict)
        ]

        # Claude
        logger.info(f"Claude ({ANTHROPIC_MODEL}) çağrısı yapılıyor...")
        system_msg = (
            "Sen Türkçe resmi belgeler ve sözleşmeler konusunda uzman bir belge asistanısın. "
            "Kullanıcının sorusunu SADECE <belge> etiketleri içindeki metne dayanarak yanıtla; "
            "belgede olmayan bilgiyi uydurma, bilgi yoksa bunu açıkça söyle. "
            "Cevabında 'bağlam', 'context', 'verilen metin' gibi ifadeler kullanma; bunun yerine "
            "'belgeye göre', 'sözleşmenin 3. maddesine göre' gibi doğal ifadeler kullan. "
            "Madde numarası varsa belirt. Kısa, net ve Türkçe cevap ver; gerekirse madde işaretli liste kullan."
        )

        try:
            full_cevap = await claude_cevap(system_msg, f"<belge>\n{ai_baglam}\n</belge>\n\nSORU: {soru}")
            logger.info("Claude yanıtı alındı.")
            cevap = full_cevap

            # Önce regex ile modelin kendi cevabındaki notu temizle (varsa)
            match = re.search(r"MISSING_INFO:\s*(\[.*?\])", full_cevap)
            if match:
                cevap = cevap.replace(match.group(0), "").strip()

            # Eksik bilgi kontrolü: belge seçiliyse belgenin TAMAMI üzerinden yapılır.
            # (Sadece bağlamdaki birkaç parçaya bakmak, belgenin başka yerinde olan
            # alanları da "eksik" gösteriyordu.)
            # Belge seçili değilse eksik bilgi paneli çalıştırılmaz.
            missing_fields: List[str] = []
            if dosya_secili:
                secili_dosyalar = req.dosya_adlari or [req.dosya_adi]
                kontrol_metni = "\n".join(get_full_text_by_filename(f) for f in secili_dosyalar)
                missing_fields = await get_missing_info(kontrol_metni)

            return {
                "cevap": cevap,
                "kaynaklar": kaynaklar,
                "missing_info": missing_fields
            }
        except RateLimitError:
            logger.error("Anthropic rate limit aşıldı.")
            raise HTTPException(429, "Anthropic rate limit aşıldı. Lütfen bekleyin.")
        except AuthenticationError:
            logger.error("Anthropic kimlik doğrulama hatası.")
            raise HTTPException(401, "Anthropic API anahtarı geçersiz.")
        except APITimeoutError:
            logger.error("Anthropic API zaman aşımı.")
            raise HTTPException(504, "Claude yanıt vermedi. Lütfen tekrar deneyin.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/sor hatası: {e}", exc_info=True)
        raise HTTPException(500, str(e))

# --- FOTOĞRAF ENDPOINT ---
@app.post("/sor/fotograf")
async def foto_analiz(file: UploadFile = File(...), soru: str = Form("Bu belgede ne yazıyor?")):
    """
    Frontend'den gelen fotoğrafı OCR ile okur ve Claude'a yorumlatır.
    """
    if not client: raise HTTPException(500, "Anthropic API Key yok!")

    req_id = uuid.uuid4().hex[:6]
    logger.info(f"[{req_id}] 📷 Fotoğraf Analizi İsteği: {file.filename}")

    try:
        contents = await file.read()

        # OCR — Thread içinde çalıştır
        ocr_result = await run_in_threadpool(goruntu_isleyerek_oku, contents)

        okunan_metin = ocr_result.get("text", "")
        ocr_guven = ocr_result.get("confidence", 0.0)
        ocr_uyari = ocr_result.get("warning")

        if ocr_uyari:
            logger.warning(f"[{req_id}] OCR uyarısı: {ocr_uyari}")

        if not okunan_metin or len(okunan_metin.strip()) < 5:
            return {
                "cevap": "Fotoğraftan anlamlı bir metin okunamadı. Lütfen daha net bir fotoğraf yükleyin.",
                "okunan_ham_veri": "",
                "ocr_confidence": ocr_guven,
                "ocr_warning": ocr_uyari,
                "missing_info": []
            }

        logger.info(f"[{req_id}] ✅ OCR Başarılı. {len(okunan_metin)} karakter, güven: {ocr_guven:.2f}")

        prompt = f"""<ocr_metni>
{okunan_metin}
</ocr_metni>

SORU: {soru}"""

        foto_sistem = (
            "Sen Türkçe belgeler konusunda uzman bir belge asistanısın. Kullanıcı bir belgenin fotoğrafını yükledi; "
            "<ocr_metni> içindeki metin bu fotoğraftan OCR ile okundu ve küçük harf hataları içerebilir. "
            "Hataları kendi içinde sessizce düzelterek anla ama düzelttiğin hataları ayrıca listeleme. "
            "Soruyu SADECE bu metne dayanarak, kısa ve net cevapla; belgede olmayan bilgiyi uydurma. "
            "Belge içeriğini anlatırken önemli bilgileri madde işaretli liste ile ver. "
            "'OCR', 'metin' gibi teknik ifadeler yerine 'belgede', 'fotoğraftaki belgede' gibi doğal ifadeler kullan."
        )

        try:
            ai_cevabi = await claude_cevap(foto_sistem, prompt, max_tokens=1024)
        except RateLimitError:
            raise HTTPException(429, "Anthropic rate limit aşıldı.")
        except AuthenticationError:
            raise HTTPException(401, "Anthropic API anahtarı geçersiz.")
        except APITimeoutError:
            raise HTTPException(504, "Claude yanıt vermedi. Lütfen tekrar deneyin.")

        missing_info = await get_missing_info(okunan_metin)

        return {
            "cevap": ai_cevabi,
            "okunan_ham_veri": okunan_metin,
            "ocr_confidence": ocr_guven,
            "ocr_warning": ocr_uyari,
            "missing_info": missing_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Fotoğraf Analiz Hatası: {e}", exc_info=True)
        raise HTTPException(500, f"İşlem başarısız: {str(e)}")

# --- NOT YÖNETİMİ ---
def get_user_file(user_email: str, filename: str) -> str:
    safe_email = user_email.replace('@', '_').replace('.', '_')
    user_dir = os.path.join("data", safe_email)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, filename)

class NoteCreate(BaseModel):
    content: str
    title: Optional[str] = None

class NoteUpdate(BaseModel):
    content: Optional[str] = None
    title: Optional[str] = None
    pinned: Optional[bool] = None

@app.get("/api/notes")
async def get_notes(x_user_email: str = Header(default="default_user")):
    notes_file = get_user_file(x_user_email, "notes.json")
    if not os.path.exists(notes_file):
        return []
    try:
        async with aiofiles.open(notes_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except Exception as e:
        logger.error(f"Notları okurken hata: {e}")
        return []

@app.post("/api/notes")
async def create_note(note: NoteCreate, x_user_email: str = Header(default="default_user")):
    notes_file = get_user_file(x_user_email, "notes.json")
    notes = []
    if os.path.exists(notes_file):
        try:
            async with aiofiles.open(notes_file, mode='r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    notes = json.loads(content)
        except Exception as e:
            logger.error(f"Notları okurken hata (ekleme öncesi): {e}")
            raise HTTPException(status_code=500, detail="Mevcut notlar okunamadı.")

    new_note = {
        "id": str(uuid.uuid4()),
        "content": note.content,
        "title": note.title or "İsimsiz Not",
        "pinned": False,
        "timestamp": datetime.datetime.now().isoformat()
    }
    notes.append(new_note)

    try:
        async with aiofiles.open(notes_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(notes, ensure_ascii=False, indent=4))
        return new_note
    except Exception as e:
        logger.error(f"Not yazma hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Dosyaya yazma hatası: Not kaydedilemedi ({str(e)}).")

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str, x_user_email: str = Header(default="default_user")):
    notes_file = get_user_file(x_user_email, "notes.json")
    if not os.path.exists(notes_file):
        raise HTTPException(status_code=404, detail="Not bulunamadı.")

    try:
        async with aiofiles.open(notes_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content.strip():
                raise HTTPException(status_code=404, detail="Not bulunamadı.")
            notes = json.loads(content)

        initial_length = len(notes)
        notes = [n for n in notes if n.get("id") != note_id]

        if len(notes) == initial_length:
            raise HTTPException(status_code=404, detail="Not bulunamadı.")

        async with aiofiles.open(notes_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(notes, ensure_ascii=False, indent=4))

        return {"success": True, "detail": "Not başarıyla silindi."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Not silme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Not silinirken hata oluştu: ({str(e)}).")

@app.put("/api/notes/update/{note_id}")
async def update_note(note_id: str, note: NoteUpdate, x_user_email: str = Header(default="default_user")):
    notes_file = get_user_file(x_user_email, "notes.json")
    if not os.path.exists(notes_file):
        raise HTTPException(status_code=404, detail="Not bulunamadı.")

    try:
        async with aiofiles.open(notes_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content.strip():
                raise HTTPException(status_code=404, detail="Not bulunamadı.")
            notes = json.loads(content)

        updated = False
        for n in notes:
            if n.get("id") == note_id:
                if note.content is not None:
                    n["content"] = note.content
                if note.title is not None:
                    n["title"] = note.title
                if note.pinned is not None:
                    n["pinned"] = note.pinned
                updated = True
                break

        if not updated:
            raise HTTPException(status_code=404, detail="Not bulunamadı.")

        async with aiofiles.open(notes_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(notes, ensure_ascii=False, indent=4))

        return {"success": True, "detail": "Not başarıyla güncellendi."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Not güncelleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Not güncellenirken hata oluştu: ({str(e)}).")

# --- GEÇMİŞ YÖNETİMİ ---
@app.get("/api/history")
async def get_history(x_user_email: str = Header(default="default_user")):
    history_file = get_user_file(x_user_email, "history.json")
    if not os.path.exists(history_file):
        return []
    try:
        async with aiofiles.open(history_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except Exception as e:
        logger.error(f"Geçmiş okuma hatası: {e}")
        return []

@app.post("/api/history")
async def save_history(history_data: list = Body(...), x_user_email: str = Header(default="default_user")):
    history_file = get_user_file(x_user_email, "history.json")
    try:
        async with aiofiles.open(history_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(history_data, ensure_ascii=False, indent=4))
        return {"success": True}
    except Exception as e:
        logger.error(f"Geçmiş yazma hatası: {e}")
        raise HTTPException(status_code=500, detail="Geçmiş kaydedilemedi.")

@app.post("/api/migrate-guest")
async def migrate_guest_data(x_user_email: str = Header(default="default_user")):
    if x_user_email == "guest" or x_user_email == "default_user":
        return {"status": "skipped", "message": "Geçersiz veya eksik hedef e-posta."}

    guest_dir = get_user_file("guest", "")
    if not os.path.exists(guest_dir):
        return {"status": "skipped", "message": "Aktarılacak misafir verisi bulunamadı."}

    target_dir = get_user_file(x_user_email, "")
    files_to_merge = ["notes.json", "calendar.json", "history.json"]

    merged_any = False

    for filename in files_to_merge:
        guest_file = os.path.join(guest_dir, filename)
        target_file = os.path.join(target_dir, filename)

        if os.path.exists(guest_file):
            try:
                with open(guest_file, 'r', encoding='utf-8') as f:
                    guest_data = json.load(f)

                if not isinstance(guest_data, list) or len(guest_data) == 0:
                    continue

                target_data = []
                if os.path.exists(target_file):
                    with open(target_file, 'r', encoding='utf-8') as f:
                        target_data = json.load(f)

                if isinstance(target_data, list):
                    merged_data = target_data + guest_data
                    with open(target_file, 'w', encoding='utf-8') as f:
                        json.dump(merged_data, f, ensure_ascii=False, indent=4)
                    merged_any = True
            except Exception as e:
                logger.error(f"{filename} aktarılırken hata: {e}")

    # Migration bittikten sonra guest klasörünün içini temizle
    try:
        if os.path.exists(guest_dir):
            for filename in os.listdir(guest_dir):
                file_path = os.path.join(guest_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Guest dosyası silinirken hata {file_path}: {e}")
    except Exception as e:
        logger.error(f"Guest dizini temizlenirken hata: {e}")

    return {"status": "success", "message": "Misafir verileri başarıyla hesabınıza aktarıldı." if merged_any else "Aktarılacak veri yoktu."}


# --- GOOGLE CALENDAR ENTEGRASYONU ---
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service(access_token: str):
    """Google Calendar API'sine erişim sağlamak için frontend'den gelen tokeni kullanır."""
    if not access_token:
        raise ValueError("Google erişim izni (token) bulunamadı.")

    creds = Credentials(token=access_token)
    service = build('calendar', 'v3', credentials=creds)
    return service

class ExecuteTaskRequest(BaseModel):
    task_id: Union[int, str]
    action: str
    task_title: Optional[str] = "Bilinmeyen Görev"
    task_date: Optional[str] = None

@app.get("/api/calendar/events")
async def get_calendar_events(x_user_email: str = Header(default="default_user"), x_google_token: str = Header(default="")):
    events_list = []

    # 1. Google Takvim'den çek
    if x_google_token:
        try:
            service = get_calendar_service(x_google_token)
            now = datetime.datetime.utcnow().isoformat() + 'Z'
            events_result = service.events().list(calendarId='primary', timeMin=now,
                                                  maxResults=10, singleEvents=True,
                                                  orderBy='startTime').execute()
            google_events = events_result.get('items', [])
            for ge in google_events:
                start = ge['start'].get('dateTime', ge['start'].get('date'))
                events_list.append({
                    "id": ge['id'],
                    "title": ge.get('summary', 'İsimsiz Etkinlik'),
                    "start": start,
                    "source": "google"
                })
        except Exception as e:
            logger.error(f"Google Calendar GET hatası: {e}")

    # 2. Yerel calendar.json'dan çek
    calendar_file = get_user_file(x_user_email, "calendar.json")
    if os.path.exists(calendar_file):
        try:
            with open(calendar_file, 'r', encoding='utf-8') as f:
                local_events = json.load(f)
            now_dt = datetime.datetime.now()
            for le in local_events:
                try:
                    le_dt = datetime.datetime.fromisoformat(le['start'])
                    if le_dt > now_dt:
                        events_list.append(le)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Yerel takvim okuma hatası: {e}")

    events_list.sort(key=lambda x: x['start'])
    return {"events": events_list}

@app.post("/api/action/calendar/add")
async def execute_task(req: ExecuteTaskRequest, x_user_email: str = Header(default="default_user"), x_google_token: str = Header(default="")):
    if req.action == "calendar_event":
        logger.info(f"'{req.task_title}' takvime ekleniyor...")

        event_link = ""
        start_time_iso = ""

        # 1. Google Takvim'e kaydet
        if x_google_token:
            try:
                service = get_calendar_service(x_google_token)

                now = datetime.datetime.now()
                start_time = now + datetime.timedelta(hours=1)
                end_time = start_time + datetime.timedelta(hours=1)

                if req.task_date:
                    try:
                        selected_date = datetime.datetime.strptime(req.task_date, "%Y-%m-%d")
                        if selected_date.date() != now.date():
                            start_time = selected_date.replace(hour=10, minute=0, second=0)
                            end_time = start_time + datetime.timedelta(hours=1)
                    except Exception:
                        pass

                start_time_iso = start_time.isoformat()

                event = {
                    'summary': req.task_title,
                    'description': 'AI Asistan tarafından uygulamanız aracılığıyla eklendi.',
                    'start': {'dateTime': start_time.isoformat() + '+03:00', 'timeZone': 'Europe/Istanbul'},
                    'end': {'dateTime': end_time.isoformat() + '+03:00', 'timeZone': 'Europe/Istanbul'},
                }

                event_result = service.events().insert(calendarId='primary', body=event).execute()
                event_link = event_result.get('htmlLink')
            except Exception as e:
                logger.error(f"Google Calendar API Hatası: {e}")
            # Hata olsa bile local'e kaydetmeye devam et

        # 2. Yerel calendar.json'a yedekle (Dual Save)
        local_event = {
            "id": str(uuid.uuid4()),
            "title": req.task_title,
            "start": start_time_iso or datetime.datetime.now().isoformat(),
            "source": "local"
        }

        calendar_file = get_user_file(x_user_email, "calendar.json")
        local_events = []
        if os.path.exists(calendar_file):
            try:
                with open(calendar_file, 'r', encoding='utf-8') as f:
                    local_events = json.load(f)
            except Exception:
                pass

        local_events.append(local_event)

        try:
            with open(calendar_file, 'w', encoding='utf-8') as f:
                json.dump(local_events, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Yerel takvim yazma hatası: {e}")

        return {
            "status": "success",
            "message": "Etkinlik başarıyla eklendi!",
            "event_link": event_link
        }

    return {"status": "error", "message": "Geçersiz action."}


# =============================================================================
# AGENT ENDPOINTS
# =============================================================================

class TaskRequest(BaseModel):
    metin: str

class ProactiveRequest(BaseModel):
    sohbet_gecmisi: List[str]

class AuditorRequest(BaseModel):
    doc1_text: Optional[str] = None
    doc2_text: Optional[str] = None
    doc1_name: Optional[str] = None
    doc2_name: Optional[str] = None
    filenames: Optional[List[str]] = None

class ValidatorRequest(BaseModel):
    extracted_text: str


@app.post("/agent/task-extract")
async def extract_task_endpoint(req: TaskRequest):
    logger.info(f"/agent/task-extract istegi alindi ({len(req.metin)} karakter).")
    try:
        baslangic = {
            "metin": req.metin,
            "cikarim_sonucu": None,
            "islem_durumu": "BASLADI",
            "error": None
        }
        # Senkron graf thread'de çalışır -> sunucu diğer istekleri beklemeden karşılar
        tamamlanmis = await run_in_threadpool(graph_invoke_with_timeout, baslangic, 30)

        if tamamlanmis.get("error"):
            logger.warning(f"Agent hata ile tamamlandi: {tamamlanmis['error']}")
            return JSONResponse(
                status_code=500,
                content={"error": True, "message": tamamlanmis["error"], "code": 500}
            )

        sonuc_model = tamamlanmis.get("cikarim_sonucu")
        sonuc_dict = sonuc_model.model_dump() if sonuc_model else None
        logger.info("Gorev cikarimi basariyla tamamlandi.")
        return {
            "success": True,
            "islem_durumu": tamamlanmis.get("islem_durumu"),
            "data": sonuc_dict
        }
    except TimeoutError as e:
        logger.error(f"Timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit asildi. Lutfen bekleyin.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API kimlik hatasi.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Anthropic API zaman asimi.")
    except Exception as e:
        logger.error(f"Gorev cikarimi hatasi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gorev cikarimi sirasinda hata: {str(e)}")


@app.post("/agent/proactive-search")
async def proactive_search_endpoint(req: ProactiveRequest):
    logger.info(f"/agent/proactive-search istegi alindi ({len(req.sohbet_gecmisi)} mesaj).")
    try:
        baslangic = {
            "sohbet_gecmisi": req.sohbet_gecmisi,
            "ilgi_alanlari": [],
            "arama_sonuclari": {},
            "error": None
        }
        sonuc = await run_in_threadpool(proactive_graph.invoke, baslangic)
        if sonuc.get("error"):
            logger.warning(f"Proaktif arama hata ile tamamlandi: {sonuc['error']}")
        logger.info("Proaktif arama tamamlandi.")
        return {
            "success": True,
            "ilgi_alanlari": sonuc.get("ilgi_alanlari", []),
            "arama_sonuclari": sonuc.get("arama_sonuclari", {}),
            "error": sonuc.get("error")
        }
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit asildi.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API kimlik hatasi.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Anthropic API zaman asimi.")
    except Exception as e:
        logger.error(f"Proaktif arama hatasi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Proaktif arama sirasinda hata: {str(e)}")


@app.post("/agent/audit-documents")
async def audit_documents_endpoint(req: AuditorRequest):
    logger.info("/agent/audit-documents istegi alindi.")
    try:
        texts = []
        if req.filenames:
            for fname in req.filenames:
                txt = get_full_text_by_filename(fname)
                if txt: texts.append({"name": fname, "text": txt})

        # Geriye dönük uyumluluk
        if not texts:
            doc1 = req.doc1_text or (get_full_text_by_filename(req.doc1_name) if req.doc1_name else None)
            doc2 = req.doc2_text or (get_full_text_by_filename(req.doc2_name) if req.doc2_name else None)
            if doc1: texts.append({"name": req.doc1_name or "Belge 1", "text": doc1})
            if doc2: texts.append({"name": req.doc2_name or "Belge 2", "text": doc2})

        if len(texts) < 2:
            raise HTTPException(400, "Karşılaştırılacak en az iki belge gereklidir.")

        initial_state = {
            "doc1_text": texts[0]["text"],
            "doc2_text": texts[1]["text"],
            "comparison_results": {},
            "executive_summary": "",
            "error": None
        }
        final_state = await auditor_app.ainvoke(initial_state)

        if final_state.get("error"):
            logger.warning(f"Denetim hata ile tamamlandi: {final_state['error']}")
            return JSONResponse(
                status_code=500,
                content={"error": True, "message": final_state["error"], "code": 500}
            )

        res = final_state.get("comparison_results", {})
        res["timestamp"] = datetime.datetime.now().isoformat()

        logger.info("Dokuman denetimi basariyla tamamlandi.")
        return {
            "success": True,
            "comparison_results": res,
            "executive_summary": final_state.get("executive_summary", "")
        }
    except HTTPException:
        raise
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit asildi.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API kimlik hatasi.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Anthropic API zaman asimi.")
    except Exception as e:
        logger.error(f"Dokuman denetimi hatasi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dokuman denetimi sirasinda hata: {str(e)}")


@app.post("/agent/validate-document")
async def validate_document_endpoint(req: ValidatorRequest):
    logger.info("/agent/validate-document istegi alindi.")
    try:
        initial_state = {
            "extracted_text": req.extracted_text,
            "validation_results": {},
            "error": None
        }
        final_state = await validator_app.ainvoke(initial_state)

        if final_state.get("error"):
            logger.warning(f"Dogrulama hata ile tamamlandi: {final_state['error']}")
            return JSONResponse(
                status_code=500,
                content={"error": True, "message": final_state["error"], "code": 500}
            )

        logger.info("Belge dogrulamasi basariyla tamamlandi.")
        return {
            "success": True,
            "validation_results": final_state.get("validation_results", {})
        }
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit asildi.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API kimlik hatasi.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Anthropic API zaman asimi.")
    except Exception as e:
        logger.error(f"Belge dogrulama hatasi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Belge dogrulamasi sirasinda hata: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)