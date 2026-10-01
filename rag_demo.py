import os

import chromadb
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- AYARLAR ---
st.set_page_config(page_title="RAG Asistanı", layout="wide")
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY bulunamadı. Proje klasöründeki .env dosyasına "
             "`ANTHROPIC_API_KEY=...` satırını ekleyin.")
    st.stop()

# LLM'in davranış kuralları: sadece bağlamdan cevap ver, kaynak göster, uydurma
SYSTEM_PROMPT = """Sen belgeler üzerinden soru yanıtlayan bir asistansın.
Kurallar:
1. SADECE verilen BAĞLAM içindeki bilgileri kullan. Kendi bilgini ekleme.
2. Her bilginin sonunda hangi kaynaktan geldiğini [1], [2] şeklinde belirt.
3. Cevap bağlamda yoksa: "Yüklenen belgelerde bu soruya ait bilgi bulunamadı." de.
4. Madde numarası varsa cevapta belirt (ör. "Madde 6'ya göre...").
5. Türkçe, açık ve kısa cevap ver."""


# --- SİSTEM ---
@st.cache_resource
def sistemi_yukle():
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    db = chromadb.PersistentClient(path="chroma_db")
    collection = db.get_or_create_collection(
        name="rag_koleksiyonu", metadata={"hnsw:space": "cosine"}
    )
    llm = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60, max_retries=2)
    return embedder, collection, llm


def kaynak_etiketi(meta):
    etiket = f"📄 **{meta.get('source', 'Bilinmiyor')}** — Sayfa {meta.get('page', '?')}"
    if meta.get("section_number") is not None:
        etiket += f" · Madde {meta['section_number']}"
        if meta.get("section_title"):
            etiket += f" ({meta['section_title']})"
    return etiket


# --- ARAYÜZ ---
st.title("🤖 Yapay Zeka Doküman Asistanı")
st.caption(f"Claude ({ANTHROPIC_MODEL}) + Madde ve Sayfa Referanslı RAG")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    esik_degeri = st.slider("Benzerlik Eşiği", 0.0, 1.0, 0.40)
    sonuc_sayisi = st.slider("Kaynak Sayısı", 1, 10, 3)

embedder, collection, llm = sistemi_yukle()
kayit_sayisi = collection.count()

if kayit_sayisi == 0:
    st.warning("Veritabanı boş. Önce belgeleri işleyin: `python process_documents.py`")
    st.stop()

st.success(f"Sistem Hazır ✅ — {kayit_sayisi} parça indekslendi.")

soru = st.text_input("Sorunuzu Yazın:")
if st.button("🔍 Yanıtla") and soru:
    with st.spinner("Araştırılıyor..."):
        # 1) RETRIEVAL: soruyu vektöre çevir, en yakın parçaları getir
        soru_vektoru = embedder.encode(soru).tolist()
        sonuclar = collection.query(
            query_embeddings=[soru_vektoru],
            n_results=min(sonuc_sayisi, kayit_sayisi),
        )

        # 2) FİLTRE: benzerlik eşiğinin altındakileri ele
        bulunanlar = []
        for doc, dist, meta in zip(
            sonuclar["documents"][0], sonuclar["distances"][0], sonuclar["metadatas"][0]
        ):
            skor = 1 - dist  # cosine distance -> similarity
            if skor >= esik_degeri:
                bulunanlar.append({"text": doc, "meta": meta or {}, "skor": skor})

    if not bulunanlar:
        st.warning("Eşik değerinin üzerinde ilgili bilgi bulunamadı. Eşiği düşürmeyi deneyin.")
        st.stop()

    # 3) AUGMENTATION: numaralı kaynaklarla bağlam oluştur
    baglam = "\n\n".join(
        f"[{i}] (Kaynak: {b['meta'].get('source', '?')}, Sayfa {b['meta'].get('page', '?')})\n{b['text']}"
        for i, b in enumerate(bulunanlar, start=1)
    )
    prompt = f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"

    # 4) GENERATION: Claude ile cevap üret
    try:
        with st.spinner("Cevap üretiliyor..."):
            response = llm.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            cevap = "".join(b.text for b in response.content if b.type == "text")

        st.markdown("### 💡 Cevap")
        st.write(cevap)

        st.markdown("---")
        st.markdown("### 📚 Kaynaklar")
        for i, b in enumerate(bulunanlar, start=1):
            with st.expander(f"[{i}] {kaynak_etiketi(b['meta'])} — benzerlik: {b['skor']:.2f}"):
                st.write(b["text"])

    except Exception as e:
        st.error(f"LLM Hatası: {e}")