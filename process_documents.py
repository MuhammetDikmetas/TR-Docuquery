import os
import re
import json
import logging
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

from pdf_reader import PDFProcessor

logger = logging.getLogger("DocumentProcessor")


class DocumentProcessor:
    """
    RAG indeksleme hattı:
    PDF / taranmış PDF  ->  sayfa bazlı metin  ->  madde bazlı (yapısal) chunking
    ->  çok dilli embedding  ->  ChromaDB (cosine)
    """

    # Madde / Bölüm başlığı yakalayıcı:
    #  - Sadece metin başında veya cümle sonu (. : ; ! ? ) \n) sonrasında eşleşir
    #    -> cümle içindeki "... Madde 5 uyarınca ..." gibi atıflar yeni madde sayılmaz
    #  - Rakamdan sonra kesme işareti gelirse eşleşmez -> "Madde 5'te" atıfları elenir
    MADDE_PATTERN = re.compile(
        r"(?:^|(?<=[\n.:;!?)]))\s*(?:MADDE|BÖLÜM)\s*(\d+)(?![\d'’])\s*[:\-–.)]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )

    MAX_CHUNK_SIZE = 800   # karakter
    OVERLAP = 100          # karakter

    def __init__(self, persist_directory="chroma_db"):
        print("⚙️ İşlemci Başlatılıyor...")

        self.log_file = "processed_files_log.json"
        self.processed_files = self.gecmisi_yukle()

        # Embedding Modeli (50+ dil, Türkçe dahil)
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Vektör Veritabanı
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_koleksiyonu",
            metadata={"hnsw:space": "cosine"}
        )

        self.pdf_reader = PDFProcessor()
        print("✅ İşlemci Hazır.")

    # ========================================
    # İşlenmiş dosya geçmişi
    # ========================================
    def gecmisi_yukle(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def gecmisi_kaydet(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=4)

    # ========================================
    # Yardımcılar
    # ========================================
    @staticmethod
    def _temiz_metadata(metadata):
        """ChromaDB metadata'da None kabul etmez -> None alanları çıkar."""
        return {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _baslik_ayikla(govde):
        """
        Madde gövdesinin başındaki başlığı ayırır. Desteklenen biçimler:
          "KONU\\nBu sözleşmenin..."          -> satır sonu ile ayrılmış başlık
          "FESİH VE İPTAL Taraflardan..."     -> BÜYÜK HARF başlık
          "Sözleşmenin Konusu: Bu sözleşme..." -> iki nokta ile ayrılmış başlık
        Returns: (baslik, metin)
        """
        govde = govde.strip()

        ilk_satir, sep, kalan = govde.partition("\n")
        if sep and 0 < len(ilk_satir.strip()) <= 100:
            return ilk_satir.strip(), kalan.strip()

        buyuk = re.match(
            r"((?:[A-ZÇĞİÖŞÜ0-9&/,\-]+\s+){1,8}?)(?=[A-ZÇĞİÖŞÜ][a-zçğıöşü])", govde
        )
        if buyuk:
            return buyuk.group(1).strip(" -,"), govde[buyuk.end():].strip()

        iki_nokta = re.match(r"([^.:]{3,60}):\s+", govde)
        if iki_nokta:
            return iki_nokta.group(1).strip(), govde[iki_nokta.end():].strip()

        return "", govde

    # ========================================
    # 1️⃣ MADDE BAZLI (YAPISAL) CHUNKING
    # ========================================
    def madde_bazli_chunk(self, text, file_name, page_num=None, extraction_method=None):
        """
        Sözleşme / yönetmelik metnini madde numarasına göre böler.
        - Her madde = 1 chunk (+ metadata). Çok uzun maddeler alt parçalara bölünür,
          her parçanın başına madde başlığı eklenir (bağlam kaybolmaz).
        - İlk maddeden önceki metin (giriş / önceki sayfadan devam) KAYBOLMAZ,
          ayrı chunk olarak eklenir.
        - Madde yoksa paragraf + overlap tabanlı klasik chunking yapılır.
        """
        chunks = []
        matches = list(self.MADDE_PATTERN.finditer(text))

        def ekle(chunk_text, **meta):
            metadata = {
                "source": file_name,
                "page": page_num,
                "type": "text",
                "extraction": extraction_method,
            }
            metadata.update(meta)
            chunks.append({"text": chunk_text, "metadata": self._temiz_metadata(metadata)})

        if not matches:
            print(f"   ⚠️ Madde bulunamadı, geleneksel chunking yapılıyor (Sayfa {page_num})")
            for idx, parca in enumerate(self.chunk_text(text, self.MAX_CHUNK_SIZE, self.OVERLAP)):
                ekle(parca, section_title="Genel", chunk_type="genel", chunk_index=idx)
            return chunks

        # --- İlk maddeden önceki metin (giriş veya önceki sayfadaki maddenin devamı) ---
        on_metin = text[:matches[0].start()].strip()
        if len(on_metin) >= 20:
            tur = "giris" if (page_num in (None, 1)) else "devam"
            baslik = "Giriş" if tur == "giris" else "Önceki maddenin devamı"
            for idx, parca in enumerate(self.chunk_text(on_metin, self.MAX_CHUNK_SIZE, self.OVERLAP)):
                ekle(parca, section_title=baslik, chunk_type=tur, chunk_index=idx)

        # --- Maddeler ---
        madde_sayisi = 0
        for i, match in enumerate(matches):
            madde_no = int(match.group(1))
            bitis = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            baslik, madde_metni = self._baslik_ayikla(text[match.end():bitis])

            if len(madde_metni) < 20:
                continue

            etiket = f"MADDE {madde_no}" + (f": {baslik}" if baslik else "")
            parcalar = (
                [madde_metni] if len(madde_metni) <= self.MAX_CHUNK_SIZE
                else self.chunk_text(madde_metni, self.MAX_CHUNK_SIZE, self.OVERLAP)
            )
            for idx, parca in enumerate(parcalar):
                ekle(
                    f"{etiket}\n\n{parca}",
                    section_number=madde_no,
                    section_title=baslik or f"Madde {madde_no}",
                    chunk_type="madde",
                    chunk_index=idx,
                )
            madde_sayisi += 1

        print(f"   📌 {madde_sayisi} madde tespit edildi (Sayfa {page_num})")
        return chunks

    def chunk_text(self, text, max_chunk_size=800, overlap=100):
        """
        Metni önce doğal paragraflarına (\\n\\n) böler.
        Uzun paragrafları karakter limitine göre keserken 'overlap' kadar
        karakter önceki chunk'tan taşınır — bağlam kopukluğunu önler.
        Kesimler kelime ortasına denk gelmez.
        """
        chunks = []
        paragraflar = re.split(r"\n\s*\n", text)

        guncel_chunk = ""
        for paragraf in paragraflar:
            paragraf = paragraf.strip()
            if not paragraf:
                continue

            if len(paragraf) > max_chunk_size:
                if guncel_chunk:
                    chunks.append(guncel_chunk.strip())
                    guncel_chunk = ""

                start = 0
                while start < len(paragraf):
                    end = start + max_chunk_size
                    if end < len(paragraf):
                        # Güvenli bölme noktası bul (nokta/boşluk)
                        bak = end
                        while bak > start and paragraf[bak] not in (".", " ", "\n"):
                            bak -= 1
                        if bak > start:
                            end = bak + 1 if paragraf[bak] == "." else bak
                    parca = paragraf[start:end].strip()
                    if parca:
                        chunks.append(parca)
                    if end >= len(paragraf):
                        break
                    # Bir sonraki chunk 'overlap' kadar geri başlar (kelime başına hizalanır)
                    yeni_start = end - overlap
                    if yeni_start > start:
                        bosluk = paragraf.find(" ", yeni_start, end)
                        start = bosluk + 1 if bosluk != -1 else yeni_start
                    else:
                        start = end
            else:
                if len(guncel_chunk) + len(paragraf) + 2 <= max_chunk_size:
                    guncel_chunk = guncel_chunk + "\n\n" + paragraf if guncel_chunk else paragraf
                else:
                    if guncel_chunk:
                        chunks.append(guncel_chunk.strip())
                    # Yeni chunk'ı önceki chunk'ın son 'overlap' karakteriyle başlat
                    kuyruk = guncel_chunk[-overlap:]
                    if " " in kuyruk:
                        kuyruk = kuyruk[kuyruk.index(" ") + 1:]
                    guncel_chunk = (kuyruk + "\n\n" + paragraf).strip() if kuyruk else paragraf

        if guncel_chunk:
            chunks.append(guncel_chunk.strip())

        logger.debug(f"chunk_text tamamlandı: {len(chunks)} parça üretildi.")
        return chunks

    # ========================================
    # Ortak: PDF -> chunk listesi
    # ========================================
    def _dosyadan_chunklar(self, file_path):
        file_name = os.path.basename(file_path)
        pages_data = self.pdf_reader.extract_text_with_metadata(file_path)

        if not pages_data:
            print("❌ HATA: Dosyadan hiç metin okunamadı!")
            return []

        tum_chunks = []
        for page_data in pages_data:
            raw_text = page_data["text"]
            page_num = page_data["page"]

            if len(raw_text) < 50:
                print(f"⚠️ Sayfa {page_num} atlandı (çok az veri: {len(raw_text)} karakter)")
                continue

            tum_chunks.extend(
                self.madde_bazli_chunk(raw_text, file_name, page_num, page_data.get("method"))
            )
        return tum_chunks

    # ========================================
    # 2️⃣ TEKİL DOSYA İŞLEME
    # ========================================
    def process_single_file(self, file_path):
        """Tek bir dosyayı işler ve ChromaDB'ye yazar."""
        file_name = os.path.basename(file_path)
        print(f"\n🚀 Tekil Dosya İşleniyor: {file_name}")

        tum_chunks = self._dosyadan_chunklar(file_path)
        if not tum_chunks:
            print("⚠️ Dosya işlendi ama veritabanına eklenecek veri çıkmadı.")
            return False

        # Deterministik ID: aynı dosya tekrar işlenirse kopya oluşmaz,
        # aynı madde numarası iki kez geçse bile ID çakışmaz.
        ids = [
            hashlib.md5(f"{file_name}|{c['metadata'].get('page')}|{i}".encode("utf-8")).hexdigest()
            for i, c in enumerate(tum_chunks)
        ]
        documents = [c["text"] for c in tum_chunks]
        metadatas = [c["metadata"] for c in tum_chunks]

        # Toplu (batch) embedding — tek tek encode etmekten çok daha hızlı
        embeddings = self.embedder.encode(
            documents, batch_size=32, show_progress_bar=False
        ).tolist()

        # Dosyanın eski kayıtlarını temizle (dosya güncellendiyse eski chunk'lar kalmasın)
        try:
            self.collection.delete(where={"source": file_name})
        except Exception as e:
            logger.debug(f"Eski kayıt silme atlandı: {e}")

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"🎉 '{file_name}' başarıyla veritabanına eklendi ({len(tum_chunks)} chunk).")

        if file_name not in self.processed_files:
            self.processed_files.append(file_name)
            self.gecmisi_kaydet()
        return True

    # ========================================
    # 3️⃣ KLASÖR İŞLEME
    # ========================================
    def process_folder(self, folder_path="documents"):
        """Klasördeki tüm PDF dosyalarını tarar"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

        if not files:
            print("⚠️ Klasörde PDF dosyası bulunamadı.")
            return

        print(f"\n📂 {len(files)} PDF dosyası bulundu.")

        for file in files:
            if file in self.processed_files:
                print(f"⏭️ '{file}' daha önce işlenmiş, atlanıyor.")
                continue
            self.process_single_file(os.path.join(folder_path, file))

    # ========================================
    # 4️⃣ API İÇİN ÖZEL FONKSİYON
    # ========================================
    def process_single_file_with_madde_chunking(self, file_path):
        """
        API endpoint'i için: chunk listesini döner
        (ChromaDB'ye ekleme API tarafında yapılır).
        """
        file_name = os.path.basename(file_path)
        print(f"\n🚀 Maddeli Chunking ile İşleniyor: {file_name}")
        tum_chunks = self._dosyadan_chunklar(file_path)
        print(f"✅ Toplam {len(tum_chunks)} chunk oluşturuldu")
        return tum_chunks