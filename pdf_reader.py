import re
import numpy as np
import pymupdf as fitz  # PyMuPDF (eski 'import fitz' kullanımı kaldırılacak)


class PDFProcessor:
    """
    PDF'lerden sayfa bazlı metin çıkarır.
    - Metin katmanı olan sayfalar: PyMuPDF ile doğrudan okunur.
    - Taranmış / metin katmanı olmayan sayfalar: sayfa görüntüye çevrilip
      ocr_engine (EasyOCR + görüntü ön işleme) ile okunur.
    """

    MIN_TEXT_LENGTH = 50   # Bu uzunluğun altındaki sayfalar "taranmış" kabul edilir
    OCR_DPI = 300          # OCR için sayfa render çözünürlüğü

    def __init__(self, ocr_fallback: bool = True):
        self.ocr_fallback = ocr_fallback
        self._ocr_func = None  # EasyOCR sadece gerektiğinde yüklenir (lazy load)

    def _get_ocr(self):
        if self._ocr_func is None:
            from ocr_engine import ocr_ile_oku
            self._ocr_func = ocr_ile_oku
        return self._ocr_func

    def clean_text(self, text):
        """Metin temizliği"""
        if not text:
            return ""
        text = text.replace("-\n", "")
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _page_to_image(self, page):
        """PDF sayfasını OpenCV'nin beklediği BGR numpy dizisine çevirir."""
        pix = page.get_pixmap(dpi=self.OCR_DPI, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:  # gri tonlamalı sayfa
            img = np.repeat(img, 3, axis=2)
        return img[:, :, ::-1].copy()  # RGB -> BGR

    def extract_text_with_metadata(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            results = []

            print(f"--- Dosya Analizi: {pdf_path} ---")

            for page_num, page in enumerate(doc):
                # sort=True: metni sayfadaki fiziksel konumuna göre (yukarıdan aşağıya) sıralar
                text = page.get_text("text", sort=True)
                print(f"[Sayfa {page_num + 1}] Okunan ham veri uzunluğu: {len(text)}")

                # --- Metin katmanı yeterliyse doğrudan kullan ---
                if len(text.strip()) >= self.MIN_TEXT_LENGTH or not self.ocr_fallback:
                    cleaned = self.clean_text(text)
                    if cleaned:
                        results.append({
                            "text": cleaned,
                            "page": page_num + 1,
                            "method": "text"
                        })
                    continue

                # --- Taranmış sayfa: OCR fallback ---
                print(f"⚠️ Sayfa {page_num + 1} metin katmanı içermiyor, OCR'a gönderiliyor...")
                try:
                    ocr_sonuc = self._get_ocr()(self._page_to_image(page))
                    cleaned = self.clean_text(ocr_sonuc.get("text", ""))
                    if ocr_sonuc.get("warning"):
                        print(f"   OCR uyarısı: {ocr_sonuc['warning']}")
                    if cleaned:
                        results.append({
                            "text": cleaned,
                            "page": page_num + 1,
                            "method": "ocr",
                            "confidence": ocr_sonuc.get("confidence", 0.0)
                        })
                except Exception as e:
                    print(f"OCR fallback hatası (Sayfa {page_num + 1}): {e}")
                    # OCR başarısızsa elde ne varsa onu kullan
                    cleaned = self.clean_text(text)
                    if cleaned:
                        results.append({"text": cleaned, "page": page_num + 1, "method": "text"})

            doc.close()
            return results

        except Exception as e:
            print(f"Hata oluştu ({pdf_path}): {e}")
            return []