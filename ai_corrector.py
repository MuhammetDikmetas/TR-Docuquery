
from transformers import pipeline


class AITextCorrector:
    """
    BERTurk (dbmdz/bert-base-turkish-cased) ile maskeli dil modeli tabanlı
    eksik kelime tamamlama. OCR'ın okuyamadığı / bozuk okuduğu kelimeler
    [MASK] ile işaretlenir, model bağlama bakarak en olası kelimeyi tahmin eder.
    """

    MASK = "[MASK]"

    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased"):
        print("🤖 BERTurk Yapay Zeka Modeli Yükleniyor... (İlk seferde indirecektir)")
        self.fill_mask = pipeline("fill-mask", model=model_name, tokenizer=model_name)

    def eksik_kelime_tamamla(self, metin: str) -> str:
        """
        İçinde [MASK] olan cümleleri tamamlar. Birden fazla maske varsa
        soldan sağa sırayla doldurur; her adımda önceki tahminler bağlama dahil olur.
        Örn: "İş sağlığı ve [MASK] önlemleri" -> "İş sağlığı ve güvenliği önlemleri"
        """
        if not metin or self.MASK not in metin:
            return metin

        try:
            while self.MASK in metin:
                sonuclar = self.fill_mask(metin)

                # Tek maske -> [ {..}, {..} ]
                # Çok maske -> [ [ {..}, .. ], [ {..}, .. ] ]  (ilki = soldaki maske)
                adaylar = sonuclar[0] if isinstance(sonuclar[0], list) else sonuclar
                en_iyi = adaylar[0]

                kelime = en_iyi["token_str"].strip()
                print(f"   🔎 [MASK] -> '{kelime}' (skor: {en_iyi['score']:.2f})")

                # Sadece ilk maskeyi değiştir, metnin geri kalanına dokunma
                metin = metin.replace(self.MASK, kelime, 1)

            return metin

        except Exception as e:
            print(f"AI Düzeltme Hatası: {e}")
            return metin