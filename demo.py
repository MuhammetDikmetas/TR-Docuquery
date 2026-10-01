"""
TR-Docuquery — Komut satırı demo istemcisi

Kullanım:
    python demo.py "documents/başvuru formu .pdf"
    python demo.py "documents/başvuru formu .pdf" "Bu formda hangi bilgiler isteniyor?"

1) PDF'i API'ye yükler (OCR + madde bazlı chunking + ChromaDB)
2) Soru verilirse belgeye sorar ve kaynaklı cevabı yazdırır
3) Soru verilmezse terminalden soru sormanızı bekler (çıkmak için boş Enter)
"""
import os
import sys
import json
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def belge_yukle(pdf_yolu: str) -> str:
    dosya_adi = os.path.basename(pdf_yolu)
    print(f"\n📤 Yükleniyor: {dosya_adi} ...")
    with open(pdf_yolu, "rb") as f:
        r = requests.post(
            f"{API_URL}/yukle",
            files=[("files", (dosya_adi, f, "application/pdf"))],
            timeout=300,
        )
    veri = r.json()
    if r.status_code != 200 or not veri.get("success"):
        print(f"❌ Yükleme başarısız (HTTP {r.status_code}):")
        print(json.dumps(veri, ensure_ascii=False, indent=2))
        sys.exit(1)

    print(f"✅ Yüklendi — {veri['chunk_sayisi']} parça oluşturuldu.")
    eksik = veri.get("missing_info") or []
    if eksik:
        print(f"⚠️  Eksik bilgi ajanının tespiti: {', '.join(eksik)}")
    return dosya_adi


def soru_sor(soru: str, dosya_adi: str) -> None:
    print(f"\n❓ {soru}")
    r = requests.post(
        f"{API_URL}/sor",
        json={"soru": soru, "dosya_adi": dosya_adi, "top_k": 5},
        timeout=120,
    )
    veri = r.json()
    if r.status_code != 200:
        print(f"❌ Hata (HTTP {r.status_code}): {veri}")
        return

    print(f"\n💡 CEVAP:\n{veri.get('cevap', '')}")

    kaynaklar = veri.get("kaynaklar") or []
    if kaynaklar:
        print("\n📚 KAYNAKLAR:")
        for i, k in enumerate(kaynaklar, start=1):
            madde = k.get("type")
            madde_txt = f" · Madde {madde}" if madde and madde not in ("genel", "audit") else ""
            print(f"  [{i}] {k.get('source')} — Sayfa {k.get('page')}{madde_txt} (benzerlik: {k.get('score')})")

    if veri.get("missing_info"):
        print(f"\n⚠️  Eksik alanlar: {', '.join(veri['missing_info'])}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_yolu = sys.argv[1]
    if not os.path.exists(pdf_yolu):
        print(f"❌ Dosya bulunamadı: {pdf_yolu}")
        sys.exit(1)

    dosya_adi = belge_yukle(pdf_yolu)

    if len(sys.argv) >= 3:
        soru_sor(sys.argv[2], dosya_adi)
        return

    print("\nBelgeye soru sorabilirsiniz (çıkmak için boş Enter).")
    while True:
        soru = input("\nSoru: ").strip()
        if not soru:
            break
        soru_sor(soru, dosya_adi)


if __name__ == "__main__":
    main()