"""
TR-Docuquery — Çapraz Belge Denetçisi demosu

Daha önce yüklenmiş iki belgeyi karşılaştırır; değişen maddeleri,
tutarsızlıkları, eksik bölümleri ve risk skorunu yazdırır.

Kullanım:
    python karsilastir.py "belge1.pdf" "belge2.pdf"
"""
import os
import sys
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def liste_yaz(baslik, ogeler):
    print(f"\n{baslik}")
    if not ogeler:
        print("  (yok)")
    for o in ogeler:
        print(f"  • {o}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    belge1 = os.path.basename(sys.argv[1])
    belge2 = os.path.basename(sys.argv[2])
    print(f"\n🔍 Karşılaştırılıyor:\n  1) {belge1}\n  2) {belge2}\n")

    r = requests.post(
        f"{API_URL}/agent/audit-documents",
        json={"filenames": [belge1, belge2]},
        timeout=180,
    )
    veri = r.json()
    if r.status_code != 200 or not veri.get("success"):
        print(f"❌ Hata (HTTP {r.status_code}): {veri}")
        sys.exit(1)

    sonuc = veri.get("comparison_results", {})
    print(f"⚠️  RİSK SKORU: {sonuc.get('risk_score', '?')} / 10")
    liste_yaz("📝 Değişen maddeler:", sonuc.get("changed_clauses"))
    liste_yaz("💰 Tutarsızlıklar (fiyat / oran / tarih):", sonuc.get("discrepancies"))
    liste_yaz("🕳️  Eksik bölümler:", sonuc.get("missing_sections"))
    print(f"\n📋 YÖNETİCİ ÖZETİ:\n{veri.get('executive_summary', '')}\n")


if __name__ == "__main__":
    main()
    