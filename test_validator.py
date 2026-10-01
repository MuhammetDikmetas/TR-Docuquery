from dotenv import load_dotenv
load_dotenv(override=True)

from agent.missing_info_agent import validator_app

# Gerçekçi bir dilekçe: TARİH ve TC KİMLİK NO bilerek yazılmadı.
# Metinde ipucu yok; ajanın eksikleri kendisi bulması beklenir.
MOCK_TEXT = (
    "HARRAN ÜNİVERSİTESİ REKTÖRLÜĞÜNE\n\n"
    "Üniversitenizin Mühendislik Fakültesi Bilgisayar Mühendisliği bölümünden mezun oldum. "
    "Bu yıl düzenlenecek mezuniyet törenine katılmak istiyorum. "
    "Gereğinin yapılmasını arz ederim.\n\n"
    "Ad Soyad: Merve Aişeoğlu\n"
    "Adres: Haliliye / Şanlıurfa\n"
    "İmza: M.A."
)

BEKLENEN_EKSIKLER = {
    "tarih": ["tarih", "date"],
    "tc_no": ["tc", "kimlik"],
}


def main():
    initial_state = {
        "extracted_text": MOCK_TEXT,
        "validation_results": {},
        "error": None,
    }

    print("Eksik Bilgi Ajanı çalıştırılıyor...\n")
    print(f"Girdi Metni:\n{MOCK_TEXT}\n")

    try:
        final_state = validator_app.invoke(initial_state)

        if final_state.get("error"):
            print(f"❌ Ajan hata döndürdü: {final_state['error']}")
            return

        results = final_state.get("validation_results", {})
        missing = results.get("missing_fields", [])

        print("=== DOĞRULAMA SONUÇLARI ===")
        print(f"Belge Türü     : {results.get('document_type')}")
        print(f"Tam mı?        : {results.get('is_complete')}")
        print(f"Eksik Alanlar  : {', '.join(missing) if missing else 'Yok'}")

        # Beklenen eksiklerin bulunup bulunmadığını kontrol et
        print("\n=== TEST ===")
        eksik_metni = " ".join(missing).lower()
        basarili = True
        for alan, anahtarlar in BEKLENEN_EKSIKLER.items():
            bulundu = any(a in eksik_metni for a in anahtarlar)
            basarili &= bulundu
            print(f"{'✅' if bulundu else '❌'} '{alan}' eksik olarak tespit edildi mi: {bulundu}")

        print(f"\nSONUÇ: {'GEÇTİ ✅' if basarili else 'KALDI ❌'}")

    except Exception as e:
        print(f"Graf çalıştırılamadı: {e}")


if __name__ == "__main__":
    main()