import os
import json
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
url = f"{API_URL}/agent/audit-documents"

payload = {
    "doc1_text": (
        "MADDE 1 - HİZMET BEDELİ: Aylık hizmet bedeli 500 USD olup her ayın 1'inde ödenir. "
        "MADDE 2 - FESİH: Sözleşmenin feshi için 30 gün önceden yazılı bildirim gerekir."
    ),
    "doc2_text": (
        "MADDE 1 - HİZMET BEDELİ: Aylık hizmet bedeli 750 USD olup her ayın 15'inde ödenir. "
        "MADDE 2 - FESİH: Fesih bildirimi hakkında bir hüküm bulunmamaktadır."
    ),
}

print(f"Cross-Document Auditor API'ye istek gönderiliyor ({url})...")
try:
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        print("\nHata: API durum kodu", response.status_code)
        print("Yanıt:", response.text)
    else:
        data = response.json()

        if data.get("success"):
            print("\n=== KARŞILAŞTIRMA SONUÇLARI ===")
            print(json.dumps(data.get("comparison_results"), indent=2, ensure_ascii=False))

            print("\n=== YÖNETİCİ ÖZETİ ===")
            print(data.get("executive_summary"))
        else:
            print("\nHata: API success=False döndü")
            print(data)

except requests.exceptions.RequestException as e:
    print(f"\nİstek başarısız: {e}")