import React from 'react';
import { AlertTriangle, CheckCircle, FileSearch } from 'lucide-react';

interface MissingInfoPanelProps {
  missingFields: string[];
  isComplete: boolean;
  /** Oturumda analiz edilmiş bir belge var mı? Yoksa "doğrulandı" yerine boş durum gösterilir. */
  hasDocument?: boolean;
}

// Sık gelen alan anahtarlarının okunabilir karşılıkları
const fieldLabels: Record<string, string> = {
  tc_no: 'TC Kimlik No',
  tc_kimlik_no: 'TC Kimlik No',
  iban: 'IBAN',
  eposta: 'E-posta Adresi',
  e_posta: 'E-posta Adresi',
  ad_soyad: 'Ad Soyad',
  telefon: 'Telefon Numarası',
  tutar: 'Tutar Bilgisi',
  tarih: 'Tarih',
  imza: 'İmza',
  adres: 'Adres',
  vergi_no: 'Vergi No',
  tc_no_veya_vergi_no: 'TC / Vergi No',
  taraf_bilgileri: 'Taraf Bilgileri',
  taraflar_kimlik_bilgileri: 'Tarafların Kimlik Bilgileri',
  sozlesme_tarihi: 'Sözleşme Tarihi',
  sozlesme_bedeli: 'Sözleşme Bedeli',
  sozlesme_suresi: 'Sözleşme Süresi',
  odeme_kosullari: 'Ödeme Koşulları',
  teslim_tarihi: 'Teslim Tarihi',
  fesih_kosullari: 'Fesih Koşulları',
  yetkili_mahkeme: 'Yetkili Mahkeme',
  kdv_bilgisi: 'KDV Bilgisi',
  kdv_durumu: 'KDV Durumu',
  hazirlayan_kurum: 'Hazırlayan Kurum',
};

// Kısaltmalar büyük harfle yazılır
const kisaltmalar: Record<string, string> = { tc: 'TC', iban: 'IBAN', kdv: 'KDV', no: 'No', eposta: 'E-posta' };

// "vergi_no" -> "Vergi No", "IBAN Bilgisi" -> "IBAN Bilgisi"
function okunabilirAd(alan: string): string {
  const anahtar = alan.trim().toLowerCase();
  if (fieldLabels[anahtar]) return fieldLabels[anahtar];
  if (!alan.includes('_')) return alan.trim();
  return anahtar
    .split('_')
    .filter(Boolean)
    .map(k => kisaltmalar[k] ?? k.charAt(0).toLocaleUpperCase('tr-TR') + k.slice(1))
    .join(' ');
}

export const MissingInfoPanel: React.FC<MissingInfoPanelProps> = ({ missingFields, isComplete, hasDocument = true }) => {
  // Okunabilir ada çevir, "Eksik" tekrarını kaldır, tekrarlayanları temizle
  const uniqueFields = Array.from(new Set(
    missingFields.map(f => okunabilirAd(f).replace(/\s*eksik\s*$/i, ''))
  ));
  const hasMissing = !isComplete && uniqueFields.length > 0;

  // Henüz analiz edilmiş belge yoksa
  if (!hasDocument && !hasMissing) {
    return (
      <div className="flex flex-col items-center justify-center py-12 px-4 text-center gap-4 bg-slate-50 dark:bg-slate-800/20 rounded-3xl border border-dashed border-slate-200 dark:border-slate-700 animate-in fade-in duration-300">
        <div className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center">
          <FileSearch className="w-8 h-8 text-slate-400" />
        </div>
        <div className="space-y-1">
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200">Henüz Belge Yok</h3>
          <p className="text-xs text-slate-500 max-w-[220px]">Bir belge yüklediğinizde eksik alanlar (TC No, imza, tarih vb.) burada listelenecektir.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-right-4 duration-500">
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shadow-sm transition-colors duration-500 ${hasMissing ? 'bg-amber-100 dark:bg-amber-500/20' : 'bg-emerald-100 dark:bg-emerald-500/20'}`}>
          {hasMissing ? (
            <AlertTriangle className="w-5 h-5 text-amber-500 animate-pulse" />
          ) : (
            <CheckCircle className="w-5 h-5 text-emerald-500" />
          )}
        </div>
        <div>
          <h2 className="text-sm font-bold text-slate-800 dark:text-slate-200">
            {hasMissing ? 'Belge Eksiklikleri' : 'Doğrulama Tamamlandı'}
          </h2>
          <p className="text-[11px] text-slate-500">
            {hasMissing ? `${uniqueFields.length} adet eksik tespit edildi` : 'Tüm kriterler sağlandı'}
          </p>
        </div>
      </div>

      {hasMissing ? (
        <div className="space-y-3">
          {uniqueFields.map((field, idx) => (
            <div
              key={field + idx}
              className="flex items-center gap-3 p-3 bg-amber-50/50 dark:bg-amber-500/5 border border-amber-100 dark:border-amber-500/20 rounded-xl animate-in slide-in-from-bottom-2 duration-300 group hover:border-amber-300 dark:hover:border-amber-500/40 transition-all"
              style={{ animationDelay: `${idx * 100}ms` }}
            >
              <div className="w-6 h-6 rounded-full bg-amber-100 dark:bg-amber-500/20 flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
              </div>
              <span className="text-sm font-medium text-amber-900 dark:text-amber-200">
                {field} <span className="font-normal text-amber-700/80 dark:text-amber-300/70">eksik</span>
              </span>
            </div>
          ))}

          <div className="mt-4 p-4 bg-slate-100 dark:bg-[#1a1d24] rounded-2xl border border-slate-200 dark:border-slate-800">
            <p className="text-[11px] text-slate-500 italic text-center">
              Bu alanlar yapay zeka tarafından belge türüne göre tespit edilmiştir. Lütfen belgeyi kontrol ediniz.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-12 px-4 text-center gap-4 bg-emerald-50/30 dark:bg-emerald-500/5 rounded-3xl border border-emerald-100 dark:border-emerald-500/10">
          <div className="w-20 h-20 bg-emerald-100 dark:bg-emerald-500/20 rounded-full flex items-center justify-center">
            <CheckCircle className="w-10 h-10 text-emerald-500" />
          </div>
          <div className="space-y-2">
            <h3 className="text-base font-bold text-emerald-600 dark:text-emerald-400">Belge Başarıyla Doğrulandı</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400">Belgede eksik bilgi bulunmamaktadır.</p>
          </div>
        </div>
      )}
    </div>
  );
};