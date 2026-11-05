# -*- coding: utf-8 -*-
"""
Ses Sinyali Temel Analizi ve Dalga Formu (Waveform) Görselleştirme
Ses dosyalarının örnekleme hızı (sample rate), kanal sayısı, süre ve genlik dağılımını inceler.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def analiz_et(dosya_yolu):
    print(f"--- Ses Dosyası Analizi: {dosya_yolu} ---")
    
    if not os.path.exists(dosya_yolu):
        print(f"[!] Belirtilen dosya bulunamadı: {dosya_yolu}")
        print("[i] Test ve demonstrasyon için 1 saniyelik sentetik 440 Hz sinyal üretiliyor...")
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        data = 0.5 * np.sin(2 * np.pi * 440 * t)
    else:
        # sr=None orijinal örnekleme hızını korur
        data, sr = librosa.load(dosya_yolu, sr=None)

    sure_saniye = len(data) / sr
    kanal_sayisi = 1 if data.ndim == 1 else data.shape[0]

    print(f"1. Örnekleme Oranı (Sample Rate): {sr} Hz")
    print(f"2. Süre: {sure_saniye:.3f} saniye")
    print(f"3. Toplam Örnek Sayısı (Frame Count): {len(data)}")
    print(f"4. Kanal Sayısı: {kanal_sayisi} (Mono)")
    print(f"5. Veri Tipi: {data.dtype}")
    print(f"6. Min / Max Genlik: {np.min(data):.4f} / {np.max(data):.4f}")

    # Dalga formu çizimi
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sr, alpha=0.85, color='#2563eb')
    plt.title("Ses Sinyali Dalga Formu (Waveform)", fontsize=13, fontweight='bold')
    plt.xlabel("Zaman (saniye)")
    plt.ylabel("Genlik")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ses Sinyali Analiz Aracı")
    parser.add_argument("dosya", nargs="?", default="sample.wav", help="İncelenecek .wav ses dosyası")
    args = parser.parse_args()
    analiz_et(args.dosya)
