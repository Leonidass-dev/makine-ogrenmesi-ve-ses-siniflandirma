# -*- coding: utf-8 -*-
"""
Mel-Spektrogram Öznitelik Çıkarımı ve Frekans Analizi
Ses sinyallerini insan kulağının algısına uygun Mel frekans ölçeğine ve desibel (dB) güç spektrumuna dönüştürür.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def mel_spektrogram_cikar(dosya_yolu, n_mels=128):
    TARGET_SR = 22050

    if not os.path.exists(dosya_yolu):
        print(f"[!] Dosya bulunamadı: {dosya_yolu}")
        print("[i] Test için sentetik çok frekanslı ses sinyali üretiliyor...")
        t = np.linspace(0, 2.0, TARGET_SR * 2, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        sr = TARGET_SR
    else:
        y, sr = librosa.load(dosya_yolu, sr=TARGET_SR)
        print(f"Yüklenen dosya: {os.path.basename(dosya_yolu)} | Örnekleme Hızı: {sr} Hz")

    # 1. Mel-Spektrogram Özniteliklerini Çıkarma
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    # Enerjiyi logaritmik (dB) ölçeğe dönüştürme
    mels_db = librosa.power_to_db(mels, ref=np.max)

    print("-" * 50)
    print(f"Mel-Spektrogram Matris Boyutu (Bant Sayısı x Zaman): {mels_db.shape}")
    print(f"Veri Tipi: {mels_db.dtype}")
    print(f"Min / Max dB Değeri: {mels_db.min():.2f} dB / {mels_db.max():.2f} dB")
    print("-" * 50)

    # Mel-Spektrogram Görselleştirme
    plt.figure(figsize=(11, 4.5))
    librosa.display.specshow(mels_db, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2, cmap='magma')
    plt.colorbar(format='%+2.0f dB', label='Güç / Enerji (dB)')
    plt.title(f'Mel-Spektrogram ({n_mels} Mel Bandı)', fontsize=13, fontweight='bold')
    plt.xlabel('Zaman (saniye)')
    plt.ylabel('Frekans (Hz / Mel)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mel-Spektrogram Analiz Aracı")
    parser.add_argument("dosya", nargs="?", default="sample.wav", help="İncelenecek .wav ses dosyası")
    parser.add_argument("--mels", type=int, default=128, help="Mel filtre bankası sayısı")
    args = parser.parse_args()
    mel_spektrogram_cikar(args.dosya, args.mels)
