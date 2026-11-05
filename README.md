# Makine Öğrenmesi ve Ses Sinyal İşleme Portföyü

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.1%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio_AI-orange?style=for-the-badge)](https://librosa.org/)

Bu depo; sıfırdan vektörize edilmiş makine öğrenmesi algoritmalarını (from-scratch k-Means ve sınıflandırıcılar) ve ses sinyali işleme (akustik analiz, Mel-spektrogram ve UrbanSound8K sınıflandırması) çalışmalarını bir araya getiren kapsamlı bir uygulamalı yapay zeka portföyüdür.

---

## Proje Modülleri ve Kapsam

### 1. Sıfırdan Makine Öğrenmesi Algoritmaları (From Scratch)
Hazır kütüphane fonksiyonlarına bağımlı kalmadan, lineer cebir ve olasılık temelleriyle geliştirilen modeller:
- **Vektörize K-Means Kümeleme:** İkili döngüler yerine matris çarpımıyla optimize edilmiş $O(n \cdot k)$ mesafesi matrisi ($X^2 + C^2 - 2XC^T$).
- **Özelleştirilmiş Standardizasyon:** Veri sızıntısını (data leakage) engelleyen train/test ayrık z-score dönüşümü.
- **Model Değerlendirme:** Doğruluk (Accuracy), Karışıklık Matrisi ve Normalleştirilmiş Karşılıklı Bilgi Skoru (NMI - Normalized Mutual Information) hesaplamaları.
- **Veri Setleri:** Iris Çiçek Veri Seti ve UCI Wine Veri Seti.

### 2. Ses Sinyal İşleme ve Akustik Sınıflandırma
Ses dalgalarının zaman ve frekans uzayında analizi ile kentsel seslerin makine öğrenmesiyle tespiti:
- **Sinyal Analizi:** Örnekleme hızı (sample rate), kanal sayısı, genlik aralığı ve dalga formu (waveform) görselleştirme.
- **Mel-Frekans Spektrogramı:** İnsan kulağının duyum eşiğine dayalı Mel ölçeği filtre bankası ve desibel (dB) güç spektrumu dönüşümü.
- **UrbanSound8K Çok Sınıflı Sınıflandırma:** 10 farklı çevresel ses kategorisinde (korna, köpek havlaması, silah sesi vb.) öznitelik mühendisliği ve Random Forest & KNN modelleri ile sınıflandırma performansı analizi.

---

## Dizin Yapısı

```text
├── algoritmalar_ve_kumeleme/
│   ├── kmeans_ve_kumeleme_iris.py       # Sıfırdan k-Means algoritması ve Iris analizi
│   └── cok_sinifli_siniflandirma_wine.py# Wine veri seti üzerinde sınıflandırma ve metrikler
├── ses_sinyal_isleme/
│   ├── ses_dalga_formu_analizi.py       # Dalga formu (waveform) ve ses sinyali istatistikleri
│   ├── mel_spektrogram_analizi.py       # Mel filtreleme ve spektrogram güç analizi
│   └── urban_sound_ses_siniflandirici.py# UrbanSound8K öznitelik çıkarımı ve model kıyaslama
├── requirements.txt                     # Proje bağımlılıkları (librosa, sklearn, numpy vb.)
└── README.md
```

---

## Kurulum ve Çalıştırma

### 1. Bağımlılıkların Yüklenmesi
Sanal ortam oluşturup gerekli kütüphaneleri yükleyin:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Algoritmaların Çalıştırılması

**Sıfırdan K-Means Kümeleme:**
```bash
python algoritmalar_ve_kumeleme/kmeans_ve_kumeleme_iris.py
```

**Wine Veri Seti Sınıflandırması:**
```bash
python algoritmalar_ve_kumeleme/cok_sinifli_siniflandirma_wine.py
```

### 3. Ses Sinyal İşleme Modüllerinin Çalıştırılması

**Ses Dalga Formu Görselleştirme:**
```bash
# Örnek ses dosyası ile veya test moduyla çalıştırma:
python ses_sinyal_isleme/ses_dalga_formu_analizi.py [ses_dosyasi.wav]
```

**Mel-Spektrogram Analizi:**
```bash
python ses_sinyal_isleme/mel_spektrogram_analizi.py [ses_dosyasi.wav] --mels 128
```

**UrbanSound8K Sınıflandırıcı Eğitimi:**
```bash
python ses_sinyal_isleme/urban_sound_ses_siniflandirici.py
```
*(UrbanSound8K veri seti kullanıldığında veri yolunu `URBANSOUND_PATH` ortam değişkeniyle belirtebilirsiniz).*
