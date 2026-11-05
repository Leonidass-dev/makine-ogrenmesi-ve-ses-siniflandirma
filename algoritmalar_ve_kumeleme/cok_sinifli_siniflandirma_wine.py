import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

# -----------------------------
# Bölüm 1.1 — Wine veri setini yükle
# -----------------------------
data = load_wine()
X = data.data
y = data.target

n_samples, n_features = X.shape
n_classes = len(np.unique(y))

print("=== BÖLÜM 1 (WINE) ===")
print(f"Örnek sayısı (n): {n_samples}")
print(f"Öznitelik sayısı (d): {n_features}")
print(f"Sınıf sayısı (ktrue): {n_classes}")

# -----------------------------
# Bölüm 1.2 — %80 train / %20 test ayır
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Boyutları ---")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# -----------------------------
# Bölüm 1.3 — Standardizasyon
# -----------------------------
def fit_standardizer(Xtr):
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)  # güvenlik
    return mu, sigma

def transform_standardizer(Xin, mu, sigma):
    return (Xin - mu) / sigma

mu, sigma = fit_standardizer(X_train)
X_train_std = transform_standardizer(X_train, mu, sigma)
X_test_std  = transform_standardizer(X_test,  mu, sigma)
# -----------------------------
# Beklenen sonuç kontrolü: train'de mean≈0, std≈1
# -----------------------------
train_mean = X_train_std.mean(axis=0)
train_std  = X_train_std.std(axis=0, ddof=0)

print("\n--- Standardizasyon Kontrolü (Train) ---")
print("mean :", np.round(train_mean, 3))
print("std  :", np.round(train_std, 3))
# -----------------------------
# Bölüm 2 — K aralığı
# -----------------------------
ntrain = X_train_std.shape[0]
ktrue = len(np.unique(y_train))

kmin = 2
kmax = ktrue + int(math.floor(math.log(ntrain)))  # ln
ks = list(range(kmin, kmax + 1))

print("=== BÖLÜM 2 (WINE - Normalized Train) ===")
print(f"ntrain={ntrain}, ktrue={ktrue} -> k aralığı: {kmin}..{kmax}")
# -----------------------------
# K-Means
# -----------------------------
def pairwise_sq_dists(X, C):
    X2 = np.sum(X**2, axis=1, keepdims=True)     # (n,1)
    C2 = np.sum(C**2, axis=1, keepdims=True).T   # (1,k)
    return X2 + C2 - 2 * (X @ C.T)

def my_kmeans(X, k, max_iter=300, tol=1e-6, random_state=42):
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if k > n:
        raise ValueError("k, örnek sayısından büyük olamaz.")

    init_idx = rng.choice(n, size=k, replace=False)
    centers = X[init_idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        d2 = pairwise_sq_dists(X, centers)
        new_labels = np.argmin(d2, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = (new_labels == j)
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)
            else:
                # boş cluster olursa: rastgele bir noktaya yeniden başlat
                new_centers[j] = X[rng.integers(0, n)]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break

    inertia = np.sum(np.min(pairwise_sq_dists(X, centers), axis=1))
    return labels, centers, inertia

def my_silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    uniq = np.unique(labels)

    if len(uniq) <= 1 or len(uniq) == n:
        return 0.0

    # mesafe matrisi (n x n)
    diff = X[:, None, :] - X[None, :, :]
    D = np.linalg.norm(diff, axis=2)

    s = np.zeros(n, dtype=float)

    for i in range(n):
        ci = labels[i]
        same = (labels == ci)

        # a(i)
        if np.sum(same) <= 1:
            a_i = 0.0
        else:
            a_i = np.sum(D[i, same]) / (np.sum(same) - 1)

        # b(i)
        b_i = np.inf
        for c in uniq:
            if c == ci:
                continue
            mask = (labels == c)
            if np.any(mask):
                b_i = min(b_i, np.mean(D[i, mask]))

        denom = max(a_i, b_i)
        s[i] = 0.0 if denom == 0 else (b_i - a_i) / denom

    return float(np.mean(s))
# -----------------------------
# k=2..kmax için inertia + silhouette
# -----------------------------
inertias = []
silhouettes = []

for k in ks:
    labels, centers, inertia = my_kmeans(X_train_std, k, random_state=42)
    sil = my_silhouette_score(X_train_std, labels)

    inertias.append(inertia)
    silhouettes.append(sil)

    print(f"k={k:2d} | inertia={inertia:10.4f} | silhouette={sil: .4f}")
# -----------------------------
# Grafikler
# -----------------------------
plt.figure()
plt.plot(ks, inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("WINE (Normalized Train) - Elbow (Inertia)")
plt.show()

plt.figure()
plt.plot(ks, silhouettes, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.title("WINE (Normalized Train) - Silhouette")
plt.show()
# -----------------------------
# Bölüm 3
# -----------------------------
print("=== BÖLÜM 3 (WINE - Raw Train) ===")

# k aralığı Bölüm 2 ile aynı
ntrain_raw = X_train.shape[0]
ktrue_raw = len(np.unique(y_train))
kmin = 2
kmax = ktrue_raw + int(math.floor(math.log(ntrain_raw)))
ks = list(range(kmin, kmax + 1))

print(f"ntrain={ntrain_raw}, ktrue={ktrue_raw} -> k aralığı: {kmin}..{kmax}")

inertias_raw = []
silhouettes_raw = []

for k in ks:
    labels, centers, inertia = my_kmeans(X_train, k, random_state=42)  # RAW train
    sil = my_silhouette_score(X_train, labels)

    inertias_raw.append(inertia)
    silhouettes_raw.append(sil)

    print(f"k={k:2d} | inertia={inertia:12.4f} | silhouette={sil: .4f}")

# Elbow (RAW)
plt.figure()
plt.plot(ks, inertias_raw, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("WINE (Raw Train) - Elbow (Inertia)")
plt.show()

# Silhouette (RAW)
plt.figure()
plt.plot(ks, silhouettes_raw, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.title("WINE (Raw Train) - Silhouette")
plt.show()

# -----------------------------
# Bölüm 3
# -----------------------------

print("=== BÖLÜM 4 (WINE - ARI & NMI, Train/Test, Normalized) ===")

ntrain = X_train_std.shape[0]
ktrue = len(np.unique(y_train))
kmin = 2
kmax = ktrue + int(math.floor(math.log(ntrain)))
ks = list(range(kmin, kmax + 1))

# -----------------------------
# Test ataması
# -----------------------------
def assign_to_centers(X, centers):
    d2 = pairwise_sq_dists(X, centers)
    return np.argmin(d2, axis=1)

# -----------------------------
# ARI
# -----------------------------
def comb2(x):
    return x * (x - 1) / 2

def my_ari(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_vals, true_inv = np.unique(y_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(y_pred, return_inverse=True)

    n_true = len(true_vals)
    n_pred = len(pred_vals)
    n = len(y_true)

    nij = np.zeros((n_true, n_pred), dtype=np.int64)
    for i in range(n):
        nij[true_inv[i], pred_inv[i]] += 1

    ai = nij.sum(axis=1)
    bj = nij.sum(axis=0)

    index = np.sum(comb2(nij))
    expected = (np.sum(comb2(ai)) * np.sum(comb2(bj))) / comb2(n)
    max_index = 0.5 * (np.sum(comb2(ai)) + np.sum(comb2(bj)))

    denom = (max_index - expected)
    if denom == 0:
        return 0.0
    return float((index - expected) / denom)

# -----------------------------
# k=2..kmax için ARI & NMI
# -----------------------------
ari_train, ari_test = [], []
nmi_train, nmi_test = [], []

for k in ks:
    yhat_train, centers, _ = my_kmeans(X_train_std, k, random_state=42)

    yhat_test = assign_to_centers(X_test_std, centers)

    # ARI (my_ari) + NMI (allowed metric)
    a_tr = my_ari(y_train, yhat_train)
    a_te = my_ari(y_test, yhat_test)

    n_tr = normalized_mutual_info_score(y_train, yhat_train)
    n_te = normalized_mutual_info_score(y_test, yhat_test)

    ari_train.append(a_tr)
    ari_test.append(a_te)
    nmi_train.append(n_tr)
    nmi_test.append(n_te)

    print(f"k={k:2d} | ARI(train)={a_tr: .4f} ARI(test)={a_te: .4f} | NMI(train)={n_tr: .4f} NMI(test)={n_te: .4f}")
# -----------------------------
# Grafikler
# -----------------------------
plt.figure()
plt.plot(ks, ari_train, marker="o", label="ARI(train)")
plt.plot(ks, ari_test, marker="o", label="ARI(test)")
plt.xlabel("k")
plt.ylabel("ARI")
plt.title("WINE (Normalized) - ARI Train/Test")
plt.legend()
plt.show()

plt.figure()
plt.plot(ks, nmi_train, marker="o", label="NMI(train)")
plt.plot(ks, nmi_test, marker="o", label="NMI(test)")
plt.xlabel("k")
plt.ylabel("NMI")
plt.title("WINE (Normalized) - NMI Train/Test")
plt.legend()
plt.show()