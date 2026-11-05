import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score
# =========================================================
# 0) Yardımcılar: uzaklık ve standardizasyon
# =========================================================
def fit_standardizer(Xtr):
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma

def transform_standardizer(Xin, mu, sigma):
    return (Xin - mu) / sigma
def pairwise_sq_dists(X, C):
    X2 = np.sum(X**2, axis=1, keepdims=True)     # (n,1)
    C2 = np.sum(C**2, axis=1, keepdims=True).T   # (1,k)
    return X2 + C2 - 2 * (X @ C.T)
# =========================================================
# 1) K-Means
# =========================================================
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
                # boş cluster olursa: rastgele bir noktaya re-init
                new_centers[j] = X[rng.integers(0, n)]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels = new_labels

        if shift < tol:
            break

    inertia = np.sum(np.min(pairwise_sq_dists(X, centers), axis=1))
    return labels, centers, inertia

def assign_to_centers(X, centers):
    d2 = pairwise_sq_dists(X, centers)
    return np.argmin(d2, axis=1)
# =========================================================
# 2) Silhouette
# =========================================================
def my_silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    uniq = np.unique(labels)

    if len(uniq) <= 1 or len(uniq) == n:
        return 0.0

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
# =========================================================
# 3) ARI
# =========================================================
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
# =========================================================
# IRIS: Bölüm 1–4
# =========================================================
if __name__ == "__main__":
    print("#############################")
    print("### IRIS: BÖLÜM 1–4 ###")
    print("#############################\n")

    # Bölüm 1.1
    iris = load_iris()
    X = iris.data
    y = iris.target
    n, d = X.shape
    ktrue = len(np.unique(y))

    print("=== IRIS BÖLÜM 1 ===")
    print(f"Örnek sayısı (n): {n}")
    print(f"Öznitelik sayısı (d): {d}")
    print(f"Sınıf sayısı (ktrue): {ktrue}")

    # Bölüm 1.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Train/Test Boyutları ---")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # Bölüm 1.3
    mu, sigma = fit_standardizer(X_train)
    X_train_std = transform_standardizer(X_train, mu, sigma)
    X_test_std  = transform_standardizer(X_test,  mu, sigma)

    print("\n--- Standardizasyon Kontrolü (Train) ---")
    print("mean (tümü):", np.round(X_train_std.mean(axis=0), 3))
    print("std  (tümü):", np.round(X_train_std.std(axis=0, ddof=0), 3))

    # k aralığı
    ntrain = X_train.shape[0]
    kmin = 2
    kmax = ktrue + int(math.floor(math.log(ntrain)))
    ks = list(range(kmin, kmax + 1))
    print(f"\n--- k aralığı ---")
    print(f"ntrain={ntrain}, ktrue={ktrue} -> k: {kmin}..{kmax}")

    # Bölüm 2: Normalized train
    print("\n=== IRIS BÖLÜM 2 (Normalized Train) ===")
    inertias_norm, sils_norm = [], []
    for k in ks:
        labels, centers, inertia = my_kmeans(X_train_std, k, random_state=42)
        sil = my_silhouette_score(X_train_std, labels)
        inertias_norm.append(inertia)
        sils_norm.append(sil)
        print(f"k={k:2d} | inertia={inertia:10.4f} | silhouette={sil: .4f}")

    plt.figure()
    plt.plot(ks, inertias_norm, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("IRIS (Normalized Train) - Elbow (Inertia)")
    plt.show()

    plt.figure()
    plt.plot(ks, sils_norm, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("IRIS (Normalized Train) - Silhouette")
    plt.show()

    # Bölüm 3: Raw train
    print("\n=== IRIS BÖLÜM 3 (Raw Train) ===")
    inertias_raw, sils_raw = [], []
    for k in ks:
        labels, centers, inertia = my_kmeans(X_train, k, random_state=42)
        sil = my_silhouette_score(X_train, labels)
        inertias_raw.append(inertia)
        sils_raw.append(sil)
        print(f"k={k:2d} | inertia={inertia:12.4f} | silhouette={sil: .4f}")

    plt.figure()
    plt.plot(ks, inertias_raw, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("IRIS (Raw Train) - Elbow (Inertia)")
    plt.show()

    plt.figure()
    plt.plot(ks, sils_raw, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("IRIS (Raw Train) - Silhouette")
    plt.show()

    # Bölüm 4: ARI & NMI (Normalized, Train/Test)
    print("\n=== IRIS BÖLÜM 4 (ARI & NMI, Train/Test, Normalized) ===")
    ari_tr, ari_te = [], []
    nmi_tr, nmi_te = [], []

    for k in ks:
        yhat_tr, centers, _ = my_kmeans(X_train_std, k, random_state=42)
        yhat_te = assign_to_centers(X_test_std, centers)

        a_tr = my_ari(y_train, yhat_tr)
        a_te = my_ari(y_test, yhat_te)

        n_tr = normalized_mutual_info_score(y_train, yhat_tr)
        n_te = normalized_mutual_info_score(y_test, yhat_te)

        ari_tr.append(a_tr); ari_te.append(a_te)
        nmi_tr.append(n_tr); nmi_te.append(n_te)

        print(f"k={k:2d} | ARI(train)={a_tr: .4f} ARI(test)={a_te: .4f} | "
              f"NMI(train)={n_tr: .4f} NMI(test)={n_te: .4f}")

    plt.figure()
    plt.plot(ks, ari_tr, marker="o", label="ARI(train)")
    plt.plot(ks, ari_te, marker="o", label="ARI(test)")
    plt.xlabel("k")
    plt.ylabel("ARI")
    plt.title("IRIS (Normalized) - ARI Train/Test")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ks, nmi_tr, marker="o", label="NMI(train)")
    plt.plot(ks, nmi_te, marker="o", label="NMI(test)")
    plt.xlabel("k")
    plt.ylabel("NMI")
    plt.title("IRIS (Normalized) - NMI Train/Test")
    plt.legend()
    plt.show()
