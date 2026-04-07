"""
对比实验: PCA / t-SNE / VAE  ×  两个数据集
  - 每种方法各自 GMM 聚类 + Label Spreading
  - 每种方法产生 2 张独立 PDF（无标题）
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================
DATA_FILES = {
    'qs':       'qs.npy',                    # (5500, 1024, 5)
    'crystals': './crystals_small.npy',  # shape TBD
}
FLJ    = './ljgp_params.npy'   # (5500, 2)
K      = 22       # GMM 聚类数
LATENT = 3      # 统一降维目标维度
BATCH  = 256      # VAE batch size
EPOCHS = 50       # VAE 训练轮数

CMAP_STR = [
    'blue', 'orange', 'green', 'red', 'purple', 'yellow', 'pink',
    'olive', 'cyan', 'darkblue', 'darkcyan', 'violet', 'darkred',
    'darkgreen', 'chocolate', 'brown', 'lime', 'dodgerblue', 'indigo',
    'peru', 'darkorange', 'magenta',
]
CMAP = ListedColormap(CMAP_STR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================
# 辅助：聚类 + Label Spreading
# ============================================================
def cluster_and_spread(enc, ljpq, K):
    gmm    = GaussianMixture(n_components=K, max_iter=300, tol=1e-4,
                              n_init=10, init_params='kmeans', random_state=42)
    labels = gmm.fit_predict(enc)
    ls     = LabelSpreading('knn', n_neighbors=30, alpha=0.1)
    ls.fit(ljpq, labels)
    labels_ls = ls.predict(ljpq)
    return labels, labels_ls


def sort_labels(labels, labels_ls):
    nbin = int(np.max(labels))
    eps  = 1e-5
    hist, _ = np.histogram(labels, bins=nbin + 1, range=(-eps, nbin + 1 - eps))
    order   = np.argsort(-hist)
    lut     = np.zeros(len(order), dtype=int)
    for i, x in enumerate(order):
        lut[x] = i
    return lut[labels], lut[labels_ls]


# ============================================================
# 辅助：保存单张 PDF（无标题）
# ============================================================
def save_scatter_pdf(path, rs, epsilons, colors):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rs, epsilons, c=colors, s=30, marker='s', cmap=CMAP,
               vmin=0, vmax=len(CMAP_STR) - 1)
    ax.set_xlim([1.0, 2.1])
    ax.set_ylim([0.1, 5.0])
    ax.set_xlabel('$r_0$', size=16)
    ax.set_ylabel('$\\epsilon$', size=16)
    plt.tight_layout()
    plt.savefig(path, format='pdf')
    plt.close(fig)
    print(f"  Saved → {path}")


# ============================================================
# 降维方法 1：PCA
# ============================================================
def reduce_pca(X, n_components=LATENT):
    print(f"  [PCA] StandardScaler + PCA({n_components}) ...")
    X_s  = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=n_components, random_state=42)
    enc  = pca.fit_transform(X_s)
    expl = pca.explained_variance_ratio_.sum() * 100
    print(f"  [PCA] Cumulative explained variance: {expl:.1f}%")
    return enc


# ============================================================
# 降维方法 2：t-SNE（sklearn barnes_hut，降到 3d）
# barnes_hut 最高支持 n_components=3，速度快且稳定。
# GMM 在 3d 空间上聚类，结果再投影到相图。
# ============================================================
def reduce_tsne(X, n_components=3):
    from sklearn.manifold import TSNE
    print(f"  [t-SNE] StandardScaler -> sklearn TSNE(barnes_hut, {n_components}d) ...")
    X_s = StandardScaler().fit_transform(X)
    enc = TSNE(n_components=n_components, perplexity=30, n_iter=1000,
               method='barnes_hut', random_state=42).fit_transform(X_s)
    return enc


# ============================================================
# 降维方法 3：VAE（PyTorch）
# ============================================================
class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        h = 512
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h),      nn.ReLU(),
        )
        self.mu_layer  = nn.Linear(h, latent_dim)
        self.var_layer = nn.Linear(h, latent_dim)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h), nn.ReLU(),
            nn.Linear(h, h),          nn.ReLU(),
            nn.Linear(h, in_dim),
        )

    def encode(self, x):
        h   = self.enc(x)
        mu  = self.mu_layer(h)
        lv  = self.var_layer(h)
        return mu, lv

    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, lv = self.encode(x)
        z      = self.reparameterize(mu, lv)
        recon  = self.dec(z)
        return recon, mu, lv


def vae_loss(recon, x, mu, lv):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kld        = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    return (recon_loss + kld) / x.size(0)


def reduce_vae(X, latent_dim=LATENT, epochs=EPOCHS, batch=BATCH):
    print(f"  [VAE] Training VAE (latent={latent_dim}, epochs={epochs}) ...")
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X).astype(np.float32)

    tensor  = torch.tensor(X_s).to(DEVICE)
    loader  = DataLoader(TensorDataset(tensor), batch_size=batch, shuffle=True)

    model   = VAE(X_s.shape[1], latent_dim).to(DEVICE)
    optim   = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for (xb,) in loader:
            optim.zero_grad()
            recon, mu, lv = model(xb)
            loss = vae_loss(recon, xb, mu, lv)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"    Epoch {ep:3d}/{epochs}  loss={total/len(X_s):.4f}")

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(tensor)
        enc   = mu.cpu().numpy()
    return enc


# ============================================================
# 主流程
# ============================================================
METHODS = {
    'pca':   reduce_pca,
    'vae':   reduce_vae,
    'tsne':  reduce_tsne,
}

lj_parameters = np.load(FLJ)           # (5500, 2)
epsilons = lj_parameters[:, 0]
rs       = lj_parameters[:, 1]

for ds_name, fqs in DATA_FILES.items():
    if not os.path.exists(fqs):
        print(f"\n[SKIP] {fqs} not found.")
        continue

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}  ({fqs})")
    qs = np.load(fqs)
    X  = qs.reshape(qs.shape[0], -1)   # flatten → (N, D)
    print(f"  Shape after flatten: {X.shape}")

    for method_name, reduce_fn in METHODS.items():
        print(f"\n--- Method: {method_name.upper()} ---")
        prefix = f"{ds_name}_{method_name}"

        # 1. 降维
        enc = reduce_fn(X)

        # 2. 聚类 + Label Spreading
        print(f"  Clustering (K={K}) + Label Spreading ...")
        labels, labels_ls = cluster_and_spread(enc, lj_parameters, K)
        labels, labels_ls = sort_labels(labels, labels_ls)

        # 3. 保存两张独立 PDF
        save_scatter_pdf(f"{prefix}_gmm.pdf",    rs, epsilons, labels)
        save_scatter_pdf(f"{prefix}_gmm_ls.pdf", rs, epsilons, labels_ls)

print("\nAll done.")