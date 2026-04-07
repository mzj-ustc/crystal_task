"""
用 spglib 对称性分析做相图
输入: crystals_small.npy  (5500, 1024, 3)  笛卡尔坐标，单质，共用立方晶格
输出: spglib_gmm.pdf  /  spglib_gmm_ls.pdf  （无标题）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import NearestNeighbors
import spglib

# ============================================================
# CONFIG
# ============================================================
FQS    = './crystals_small.npy'   # (5500, 1024, 3)
FLJ    = './ljgp_params.npy'  # (5500, 2)  [epsilon, r0]
SYMPREC_LIST = [1e-1, 5e-2, 1e-2]    # 依次尝试，取第一个识别成功的
PREFIX = 'spglib'

CMAP_STR = [
    'blue', 'orange', 'green', 'red', 'purple', 'yellow', 'pink',
    'olive', 'cyan', 'darkblue', 'darkcyan', 'violet', 'darkred',
    'darkgreen', 'chocolate', 'brown', 'lime', 'dodgerblue', 'indigo',
    'peru', 'darkorange', 'magenta',
]
CMAP = ListedColormap(CMAP_STR)

# ============================================================
# 步骤1: 从笛卡尔坐标估算立方盒子边长
# ============================================================
def estimate_lattice(coords):
    """
    coords: (N, 3) 笛卡尔坐标（单个结构）
    返回: (3,3) 立方晶格矩阵
    策略: 盒子边长 = 坐标范围 + 最近邻距离（补周期性 buffer）
    """
    # 最近邻距离估算（取第2近邻，第1近邻是自己）
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    d_nn = np.median(dists[:, 1])   # 中位数，鲁棒

    ranges = coords.max(axis=0) - coords.min(axis=0)   # (3,)
    L = ranges + d_nn                                   # 每个方向加一个 buffer

    # 取三个方向的均值作为立方盒子边长（强制立方对称）
    L_cubic = np.mean(L)
    return np.diag([L_cubic, L_cubic, L_cubic])

# ============================================================
# 步骤2: 笛卡尔 → 分数坐标
# ============================================================
def cart_to_frac(coords, lattice):
    """
    coords:  (N, 3) 笛卡尔
    lattice: (3, 3)
    返回:    (N, 3) 分数坐标，clip 到 [0, 1)
    """
    frac = coords @ np.linalg.inv(lattice)
    frac = frac - np.floor(frac)          # 折叠到 [0,1)
    return frac

# ============================================================
# 步骤3: spglib 识别空间群
# ============================================================
def get_spacegroup_number(lattice, frac_pos, numbers, symprec_list):
    """
    依次尝试不同 symprec，返回第一个成功识别的空间群编号。
    全部失败则返回 1（P1，无序）。
    """
    for symprec in symprec_list:
        cell = (lattice, frac_pos, numbers)
        sg_str = spglib.get_spacegroup(cell, symprec=symprec)
        if sg_str is not None:
            # 格式: "Im-3m (229)"，提取括号内编号
            sg_num = int(sg_str.split('(')[1].rstrip(')'))
            return sg_num
    return 1   # 无法识别，归为 P1

# ============================================================
# 步骤4: 排序标签（按出现频率从高到低重新编号）
# ============================================================
def sort_labels(labels, labels_ls):
    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts)]
    lut   = np.zeros(int(labels.max()) + 1, dtype=int)
    for new_id, old_id in enumerate(order):
        lut[old_id] = new_id
    return lut[labels], lut[labels_ls]

# ============================================================
# 步骤5: 保存单张 PDF（无标题）
# ============================================================

def save_scatter_pdf(path, rs, epsilons, colors, vmax):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rs, epsilons, c=colors, s=30, marker='s', cmap=CMAP,
               vmin=0, vmax=vmax)
    ax.set_xlim([1.0, 2.1])
    ax.set_ylim([0.1, 5.0])
    ax.set_xlabel('$r_0$', size=16)
    ax.set_ylabel('$\\epsilon$', size=16)

    # ── 右上角 (c) 标签 ──────────────────────────────────────────
    ax.text(0.03, 0.97, '(b)',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=14,
        bbox=dict(facecolor='white', edgecolor='none',
                  pad=2))
    # ────────────────────────────────────────────────────────────

    plt.tight_layout()
    plt.savefig(path, format='pdf')
    plt.close(fig)
    print(f"  Saved -> {path}")

# ============================================================
# 主流程
# ============================================================
print("Loading data ...")
coords_all    = np.load(FQS)          # (5500, 1024, 3)
lj_parameters = np.load(FLJ)         # (5500, 2)
epsilons      = lj_parameters[:, 0]
rs            = lj_parameters[:, 1]
N_struct      = coords_all.shape[0]
N_atoms       = coords_all.shape[1]
atom_numbers  = np.zeros(N_atoms, dtype=int)   # 单质，全部标0

# 用第一个结构估算共用晶格（假设所有结构盒子尺寸相同）
print("Estimating lattice from first structure ...")
lattice = estimate_lattice(coords_all[0])
print(f"  Estimated cubic lattice: L = {lattice[0,0]:.4f}")

# 逐结构识别空间群
print(f"Running spglib on {N_struct} structures ...")
sg_labels = np.zeros(N_struct, dtype=int)
failed    = 0
for i in range(N_struct):
    frac = cart_to_frac(coords_all[i], lattice)
    sg_labels[i] = get_spacegroup_number(lattice, frac, atom_numbers,
                                          SYMPREC_LIST)
    if sg_labels[i] == 1:
        failed += 1
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N_struct} done  (P1 so far: {failed})")

unique_sg, counts = np.unique(sg_labels, return_counts=True)
print(f"\nSpace groups found: {len(unique_sg)}")
for sg, cnt in sorted(zip(unique_sg, counts), key=lambda x: -x[1]):
    print(f"  SG {sg:4d}  count={cnt}")

# Label Spreading（在 LJ 参数空间中平滑）
print("\nLabel Spreading ...")
ls = LabelSpreading('knn', n_neighbors=30, alpha=0.1)
ls.fit(lj_parameters, sg_labels)
sg_labels_ls = ls.predict(lj_parameters)

# 重新编号（按频率排序，方便 colormap 连续着色）
sg_labels, sg_labels_ls = sort_labels(sg_labels, sg_labels_ls)
vmax = len(CMAP_STR) - 1

# 保存相图
save_scatter_pdf(f"{PREFIX}_gmm.pdf",    rs, epsilons, sg_labels,    vmax)
save_scatter_pdf(f"{PREFIX}_gmm_ls.pdf", rs, epsilons, sg_labels_ls, vmax)

# 保存原始结果（空间群编号 + LJ 参数）供后续分析
out = np.column_stack([lj_parameters, sg_labels, sg_labels_ls])
np.savetxt(f"{PREFIX}.out", out,
           fmt='%10.5f %10.5f %d %d',
           header='epsilon  r0  sg_sorted  sg_ls_sorted')
print(f"  Saved -> {PREFIX}.out")
print("\nAll done.")