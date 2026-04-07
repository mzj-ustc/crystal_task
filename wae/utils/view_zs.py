#!/usr/bin/env python3

import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap
from PIL import Image

def phase_cns():
    '''
    Predict phase diagram from coordinate numbers.
    '''

    # data directory 
    dname = '/data5/store1/hy/MD/lammps/LJ_Gaussian_Potential/sim/relax/'
    fname = dname + 'coord_number.dat'
    with open(fname, 'r') as fid:
        lines = fid.readlines()

    cns = []
    for lino, line in enumerate(lines):
        words = line.split()
        struct_id = int(words[0])
        cn1, cn2, cn3 = int(words[1]), int(words[2]), int(words[3])

        cns.append(' '.join([str(cn1), str(cn2), str(cn3)]))

    cns_0 = cns.copy()
    cns.sort()
    label = 0

    last_name = cns[0]
    labels = {}
    labels[cns[0]] = 0
    for name in cns:
        if not name == last_name:
            label += 1
            labels[name] = label
            last_name = name

    colors = []
    for name in cns_0:
        colors.append(labels[name])
    colors = np.array(colors) % 28 # confine colors within [0, 28)

    params = np.load(dname + 'ljgp_params.npy')


    nbin = int(np.max(colors))
    eps = 0.00001
    hist, bin_edges = np.histogram(colors, bins=nbin+1, range=(-eps, nbin-eps+1))
    indices = np.argsort(-hist)
    #print(hist)
    #print(indices)
    l = len(indices)
    ihash = np.zeros(l)
    for i, x in enumerate(indices):
        ihash[x] = i
        
    colors = np.array([ihash[label] for label in colors])

    return params, colors

    
def find_max(labels, wi, li):

    counts = np.zeros(100)
    
    for wx in np.arange(wi-1, wi+2):
        for lx in np.arange(li-1, li+2):
            counts[int(labels[wx, lx])] += 1
    return np.argmax(counts)
    
def smooth_labels(labels):
    w, l = labels.shape

    labels_smooth = np.copy(labels)
    
    for wi in range(1, w-1):
        for li in range(1, l-1):
            labels_smooth[wi, li] = find_max(labels, wi, li)
            
    return np.ravel(labels_smooth)

def load_cmap():
    fcolor = '/data5/store1/hy/MD/lammps/LJ_Gaussian_Potential/models/wae/utils/COLORMAP'
    with open(fcolor, 'r') as fid:
        lines = fid.readlines()
    colors = [line.strip() for line in lines]
    return colors

parser = OptionParser()
parser.add_option("--output", dest="output", default=None,
                  help="Specify an output from evaluation.")

(options, args) = parser.parse_args()
fout = options.output
prefix = fout.split('.')[0]
fproto = prefix + '.pparam'

with open('indices_table.txt', 'r') as fid:
    lines = fid.readlines()
    
lookup = []    
for lino, line in enumerate(lines):
    lookup.append(line.split()[1].split('_')[1])
params_proto = np.loadtxt(fproto)
epsilons_proto, rs_proto = params_proto[:, 0], params_proto[:, 1]

# data format in fout
# [epsilon, r, latent parameters, labels, labels_LS]
datas = np.loadtxt(fout)
epsilons, rs = datas[:, 0], datas[:, 1]
latents = datas[:, 2:-2]
z1, z2, z3 = latents[:, 0], latents[:, 1], latents[:, 2]
labels = datas[:, -2].astype('int')
labels_LS = datas[:, -1].astype('int')

nbin = int(np.max(labels))+1
eps = 0.00001
hist, bin_edges = np.histogram(labels, bins=nbin, range=(-eps, nbin-eps))

indices = np.argsort(-hist)
#print(hist)
#print(indices)
l = len(indices)
ihash = np.zeros(l)
for i, x in enumerate(indices):
    ihash[x] = i

labels = np.array([ihash[label] for label in labels])
labels_LS = np.array([ihash[label] for label in labels_LS])
# epsilons = np.reshape(epsilons, (110, 50))
# rs = np.reshape(rs, (110, 50))
labels_mat = np.reshape(labels, (110, 50))
labels_smooth = smooth_labels(labels_mat)
labels_LS_mat = np.reshape(labels_LS, (110, 50))
labels_LS_smooth = smooth_labels(labels_LS_mat)

cmap_str = load_cmap()
ncolor = len(cmap_str)
if ncolor > nbin:
    ncolor = nbin
#print(cmap_str)
cmap = ListedColormap(cmap_str[:ncolor])

############################################################
# plot histogram
if False:
    fig, ax = plt.subplots()
    N, bins, patches = ax.hist(labels, bins=nbin, range=(-eps, nbin-eps))

    for i in range(len(patches)):
        patches[i].set_facecolor(cmap_str[i])
    plt.xlim([0, 20])
    plt.ylim([0, 2500])
    plt.xlabel('type ID', size=16)
    plt.ylabel('# of samples', size=16)


############################################################
# plot histogram
if False:
    fig = plt.figure()

    axs1 = fig.add_subplot(2, 2, 1)
    axs1.scatter(z1, z2, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)

    axs1 = fig.add_subplot(2, 2, 2)
    axs1.scatter(z1, z3, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    
    axs1 = fig.add_subplot(2, 2, 3)
    axs1.scatter(z2, z3, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    
    axs1 = fig.add_subplot(2, 2, 4, projection='3d')
    axs1.scatter(z1, z2, z3, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_zlim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    axs1.set_zlabel('$z_3$', size=16)
    axs1.set_title('Predicted phase diagram')
    #plt.savefig('phase_diagram_cn1.eps')


############################################################
# plot histogram
if False:
    fig = plt.figure()
    
    axs1 = fig.add_subplot(2, 2, 1)
    axs1.scatter(z1, z2, c=labels_LS, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    
    axs1 = fig.add_subplot(2, 2, 2)
    axs1.scatter(z1, z3, c=labels_LS, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    
    axs1 = fig.add_subplot(2, 2, 3)
    axs1.scatter(z2, z3, c=labels_LS, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    
    axs1 = fig.add_subplot(2, 2, 4, projection='3d')
    axs1.scatter(z1, z2, z3, c=labels_LS, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_zlim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    axs1.set_zlabel('$z_3$', size=16)
    axs1.set_title('Predicted phase diagram')
    #plt.savefig('phase_diagram_cn1.eps')
    

############################################################
# plot histogram
if False:
    fig = plt.figure()

    axs1 = fig.add_subplot(2, 2, 1)
    axs1.scatter(z1, z2, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    axs1.annotate(
        '(a)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    axs1 = fig.add_subplot(2, 2, 2)
    axs1.scatter(z1, z3, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_3$', size=16)
    axs1.annotate(
        '(b)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    axs1 = fig.add_subplot(2, 2, 3, projection='3d')
    a1 = np.abs(z1) < 25
    a2 = np.abs(z2) < 25
    a3 = np.abs(z3) < 25
    aa = a1 & a2 & a3
    axs1.scatter(z1[aa], z2[aa], z3[aa], c=labels[aa], s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-25, 25])
    axs1.set_ylim([-25, 25])
    axs1.set_zlim([-25, 25])
    axs1.set_xlabel('$z_1$', size=16)
    axs1.set_ylabel('$z_2$', size=16)
    axs1.set_zlabel('$z_3$', size=16)
    axs1.annotate(
        '(c)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))
    #plt.savefig('phase_diagram_cn1.eps')

    ax = fig.add_subplot(2, 2, 4)
    N, bins, patches = ax.hist(labels, bins=nbin, range=(-eps, nbin-eps))
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap_str[i])
        patches[i].set_label('BCC')

    plt.legend(ncol = 3, prop={'size': 14})
    plt.xlim([0, 25])
    plt.ylim([0, 2500])
    plt.xlabel('type ID', size=16)
    plt.ylabel('# of samples', size=16)
    ax.annotate(
        '(d)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=20, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

if True:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2,3)#,width_ratios=[1,2])
    
    ax = fig.add_subplot(gs[0,0])
    N, bins, patches = ax.hist(labels, bins=nbin, range=(-eps, nbin-eps))
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap_str[i])
        patches[i].set_label('$G_{' + str(i) + '}$')

    plt.legend(ncol = 4, columnspacing=0.8, frameon=False, prop={'size': 12})
    plt.xlim([0, nbin])
    plt.ylim([0, 800])
    plt.tick_params('x', labelsize=16)
    plt.tick_params('y', labelsize=16)
    plt.yticks([0, 200, 400, 600, 800], [0, 2, 4, 6, 8])
    plt.xlabel('type ID', size=20)
    plt.ylabel('# of samples [x100]', size=20)
    ax.annotate(
        '(a)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=24, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    ax = fig.add_subplot(gs[0,1])
    #labels=np.where(labels==6, 3, 0)
    ax.scatter(rs, epsilons, c=labels, s = 30, marker='s', cmap=cmap)
    #ax.scatter(rs_proto, epsilons_proto, s = 30, c = 'black', marker='o')
    ax.set_xlim([1.0, 2.1])
    ax.set_ylim([0.1, 5.0])
    ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('$r_0$', size=20)
    ax.set_ylabel('$\epsilon$', size=20)
    #ax.set_title('Predicted phase diagram')

    nproto = len(rs_proto)
    # for proto_id in range(nproto):
    #     ax.annotate(
    #         lookup[proto_id],
    #         xy=(rs_proto[proto_id], epsilons_proto[proto_id]), xycoords='data',
    #         xytext=(+0.0, -0.0), textcoords='offset fontsize',
    #         fontsize=10, verticalalignment='top', fontfamily='serif',
    #         bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    ax.annotate(
        '(b)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=24, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    ax = fig.add_subplot(gs[0,2])
    params, colors = phase_cns()
    ax.scatter(params[:,1], params[:,0], c=colors % ncolor, s = 30, marker='s', cmap=cmap)
    #ax.scatter(rs_proto, epsilons_proto, s = 30, c = 'black', marker='o')
    ax.set_xlim([1.0, 2.1])
    ax.set_ylim([0.1, 5.0])
    ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('$r_0$', size=20)
    ax.set_ylabel('$\epsilon$', size=20)
    #ax.set_title('Predicted phase diagram')

    nproto = len(rs_proto)
    # for proto_id in range(nproto):
    #     ax.annotate(
    #         lookup[proto_id],
    #         xy=(rs_proto[proto_id], epsilons_proto[proto_id]), xycoords='data',
    #         xytext=(+0.0, -0.0), textcoords='offset fontsize',
    #         fontsize=10, verticalalignment='top', fontfamily='serif',
    #         bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    ax.annotate(
        '(c)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=24, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    axs1 = fig.add_subplot(gs[1,0])
    axs1.scatter(z1, z3, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
    axs1.set_xlim([-20, 20])
    axs1.set_ylim([-20, 20])
    axs1.set_xticks([-20, -10, 0, 10, 20])
    axs1.set_yticks([-20, -10, 0, 10, 20])
    axs1.xaxis.set_tick_params(labelsize=16)
    axs1.yaxis.set_tick_params(labelsize=16)
    axs1.set_xlabel('$z_1$', size=20)
    axs1.set_ylabel('$z_3$', size=20)
    axs1.annotate(
        '(d)',
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=24, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))


    #ax = fig.add_subplot(gs[1,1:])
plt.savefig(prefix + '_pd.eps', dpi=150, bbox_inches='tight')


fig = plt.figure(figsize=(8,8))
axs2 = fig.add_subplot(projection='3d')
axs2.view_init(elev=30, azim=120, roll=0)
a1 = np.abs(z1) < 20
a2 = np.abs(z2) < 20
a3 = np.abs(z3) < 20
aa = a1 & a2 & a3
#axs2.scatter(z1[aa], z2[aa], z3[aa], c=labels[aa], s = 25, marker='o', alpha = 1, cmap=cmap)
axs2.scatter(z1, z2, z3, c=labels, s = 25, marker='o', alpha = 1, cmap=cmap)
axs2.set_xlim([-20, 20])
axs2.set_ylim([-20, 20])
axs2.set_zlim([-20, 20])
axs2.set_xticks([-20, -10, 0, 10, 20])
axs2.set_yticks([-20, -10, 0, 10, 20])
axs2.set_zticks([-20, -10, 0, 10, 20])
axs2.xaxis.set_tick_params(labelsize=16)
axs2.yaxis.set_tick_params(labelsize=16)
axs2.zaxis.set_tick_params(labelsize=16)

axs2.set_xlabel('$z_1$', size=20)
axs2.set_ylabel('$z_2$', size=20)
axs2.set_zlabel('$z_3$', size=20)
plt.savefig(prefix + '_latent.eps', dpi=150, bbox_inches='tight')
# axs2.annotate(
#     '(b)',
#     xy=(0, 1), xycoords='axes fraction',
#     xytext=(+0.5, -0.5), textcoords='offset fontsize',
#     fontsize=24, verticalalignment='top', fontfamily='serif',
#     bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

plt.show()
