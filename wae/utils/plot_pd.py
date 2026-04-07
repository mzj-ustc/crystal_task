#!/usr/bin/env python3

import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap

parser = OptionParser()
parser.add_option("--output", dest="output", default=None,
                  help="Specify an output from evaluation.")

(options, args) = parser.parse_args()
fout = options.output

# data format in fout
# [epsilon, r, latent parameters, labels, labels_LS]
datas = np.loadtxt(fout)
epsilons, rs = datas[:, 0], datas[:, 1]
latents = datas[:, 2:-2]
z1, z2, z3 = latents[:, 0], latents[:, 1], latents[:, 2]
labels = datas[:, -2].astype('int')
labels_LS = datas[:, -1].astype('int')

nbin = int(np.max(labels))
eps = 0.00001
hist, bin_edges = np.histogram(labels, bins=nbin+1, range=(-eps, nbin-eps+1))
indices = np.argsort(-hist)
#print(hist)
#print(indices)
l = len(indices)
ihash = np.zeros(l)
for i, x in enumerate(indices):
    ihash[x] = i

labels = np.array([ihash[label] for label in labels])
labels_LS = np.array([ihash[label] for label in labels_LS])

cmap_str = ['blue', 'orange', 'green', 'red', 'purple', 'yellow', 'pink', 'olive', 'cyan',
            'darkblue', 'darkcyan', 'violet', 'darkred', 'darkgreen', 'chocolate', 'brown',
            'lime', 'dodgerblue', 'indigo', 'peru', 'darkorange', 'magenta']
cmap = ListedColormap(cmap_str)

fig, (axs1, axs2) = plt.subplots(1, 2)

axs1.scatter(rs, epsilons, c=labels, s = 30, marker='s', cmap=cmap)
axs1.set_xlim([1.0, 2.1])
axs1.set_ylim([0.1, 5.0])
axs1.set_xlabel('$r_0$', size=16)
axs1.set_ylabel('$\epsilon$', size=16)
axs1.set_title('Predicted phase diagram')
#plt.savefig('phase_diagram_cn1.eps')

axs2.scatter(rs, epsilons, c=labels_LS, s = 30, marker='s', cmap=cmap)
axs2.set_xlim([1.0, 2.1])
axs2.set_ylim([0.1, 5.0])
axs2.set_xlabel('$r_0$', size=16)
axs2.set_ylabel('$\epsilon$', size=16)
axs2.set_title('Predicted phase diagram with LS')

#plt.savefig('phase_diagram_cn1.eps')

plt.show()
