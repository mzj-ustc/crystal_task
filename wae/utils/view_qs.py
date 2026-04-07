#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

datas = np.load('qs.npy')
#datas = np.load('proto_qs.npy')
print(datas.shape)

idx = 15
print(datas[idx, :, 4])

counts, bins = np.histogram(datas[idx, :, 4], bins=50)

plt.figure()
plt.stairs(counts, bins)
plt.show()
