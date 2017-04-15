import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

# Learning curve plots for RankLib

epochs = list(range(21))[1:]

l1_n5_lr_5 = [0.3062, 0.3078, 0.3095, 0.3107, 0.3048, 0.2814, 0.2679, 0.2633, 0.2562, 0.2504,
              0.248,  0.1654, 0.155,  0.152,  0.1515, 0.1511, 0.1494, 0.1471, 0.1449, 0.1437]

l1_n10_lr_5 = [0.3087, 0.3145, 0.3168, 0.3193, 0.318,  0.3162, 0.2934, 0.2684, 0.2516, 0.2379,
               0.2332, 0.2284, 0.2247, 0.2224, 0.2209, 0.22,   0.2194, 0.2183, 0.2173, 0.2165]

l1_n20_lr_5 = [0.3091, 0.3199, 0.3302, 0.3429, 0.354,  0.3606, 0.3627, 0.3661, 0.3671, 0.365,
               0.3609, 0.3569, 0.3535, 0.3491, 0.3451, 0.3457, 0.3479, 0.3498, 0.3508, 0.3513]

l1_n40_lr_5 = [0.3125, 0.3246, 0.3381, 0.3551, 0.3674, 0.3676, 0.3483, 0.3313, 0.331,  0.3482,
               0.3502, 0.3461, 0.3386, 0.3333, 0.3298, 0.3289, 0.3316, 0.337,  0.3414, 0.344]

l2_n5_lr_5 = [0.2586, 0.2685, 0.2774, 0.2849, 0.2935, 0.2997, 0.3049, 0.3079, 0.3093, 0.3101,
              0.3103, 0.3088, 0.3088, 0.309 , 0.3009, 0.2607, 0.235,  0.2202, 0.2124, 0.2087]

l2_n10_lr_5 = [0.2454, 0.2775, 0.2906, 0.2971, 0.3012, 0.3035, 0.305,  0.309,  0.3147, 0.3154,
               0.3155, 0.3158, 0.3159, 0.3143, 0.3035, 0.285,  0.2735, 0.2638, 0.2589, 0.2504]

l2_n20_lr_5 = [0.2817, 0.2954, 0.3024, 0.3066, 0.3083, 0.3102, 0.3097, 0.313,  0.3081, 0.2323,
               0.2091, 0.1968, 0.1917, 0.1895, 0.1886, 0.1882, 0.1876, 0.1872, 0.1869, 0.1869]


# Plot learning curves
fig = plt.figure()
plt.plot(epochs, l1_n5_lr_5, color="black", ls='-', label=r'5 nodes')
plt.plot(epochs, l1_n10_lr_5, color="blue", ls=':', label=r'10 nodes')
plt.plot(epochs, l1_n20_lr_5, color="red", ls='--', label=r'20 nodes')
plt.plot(epochs, l1_n40_lr_5, color="green", ls=':', label=r'40 nodes')
plt.xlabel(r"Training epochs")
plt.ylabel(r"nDCG@10")
plt.legend(loc='lower left')
plt.xticks(list(range(0, 21, 5)))
plt.xlim(1,20)
fig.savefig(parentdir + '/document/figures/' + 'RankNet_1_layer_training')
plt.close(fig)

# Plot learning curves
fig = plt.figure()
plt.plot(epochs, l2_n5_lr_5, color="black", ls='-', label=r'5 nodes')
plt.plot(epochs, l2_n10_lr_5, color="blue", ls=':', label=r'10 nodes')
plt.plot(epochs, l2_n20_lr_5, color="red", ls='--', label=r'20 nodes')
plt.xlabel(r"Training epochs")
plt.ylabel(r"nDCG@10")
plt.legend(loc='lower left')
plt.xticks(list(range(0, 21, 5)))
plt.xlim(1,20)
fig.savefig(parentdir + '/document/figures/' + 'RankNet_2_layer_training')
plt.close(fig)