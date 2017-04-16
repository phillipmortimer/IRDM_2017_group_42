import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect
import numpy as np

# Learning curve plots for RankLib

iters =  np.linspace(0,250,26)
iters[0] = 1
# len(iters) = 26
# array([   1.,   10.,   20.,   30.,   40.,   50.,   60.,   70.,   80.,
        #  90.,  100.,  110.,  120.,  130.,  140.,  150.,  160.,  170.,
        # 180.,  190.,  200.,  210.,  220.,  230.,  240.,  250.])

tc64_lr01_l1 = [0.2910, 0.3097, 0.3097, 0.335,
                0.3523, 0.3719, 0.3837, 0.3864,
                0.3894, 0.3946, 0.3972, 0.3982,
                0.4006, 0.4027, 0.4043, 0.4052,
                0.4071, 0.4082, 0.4093, 0.41,
                0.4108, 0.4111, 0.4118, 0.4126,
                0.4132, 0.4142]

tc64_lr01_l10 = [0.3548, 0.3807, 0.4, 0.4116,
                0.4182, 0.4236, 0.4281, 0.4306,
                0.4348, 0.4378, 0.4407, 0.4427,
                0.4443, 0.446, 0.4478, 0.4492,
                0.4504, 0.4516, 0.4523, 0.4533,
                0.4541, 0.4554, 0.4562, 0.4565,
                0.4571, 0.4584]

tc64_lr01_l100 = [0.3747, 0.4126, 0.4323, 0.446,
                0.4577, 0.4665, 0.4735, 0.4797,
                0.4849, 0.4899, 0.4941, 0.4983,
                0.5022, 0.5058, 0.5092, 0.5125,
                0.516, 0.5188, 0.5217, 0.5245,
                0.5276, 0.5299, 0.5326, 0.5353,
                0.5376, 0.5398]

tc64_lr01_l1000 = [0.4062, 0.4674, 0.5028, 0.5337,
                0.5611, 0.5848, 0.6055, 0.6249,
                0.6435, 0.6596, 0.6752, 0.6899,
                0.7044, 0.7174, 0.729, 0.7401,
                0.7506, 0.7609, 0.7712, 0.7805,
                0.7889, 0.7968, 0.8042, 0.8114,
                0.818, 0.825]

tc256_lr01_l1 = []

tc256_lr01_l10 = []

tc256_lr01_l100 = []

tc256_lr01_l1000 = []

# Plot learning curves
fig = plt.figure()
plt.plot(iters, tc64_lr01_l1, color="black", ls='-', label=r'1 leaf')
plt.plot(iters, tc64_lr01_l10, color="blue", ls=':', label=r'10 leaf')
plt.plot(iters, tc64_lr01_l100, color="red", ls='--', label=r'100 leaf')
plt.plot(iters, tc64_lr01_l1000, color="green", ls=':', label=r'1000 leaf')
plt.xlabel(r"Training iters (Tree)")
plt.ylabel(r"nDCG@10")
plt.legend(loc='upper left')
plt.xticks(list(range(0, 250, 20)))
plt.xlim(1,250)
fig.savefig('LambdaMART_64_training')
plt.close(fig)

# Plot learning curves
# fig = plt.figure()
# plt.plot(iters, tc256_lr01_l1, color="black", ls='-', label=r'1 leaves')
# plt.plot(iters, tc256_lr01_l10, color="blue", ls=':', label=r'10 leaves')
# plt.plot(iters, tc256_lr01_l100, color="red", ls='--', label=r'100 leaves')
# plt.plot(iters, tc256_lr01_l1000, color="green", ls=':', label=r'1000 leaves')
# plt.xlabel(r"Training iters (Tress)")
# plt.ylabel(r"nDCG@10")
# plt.legend(loc='lower left')
# plt.xticks(list(range(0, 21, 5)))
# plt.xlim(1,250)
# fig.savefig('LambdaMART_256_training')
# plt.close(fig)
