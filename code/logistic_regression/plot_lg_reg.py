import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


C = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
ndcg_vali = [0.310, 0.301, 0.303, 0.299, 0.295, 0.286, 0.277, 0.270]

# Plot learning curves
fig = plt.figure()
plt.semilogx(C, ndcg_vali, color="blue", ls=':', label=r'')
plt.xlabel(r"L2 reguralisation coefficient")
plt.ylabel(r"nDCG@10")
plt.xticks(C)
fig.savefig('../../document/figures/lg_reg.png')
plt.close(fig)
