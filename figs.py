import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[0.712, 0.732, 0.734, 0.735], [0.730, 0.742, 0.745, 0.746], [0.731, 0.746, 0.751, 0.747],
         [0.729, 0.749, 0.747, 0.739]]
labels = [16, 32, 64, 128]
df_cm = pd.DataFrame(array)

plt.figure(figsize=(10, 7))
ax = plt.subplot()
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, fmt=".3g", ax=ax)  # font size
ax.set_xlabel('Dimension of the word vectors', fontsize=18)
ax.set_ylabel('Number of filters', fontsize=18)
ax.xaxis.set_ticklabels(labels, fontsize=14)
ax.yaxis.set_ticklabels(labels, fontsize=14)
# plt.show()
plt.savefig("OPENSTACK.pdf", bbox_inches='tight')

# array = [[0.731, 0.741, 0.747, 0.742], [0.745, 0.747, 0.754, 0.745], [0.747, 0.761, 0.768, 0.763],
#          [0.746, 0.756, 0.765, 0.766]]
# labels = [16, 32, 64, 128]
# df_cm = pd.DataFrame(array)
#
# plt.figure(figsize=(10, 7))
# ax = plt.subplot()
# sn.set(font_scale=1.4)  # for label size
# sn.heatmap(df_cm, annot=True, fmt=".3g", ax=ax)  # font size
# ax.set_xlabel('Dimension of the word vectors', fontsize=18)
# ax.set_ylabel('Number of filters', fontsize=18)
# ax.xaxis.set_ticklabels(labels, fontsize=14)
# ax.yaxis.set_ticklabels(labels, fontsize=14)
# plt.savefig("QT.pdf", bbox_inches='tight')
