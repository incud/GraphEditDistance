import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

diff_df = pd.read_pickle("data/difference_value_simple.pickle")
diff_df = diff_df.rename(columns={'GED_vs_sim': 'sim',
                                  'GED_vs_2000': '2k',
                                  'GED_vs_2000_lt': '2k L',
                                  'GED_vs_2000_pm': '2k P',
                                  'GED_vs_adv':     'Adv',
                                  'GED_vs_adv_lt':  'Adv L',
                                  'GED_vs_adv_pm':  'Adv P',
                                  'GED_vs_leap': 'Leap'})

diff_b10 = diff_df[diff_df['b'] == 0.10].drop('b', axis=1).fillna(1000).groupby(['vertices']).max()
diff_b05 = diff_df[diff_df['b'] == 0.05].drop('b', axis=1).fillna(1000).groupby(['vertices']).max()
diff_b01 = diff_df[diff_df['b'] == 0.01].drop('b', axis=1).fillna(1000).groupby(['vertices']).max()

sns.set()

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig.suptitle('Difference between expected and actual edit distance using many values of b')

ax = sns.heatmap(diff_b10, ax=axes[0], linewidth=0, vmin=0, vmax=50)
ax.tick_params(length=0)
ax.set_title('b=0.10')
# for lbl in ax.get_xticklabels():
#     lbl.set_rotation(30)

# ax = sns.heatmap(diff_b05, ax=axes[1], linewidth=0, vmin=0, vmax=50)
# ax.tick_params(length=0)
# ax.set_title('b=0.05')
# # for lbl in ax.get_xticklabels():
# #     lbl.set_rotation(30)
#
# ax = sns.heatmap(diff_b01, ax=axes[2], linewidth=0, vmin=0, vmax=50)
# ax.tick_params(length=0)
# ax.set_title('b=0.01')
# # for lbl in ax.get_xticklabels():
# #     lbl.set_rotation(30)

plt.show()
