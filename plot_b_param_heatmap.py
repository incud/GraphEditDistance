import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

diff_df = pd.read_pickle("data/difference_value_simple.pickle")
diff_2_df = pd.read_pickle("data/qaoa_vqe_difference_WITHB.pickle")
diff_2_df = diff_2_df.drop('b',axis=1)
diff_2_df = diff_2_df.drop('v',axis=1)
diff_2_df = diff_2_df.drop('GED',axis=1)

diff_df = pd.concat([diff_df,diff_2_df], axis = 1)
diff_df = diff_df.rename(columns={'GED_vs_sim': 'sim',
                                  'GED_vs_2000': '2k',
                                  'GED_vs_2000_lt': '2k L',
                                  'GED_vs_2000_pm': '2k P',
                                  'GED_vs_adv':     'Adv',
                                  'GED_vs_adv_lt':  'Adv L',
                                  'GED_vs_adv_pm':  'Adv P',
                                  'GED_vs_leap': 'Leap',
                                  'GED_vs_qaoaP1': 'QAOA p=1',
                                  'GED_vs_qaoaP3':'QAOA p=3',
                                  'GED_vs_vqeP1': 'VQE p=1',
                                  'GED_vs_vqeP3':'VQE p=3'})

diff_b10 = diff_df[diff_df['b'] == 0.10].drop('b', axis=1).fillna(1000).groupby(['vertices']).max()
diff_b05 = diff_df[diff_df['b'] == 0.05].drop('b', axis=1).fillna(1000).groupby(['vertices']).max()
diff_b01 = diff_df[diff_df['b'] == 0.01].drop('b', axis=1).fillna(1000).groupby(['vertices']).max()

sns.set()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=80)
fig.suptitle('Difference between expected and actual edit distance using many values of b')

ax = sns.heatmap(diff_b10, ax=axes[0], linewidth=0, vmin=0, vmax=500)
ax.tick_params(length=0)
ax.set_title('b=0.10')
#for lbl in ax.get_xticklabels():
#    lbl.set_rotation(30)

ax = sns.heatmap(diff_b05, ax=axes[1], linewidth=0, vmin=0, vmax=500)
ax.tick_params(length=0)
ax.set_title('b=0.05')
#for lbl in ax.get_xticklabels():
#    lbl.set_rotation(30)

ax = sns.heatmap(diff_b01, ax=axes[2], linewidth=0, vmin=0, vmax=500)
ax.tick_params(length=0)
ax.set_title('b=0.01')
#for lbl in ax.get_xticklabels():
#    lbl.set_rotation(30)

plt.show()
