from hep_ml.reweight import GBReweighter
import uproot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

tree = uproot.open("../forward_MC/user.lehrke.mc16_13TeV.361106.Zee.EGAM8.e3601_e5984_s3126_r10724_r10726_p3648.ePID18_NTUP3_v01_myOutput.root/user.lehrke.17118381._000003.myOutput.root")[b'tree;1']
reweight_data = tree.pandas.df(["averageInteractionsPerCrossing","p_et_calo","p_eta","p_TruthType"])
reweight_data_small = reweight_data.sample(frac=1,replace=False, random_state=42)

p_TruthType_reweight_data = reweight_data_small.where(reweight_data['p_TruthType'] == 2)
p_TruthType_reweight_data_s_dropped = p_TruthType_reweight_data.drop(['p_TruthType'], axis=1)
p_TruthType_reweight_data_nan_s_dropped = p_TruthType_reweight_data_s_dropped.dropna(axis=0)
background_reweight_data = reweight_data_small.where(reweight_data['p_TruthType'] !=2)
background_reweight_data_s_dropped = background_reweight_data.drop(['p_TruthType'], axis=1)
background_reweight_data_nan_s_dropped =background_reweight_data_s_dropped.dropna(axis=0)

ratio=len(p_TruthType_reweight_data_nan_s_dropped)/len(background_reweight_data_nan_s_dropped)

reweighter = GBReweighter(n_estimators=40)
reweighter.fit(background_reweight_data_nan_s_dropped, p_TruthType_reweight_data_nan_s_dropped)
weights = reweighter.predict_weights(background_reweight_data_nan_s_dropped)
print(weights)

total_weights=ratio*weights /np.mean(weights)
np.savetxt('../weights/weights_MC_03.csv', total_weights, delimiter=',')
#reweighted_background = background_reweight_data.multiply(weights, axis=0)

fig_weight, ax_weight = plt.subplots(3,2, figsize=(15,15))


ax_weight[0,0].hist(p_TruthType_reweight_data_nan_s_dropped.p_et_calo.ravel(),bins=50,range=(0,100000), color = 'r', alpha = 0.5, label = "p_TruthType")
ax_weight[0,0].hist(background_reweight_data_nan_s_dropped.p_et_calo.ravel(), bins=50,range=(0,100000), color = 'blue', alpha = 0.5, label = "Background")
ax_weight[0,0].legend(loc="upper right")
ax_weight[0,0].set_title('P_et_calo (before weight)')


ax_weight[0,1].hist(p_TruthType_reweight_data_nan_s_dropped.p_et_calo.ravel(), bins=50,range=(0,100000), color = 'r', alpha = 0.5, label = "p_TruthType")
ax_weight[0,1].hist(background_reweight_data_nan_s_dropped.p_et_calo.ravel(),  bins=50, range=(0,100000), color = 'blue', weights=total_weights, alpha = 0.5, label = "Background")
ax_weight[0,1].legend(loc="upper right")
ax_weight[0,1].set_title('P_et_calo (after weight)')

ax_weight[1,0].hist(p_TruthType_reweight_data_nan_s_dropped.p_eta.ravel(), bins=50, color = 'r', alpha = 0.5, label = "p_TruthType")
ax_weight[1,0].hist(background_reweight_data_nan_s_dropped.p_eta.ravel(), bins=50, color = 'blue', alpha = 0.5, label = "Background")
ax_weight[1,0].legend(loc="upper right")
ax_weight[1,0].set_title('p_eta (before weight)')

#ax_weight[1,0].hist(background_reweight_data_nan_s_dropped.p_eta.ravel(), bins = 50, color = 'blue',weights=total_weights, alpha = 0.5, label = "Background")
#ax_weight[1,0].legend(loc="upper right")
#ax_weight[1,0].set_title('p_et_calo, background only, after')


ax_weight[1,1].hist(p_TruthType_reweight_data_nan_s_dropped.p_eta.ravel(), bins=50, color = 'r', alpha = 0.5, label = "p_TruthType")
ax_weight[1,1].hist(background_reweight_data_nan_s_dropped.p_eta.ravel(), bins=50, color = 'blue',weights=weights, alpha = 0.5, label = "Background")
ax_weight[1,1].legend(loc="upper right")
ax_weight[1,1].set_title('p_eta(after weight)')

ax_weight[2,0].hist(p_TruthType_reweight_data_nan_s_dropped.averageInteractionsPerCrossing, bins=50, color = 'r', alpha = 0.5, label = "p_TruthType")
ax_weight[2,0].hist(background_reweight_data_nan_s_dropped.averageInteractionsPerCrossing, bins=50, color = 'blue', alpha = 0.5, label = "Background")
ax_weight[2,0].legend(loc="upper right")
ax_weight[2,0].set_title('averageInteractionsPerCrossing (before weight)')

ax_weight[2,1].hist(p_TruthType_reweight_data_nan_s_dropped.averageInteractionsPerCrossing, bins=50, color = 'r', alpha = 0.5, label = "p_TruthType")
ax_weight[2,1].hist(background_reweight_data_nan_s_dropped.averageInteractionsPerCrossing, bins=50, color = 'blue',weights=weights, alpha = 0.5, label = "Background")
ax_weight[2,1].legend(loc="upper right")
ax_weight[2,1].set_title('averageInteractionsPerCrossing(after weight)')

plt.savefig("../graphs/weights/reweights_p_et-p_eta-average_interactions")

