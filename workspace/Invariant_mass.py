import uproot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path




str_to_parent_folder=str(Path(__file__).resolve().parent.parent)
files=str_to_parent_folder+"/forward_data/*/*.myOutput.root"
tree_name=b'tree;1'
list_of_variables=[b"p_et_calo",b"p_phi",b"p_eta",b"tag_et_calo",b"tag_phi",b"tag_eta"]

#tree = uproot.open(str_to_parent_folder+"/forward_data/*/*.myOutput.root")[b'tree;1']

tree = uproot.iterate(str_to_parent_folder+"/forward_data/*/*.myOutput.root",b'tree;1',[b"p_et_calo",b"p_phi",b"p_eta",b"tag_et_calo",b"tag_phi",b"tag_eta"])

list_of_dataframes = []
for arrays in uproot.iterate(files,tree_name,list_of_variables,outputtype=pd.DataFrame,reportpath=True,reportfile=True):
    print(arrays[0])
    list_of_dataframes.append(arrays[2])
Invariant_mass_substuents=pd.concat(list_of_dataframes)



""" for p_et_calo,p_phi,p_eta,tag_et_calo,tag_phi,tag_eta in tree.iterate(["p_et_calo","p_phi","p_eta","tag_et_calo","tag_phi","tag_eta"], outputtype=tuple):
    p_p = p_et_calo
    t_p = tag_et_calo
    p_phis = p_phi
    t_phis = tag_phis
    p_etas = p_eta
    t_etas = tag_eta
    print(p_p) """
""" Invariant_mass_substuents= tree.pandas.df(["p_et_calo","p_phi","p_eta","tag_et_calo","tag_phi","tag_eta"])
 """

def Invariant_mass(Invariant_mass_substuents):
    Invariant_mass_result =2*Invariant_mass_substuents["p_et_calo"].to_numpy()*Invariant_mass_substuents["tag_et_calo"].to_numpy()*(np.cosh(Invariant_mass_substuents["p_eta"].to_numpy()-Invariant_mass_substuents["tag_eta"].to_numpy())-np.cos(Invariant_mass_substuents["p_phi"].to_numpy()-Invariant_mass_substuents["tag_phi"].to_numpy()))
    Invariant_mass_result =Invariant_mass_result[Invariant_mass_result !=-np.inf]*10**(-9)
    p_et_calo = Invariant_mass_substuents["p_et_calo"]*10**(-3)
    tag_et_calo = Invariant_mass_substuents["tag_et_calo"]*10**(-3)
    return Invariant_mass_result,p_et_calo,tag_et_calo

Invariant_mass_final,p_et_calo,tag_et_calo = Invariant_mass(Invariant_mass_substuents)




def hists(data,binwidth):
    plt.hist(data, bins=np.arange(0, 65 + binwidth, binwidth),log=True)


hists(Invariant_mass_final,2)
hists(p_et_calo,2)
hists(tag_et_calo,2)