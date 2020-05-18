import uproot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import h5py    

str_to_parent_folder=str(Path(__file__).resolve().parent.parent)

file_path = str_to_parent_folder+"/Data_pid.h5"

f1 = h5py.File(file_path,'r+')  

key_list = list(f1.items())
print(key_list[0][0])
all_scalar_eval = f1.get(str(key_list[0][0]))       

#print(all_scalar_eval[:].shape)
#print(all_scalar_eval[:]dtype)
Invariant_mass_substuents = all_scalar_eval["p_et_calo","p_phi","p_eta","tag_et_calo","tag_phi","tag_eta"]

def Invariant_mass(Invariant_mass_substuents):
    Invariant_mass_result =2*Invariant_mass_substuents["p_et_calo"]*Invariant_mass_substuents["tag_et_calo"]*(np.cosh(Invariant_mass_substuents["p_eta"]-Invariant_mass_substuents["tag_eta"])-np.cos(Invariant_mass_substuents["p_phi"]-Invariant_mass_substuents["tag_phi"]))
    Invariant_mass_result =Invariant_mass_result[Invariant_mass_result !=-np.inf]*10**(-3)
    p_et_calo = Invariant_mass_substuents["p_et_calo"]*10**(-3)
    tag_et_calo = Invariant_mass_substuents["tag_et_calo"]*10**(-3)
    return Invariant_mass_result,p_et_calo,tag_et_calo


Invariant_mass_final,p_et_calo,tag_et_calo = Invariant_mass(Invariant_mass_substuents)

print(Invariant_mass_final)


def hists(data,binwidth):
    plt.hist(data, bins=np.arange(0, 150 + binwidth, binwidth),log=True)


hists(Invariant_mass_final,2)
hists(p_et_calo,2)
hists(tag_et_calo,2)