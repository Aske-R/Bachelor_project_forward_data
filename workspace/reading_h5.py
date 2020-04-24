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
