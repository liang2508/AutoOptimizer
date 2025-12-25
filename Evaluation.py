# -*- coding: utf-8 -*-

import sys
sys.path.append(r'/data/lliang/molecular optimization')
import os
os.chdir('/data/lliang/molecular optimization')

# ************************* change into dti env ****************************
# Properties
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from tqdm import tqdm
from joblib import Parallel,delayed
import sascorer

def calc_similarity(smi,smi_ref):
    ref_mol = Chem.MolFromSmiles(smi_ref)
    mol = Chem.MolFromSmiles(smi)
    sim = TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(ref_mol,radius=2,nBits=2048), AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048))
    return sim

def properties(smi):
    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    qed = QED.qed(mol)
    sa = sascorer.my_score(mol)
    tpsa = Descriptors.TPSA(mol)
    return smi, mw, clogp, qed, sa, tpsa

smis_pre = [Chem.MolToSmiles(mol) for mol in tqdm(Chem.SDMolSupplier('out2_pre.sdf')) if mol is not None]
results_pre = Parallel(n_jobs=64)(delayed(properties)(smi) for smi in tqdm(smis_pre))
smis_pre, mw_pre, clogp_pre, qed_pre, sa_pre, tpsa_pre = zip(*results_pre)
sim_pre = []
smi_ref = Chem.MolToSmiles(Chem.SDMolSupplier('LT-842-737.sdf')[0])
for smi in tqdm(smis_pre):
    sim = calc_similarity(smi,smi_ref)
    sim_pre.append(sim)

smis_tl = [Chem.MolToSmiles(mol) for mol in tqdm(Chem.SDMolSupplier('out2_tl.sdf')) if mol is not None]
results_tl = Parallel(n_jobs=96)(delayed(properties)(smi) for smi in tqdm(smis_tl))
smis_tl, mw_tl, clogp_tl, qed_tl, sa_tl, tpsa_tl = zip(*results_tl)
sim_tl = []
smi_ref = Chem.MolToSmiles(Chem.SDMolSupplier('LT-842-737.sdf')[0])
for smi in tqdm(smis_tl):
    sim = calc_similarity(smi,smi_ref)
    sim_tl.append(sim)


df_pre = pd.DataFrame(list(zip(smis_pre, mw_pre, clogp_pre, qed_pre, sa_pre, tpsa_pre, sim_pre)), columns=['smis_pre', 'mw_pre', 'clogp_pre', 'qed_pre', 'sa_pre', 'tpsa_pre', 'sim_pre'])
df_pre.to_csv('properties_pre.csv',index=False)
df_tl = pd.DataFrame(list(zip(smis_tl, mw_tl, clogp_tl, qed_tl, sa_tl, tpsa_tl, sim_tl)), columns=['smis_tl', 'mw_tl', 'clogp_tl', 'qed_tl', 'sa_tl', 'tpsa_tl', 'sim_tl'])
df_tl.to_csv('properties_tl.csv',index=False)
'''

df_pre, df_tl = pd.read_csv('properties_pre.csv'), pd.read_csv('properties_tl.csv')
mw_pre, mw_tl = [i for i in list(df_pre.mw_pre) if i<700], [j for j in list(df_tl.mw_tl) if j<700]
diff_mw = ['Pre-training']*len(mw_pre) + ['Fine-tuning']*len(mw_tl)
data_mw = pd.DataFrame([mw_pre+mw_tl,diff_mw],index=['Molecular Weight','diff_mw']).T
clogp_pre, clogp_tl = [i for i in list(df_pre.clogp_pre) if i<700], [j for j in list(df_tl.clogp_tl) if j<700]
diff_clogp = ['Pre-training']*len(clogp_pre) + ['Fine-tuning']*len(clogp_tl)
data_clogp = pd.DataFrame([clogp_pre+clogp_tl,diff_clogp],index=['cLogP','diff_clogp']).T
qed_pre, qed_tl = [i for i in list(df_pre.qed_pre) if i<700], [j for j in list(df_tl.qed_tl) if j<700]
diff_qed = ['Pre-training']*len(qed_pre) + ['Fine-tuning']*len(qed_tl)
data_qed = pd.DataFrame([qed_pre+qed_tl,diff_qed],index=['QED','diff_qed']).T
sa_pre, sa_tl = [i for i in list(df_pre.sa_pre) if i<10], [j for j in list(df_tl.sa_tl) if j<10]
diff_sa= ['Pre-training']*len(sa_pre) + ['Fine-tuning']*len(sa_tl)
data_sa = pd.DataFrame([sa_pre+sa_tl,diff_sa],index=['Synthetic accessibility score','diff_sa']).T

#palette = 'husl'
#custom_palette = sns.color_palette(["#D65DB1", "#FF6F91", "#FF9671", "#FFC75F", "#F9F871"])
custom_palette = sns.color_palette(["#af2dff", "#e1bc2a"])
sns.set_palette(["#af2dff", "#e1bc2a"],desat=0.8)

fig,ax = plt.subplots(2, 2, figsize=(12, 12))
g1 = sns.kdeplot(x='Molecular Weight',data=data_mw,hue=diff_mw,fill=True,palette=custom_palette,ax=ax[0][0])
g2 = sns.kdeplot(x='cLogP',data=data_clogp,hue=diff_clogp,fill=True,palette=custom_palette,ax=ax[0][1])
g3 = sns.kdeplot(x='QED',data=data_qed,hue=diff_qed,fill=True,palette=custom_palette,ax=ax[1][0])
g4 = sns.kdeplot(x='Synthetic accessibility score',data=data_sa,hue=diff_sa,fill=True,palette=custom_palette,ax=ax[1][1])

# sns.set_style('darkgrid')
for ax in plt.gcf().axes:
    l = ax.get_xlabel()
    ll = ax.get_ylabel()
    ax.set_xlabel(l, fontsize=16)
    ax.set_ylabel(ll,fontsize=16)
plt.savefig('./properties.png',bbox_inches='tight',dpi=800)
plt.show()
'''