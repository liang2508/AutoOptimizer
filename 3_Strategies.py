# -*- coding: utf-8 -*-

import sys
sys.path.append(r'/data/lliang/molecular optimization')
import torch
import utils
from Model import MolOpt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import re
import random
import os
os.chdir('/data/lliang/molecular optimization')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def find_long_carbon_chains(mol):
    n = mol.GetNumAtoms()
    i = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            i += 1
    if n == i or n < 2:
        return False
    else:
        return True

def main():

    args = {'device': 'cuda:2', 'generator': './result/tl-25.pth', 'out': './result/rxn/'}

    pairs = list(open('./data/pairs_new.txt'))
    pairs = random.sample(pairs, 10000)

    np.random.seed(0)
    torch.manual_seed(0)
    #empty_cache###
    torch.cuda.empty_cache()
    #using cudnn###
    torch.backends.cudnn.enabled = True

    model = MolOpt().to(args['device'])
    model.load_state_dict(torch.load(args['generator']))
    model.eval()


    for i in range(len(pairs)):
        pair = pairs[i]
        core_fragment = pair.split('  ')[0].strip()
        source_fragment = pair.split('  ')[1].strip()
        if Chem.MolFromSmiles(core_fragment) is not None:
            if Chem.MolFromSmiles(source_fragment) is not None:
                if Chem.MolFromSmiles(source_fragment.replace('*','')) is not None:
                    mol = Chem.MolFromSmiles(core_fragment)
                    core_fragment = Chem.MolToSmiles(mol)
                    generated_fragments, generated_fragments_new = [], []
                    new_x = model.optimization(256, core_fragment=core_fragment, source_fragment=source_fragment, T=1)
                    generated_fragments += utils.decode(new_x)
                    generated_fragments = list(set(generated_fragments))
                    for frag in generated_fragments:
                        if frag != '':
                            m = Chem.MolFromSmiles(frag)
                            if m is not None:
                                if find_long_carbon_chains(m):
                                    generated_fragments_new.append(frag)
                    if len(generated_fragments_new) != 0:
                        print(i,end=" ")
                        print(source_fragment,end=" ")
                        print(generated_fragments_new)
                        for j in range(len(generated_fragments_new)):
                            target_fragment = generated_fragments_new[j]
                            rxn = AllChem.ReactionFromSmarts(">>")
                            reactant = Chem.MolFromSmiles(source_fragment.replace('*',''))
                            rxn.AddReactantTemplate(reactant)
                            product = Chem.MolFromSmiles(target_fragment)
                            rxn.AddProductTemplate(product)
                            with open(args['out']+"reaction_{0}_{1}.rxn".format(i,j), "w") as file:
                                file.write(AllChem.ReactionToRxnBlock(rxn))

if __name__ == '__main__':
    main()



'''
from rdkit import Chem
from rdkit.Chem import AllChem
import shutil
import os
os.chdir('/data/lliang/molecular optimization')

def find_long_carbon_chains(mol):
    n = mol.GetNumAtoms()
    i = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            i += 1
    if n == i or n < 2:
        return False
    else:
        return True

rxn_tl = os.listdir('./result/rxns')
for rxn in rxn_tl:
    reaction = AllChem.ReactionFromRxnFile('./result/rxns/' + rxn)
    for mol in reaction.GetProducts():
        if mol is not None:
            if find_long_carbon_chains(mol):
                shutil.move('./result/rxns/' + rxn, './result/rxn_tl_new/')
'''