# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
import rdkit
from rdkit import Chem


_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar',
          'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
          'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
          'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
          'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
          'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
          'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
          'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


def get_tokenizer_re(atoms):
    return re.compile('('+'|'.join(atoms)+r'|\%\d\d|.)')


_atoms_re = get_tokenizer_re(_atoms)
__i2t = {
    0: 'pad', 1: 'G', 2: 'E', 3: '2', 4: 'F', 5: 'Cl', 6: 'N',
    7: '[', 8: '6', 9: 'O', 10: 'c', 11: ']', 12: '#',
    13: '=', 14: '3', 15: ')', 16: '4', 17: '-', 18: 'n',
    19: 'o', 20: '5', 21: 'H', 22: '(', 23: 'C',
    24: '1', 25: 'S', 26: 's', 27: 'Br',28: '@',29:'+',30:'/',31:'\\',32:'I',33:'P',34:'7',35:'8',36:'9',37:'unused'
}


__t2i = {
    'pad':0,'G': 1, 'E': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
    'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
    '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
    'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27, '@' : 28, '+':29, '/':30, '\\':31,'I':32,
    'P':33,'7':34,'8':35,'9':36
}


def smiles_tokenizer(line, atoms=None):
    """
    Tokenizes SMILES string atom-wise using regular expressions. While this
    method is fast, it may lead to some mistakes: Sn may be considered as Tin
    or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
    specify a set of two-letter atoms explicitly.

    Parameters:
         atoms: set of two-letter atoms for tokenization
    """
    if atoms is not None:
        reg = get_tokenizer_re(atoms)
    else:
        reg = _atoms_re
    return reg.split(line)[1::2]

def encode(sm_list, pad_size):
    """
    Encoder list of smiles to tensor of tokens
    """
    res = []
    lens = []
    for s in sm_list:
        tokens = [__t2i[tok]
            for tok in smiles_tokenizer(s)]
        tokens = [1] + tokens + [2]
        lens.append(len(tokens))
        tokens += (pad_size - len(tokens)) * [0]
        res.append(tokens)
    return (torch.tensor(res).long(), lens)

def encode_fragment(string):
    token = [__t2i[tok]
              for tok in smiles_tokenizer(string)]
    return token

def decode(tokens_tensor):
    """
    Decodes from tensor of tokens to list of smiles
    """
    smiles_res = []
    for i in range(len(tokens_tensor)):
        cur_sm = ''
        for t in tokens_tensor[i].detach().cpu().numpy():
            if t == 1:
                continue
            if t == 2:
                break
            elif t > 2:
                cur_sm += __i2t[t]
                
        smiles_res.append(cur_sm)

    return smiles_res

def decode_single(tokens_tensor):
    """
    Decodes from tensor of tokens to one smile
    """
    cur_sm = ''
    for t in tokens_tensor.detach().cpu().numpy():
        if t==1:
            continue
        if t == 2:
            break
        elif t > 2:
            cur_sm += __i2t[t]

    return cur_sm

def get_vocab_size():
    return len(__i2t)

####process core fragment#########
def ProcessCoreFragment(fragment):#input is the SMILES of core fragment
    pattern = '\[\d*\*\]'

    #extract fragment
    p1 = re.compile(pattern)
    fragment = re.sub(p1,'[Au]',fragment)
    fragment = Chem.MolToSmiles(Chem.MolFromSmiles(fragment))
    fragment = fragment.replace('[Au]','[*]')#this step is intended for increasing the randomness of the position of '*'
    
    #growing site
    fragment_mol = Chem.MolFromSmiles(fragment)
    for idx in range(fragment_mol.GetNumAtoms()):
        atom = fragment_mol.GetAtomWithIdx(idx)
        if idx == 0 and atom.GetSymbol() == '*':
            growing_site = idx#if the idx of '*' is 0, the idx of the growing site will be 0 after '*' is removed
            break#end loop in advance
        else:
            neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]#obtain the symbol of neighboring atoms
            if '*' in neighbors:
                growing_site = idx
                break        
    return fragment,growing_site
                                
def ProcessCoreFragmentBatch(core_fragment_smi_list):
    fragment_list = []#fragment smile with '*'
    first_atom_list = []#the index of break site
    count = 0
    for fragment in core_fragment_smi_list:
        count += 1
        try:
            fragment,growing_site = ProcessCoreFragment(fragment)
        except:
            a = a + b
        fragment_list.append(fragment)
        first_atom_list.append(growing_site)
        #if count%50==0:
            #print(count)
    return fragment_list,first_atom_list

def normalize_adj(adj):#standardize the edge
    degrees = np.sum(adj,axis=2)
    # print('degrees',degrees)
    D = np.zeros((adj.shape[0],adj.shape[1],adj.shape[2]))
    for i in range(D.shape[0]):
        D[i,:,:] = np.diag(np.power(degrees[i,:],-0.5))
    adj_normal = D@adj@D
    adj_normal[np.isnan(adj_normal)]=0
    return adj_normal

def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z

def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([1,n_nodes, n_nodes*n_edge_types*2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[0,tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        a[0,src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1
    return a

def EncodeCoreFragment(smile):
    bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
    possible_atom_types = ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's','H', 'F', 'I', 'Cl','Br']
    max_atom = 60
    
    # remove stereo information, such as inward and outward edges
    #Chem.RemoveStereochemistry(mol)
    mol = Chem.MolFromSmiles(smile)
    mol=Chem.RWMol(mol)
    Chem.SanitizeMol(mol)

    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol()=='*':
            mol.RemoveAtom(idx)
            break

    edges = []
    nodes = []    
    #encoding node
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        nodes.append(onehot(possible_atom_types.index(symbol), len(possible_atom_types)))
    #padding
    for i in range(max_atom-mol.GetNumAtoms()):
        nodes.append([0]*len(possible_atom_types))
    
    #encoding edge
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
        #assert bond_dict[str(bond.GetBondType())] != 3    
    edges=create_adjacency_matrix(edges,max_atom, 4)
    return nodes,edges

def EncodeCoreFragmentBatch(smi_list):
    final_nodes=[]
    final_edges=np.zeros([1, 60, 60*4*2])
    
    for smi in smi_list:
        nodes,edges=EncodeCoreFragment(smi)
        final_nodes.append(nodes)
        final_edges=np.vstack((final_edges,edges))

    merge_state={}
    merge_state['adj']=final_edges[1:]
    merge_state['node']=final_nodes
    return merge_state

def ComputeAtomAromatic(m,atom_idx):#m refers to mol    
    atom=m.GetAtomWithIdx(atom_idx)
    neighbors= [x.GetIdx() for x in atom.GetNeighbors()]
    bond_type=[ str(m.GetBondBetweenAtoms(atom_idx,x).GetBondType()) for x in neighbors]
    if 'AROMATIC' in bond_type:
        standardized_atom=atom.GetSymbol().lower()#如果是芳香性原子，则变成小写的形式
    else:
        standardized_atom=atom.GetSymbol()
    return standardized_atom


