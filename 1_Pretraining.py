# -*- coding: utf-8 -*-

import sys
sys.path.append(r'/data/lliang/molecular optimization')
import utils
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from Model import MolOpt
import matplotlib.pyplot as plt
import re
import time
from rdkit import Chem
import random
from tqdm import tqdm
import os
os.chdir('/data/lliang/molecular optimization')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def Encode_data(final_pairs):
    print('Start encoding data.........')
    # Encode Data
    core_fragment_list = []
    source_fragment_list = []
    target_fragment_list = []
    for pair in final_pairs:
        core_fragment = pair[0]
        core_fragment_list.append(core_fragment)
        source_fragment, target_fragment = pair[1], pair[2]
        source_fragment_list.append(source_fragment[1:])
        target_fragment_list.append(target_fragment[1:])

    core_fragment_list, growing_site_list = utils.ProcessCoreFragmentBatch(
        core_fragment_list)  # the atom number within breaking site in growing_site_list
    subsource_data = utils.encode(source_fragment_list, pad_size=60)
    subtarget_data = utils.encode(target_fragment_list, pad_size=60)
    print('Finish encoding data.........')

    return core_fragment_list, growing_site_list, subsource_data, subtarget_data

def training(args,model,optimizer,train_core_fragment,train_growing_site,train_source_fragment,train_source_fragment_lens,train_target_fragment,train_target_fragment_lens):
    if len(train_source_fragment) % args['batch_size'] == 0:
        train_batches = int(len(train_source_fragment)/args['batch_size'])
    else:
        train_batches = int(len(train_source_fragment)/args['batch_size']) + 1

    model.train()
    train_loss = 0
    for batch_idx in range(train_batches):
        optimizer.zero_grad()

        fragment_batch = train_core_fragment[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(train_source_fragment))]
        core_fragment_data = utils.EncodeCoreFragmentBatch(fragment_batch)
        growing_site = torch.tensor(train_growing_site[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(train_source_fragment))]).long().to(args['device'])
        #Construct source and target for RNN
        data_source = train_source_fragment[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(train_source_fragment))].to(args['device'])
        data_target = train_target_fragment[batch_idx * args['batch_size']:min((batch_idx + 1) * args['batch_size'],len(train_target_fragment))].to(args['device'])
        lengths = train_target_fragment_lens[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(train_target_fragment))]
        lengths = [i-1 for i in lengths]#the length of prev and nexts should reduce by 1
        prevs = data_source[:,:-1]
        nexts = data_target[:,1:]
        nexts = rnn_utils.pack_padded_sequence(nexts, lengths, enforce_sorted=False,batch_first=True)
        nexts, _ = rnn_utils.pad_packed_sequence(nexts, batch_first=True)
        outputs,hidden,_ = model(core_fragment_data,growing_site,data_source,prevs,lengths,hidden=None)
        #loss
        loss = F.cross_entropy(outputs.contiguous().view(-1, outputs.shape[-1]),
                         nexts.contiguous().view(-1),
                         ignore_index=0)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
    return train_loss / train_batches

def testing(args,model,test_core_fragment,test_growing_site,test_source_fragment,test_source_fragment_lens,test_target_fragment,test_target_fragment_lens):
    if len(test_source_fragment)%args['batch_size'] == 0:
        test_batches = int(len(test_source_fragment)/args['batch_size'])
    else:
        test_batches = int(len(test_source_fragment)/args['batch_size'])+1
    
    model.eval()
    test_loss = 0
    for batch_idx in range(test_batches):
        fragment_batch = test_core_fragment[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(test_source_fragment_lens))]
        core_fragment_data = utils.EncodeCoreFragmentBatch(fragment_batch)
        growing_site = torch.tensor(test_growing_site[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(test_source_fragment_lens))]).long().to(args['device'])

        #Construct source and target for RNN
        data_source = test_source_fragment[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(test_source_fragment_lens))].to(args['device'])
        data_target = test_target_fragment[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(test_target_fragment_lens))].to(args['device'])
        lengths = test_target_fragment_lens[batch_idx*args['batch_size']:min((batch_idx+1)*args['batch_size'],len(test_target_fragment_lens))]
        lengths = [i-1 for i in lengths]#the length of prev and nexts should reduce by 1
        prevs = data_source[:,:-1]
        nexts = data_target[:,1:]
        nexts = rnn_utils.pack_padded_sequence(nexts, lengths, enforce_sorted=False,batch_first=True)#return（batch_size,max_length）
        nexts, _ = rnn_utils.pad_packed_sequence(nexts, batch_first=True)
        outputs,hidden,_ = model(core_fragment_data,growing_site,data_source,prevs,lengths,hidden=None)

        loss = F.cross_entropy(outputs.contiguous().view(-1, outputs.shape[-1]),
                         nexts.contiguous().view(-1),
                         ignore_index=0)
        test_loss += loss.item()
        test_loss += loss.item()
    return test_loss / test_batches

def plot_loss(x_axis,train_loss_list,test_loss_list):
    plt.figure(figsize=(10,8))
    plt.grid(True)
    plt.plot(x_axis,train_loss_list,'bo',label='Train Loss')
    plt.plot(x_axis,train_loss_list,'b')
    plt.plot(x_axis,test_loss_list,'ro',label='Test Loss')
    plt.plot(x_axis,test_loss_list,'r')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Epochs',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    #plt.title('Training and validation acc',fontsize=25)
    plt.legend(fontsize=18)
    plt.savefig(args['out_path']+'/prior-loss.png',dpi=200)
    plt.show()
    return 

def smi_filter(smi):
    atoms =  ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's','H', 'F', 'I', 'Cl','Br']
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if mol.GetNumAtoms() < 60:
            count = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in atoms:
                    count += 1
            if count == len(mol.GetAtoms()):
                return True
    return False

'''
data = pd.read_csv('./data/pretrain/pretrain_4881w_f_cut.csv',header=None)
data = data.iloc[:,1:]
final_pairs = []
core_iden = []
for i in range(data.shape[0]):
    if i % 2000000 == 0:
        print(i)
    if smi_filter(data.iloc[i,0]) and smi_filter(data.iloc[i,1]):
        core_fragment = re.sub(r"/|\\", "", data.iloc[i,2])
        subsource = re.sub(r"/|\\", "", data.iloc[i,3])
        subtarget = re.sub(r"/|\\", "", data.iloc[i,4])
      #  if core_fragment not in core_iden:
        if core_iden.count(core_fragment) < 6:
            core_iden.append(core_fragment)
            if len(subsource[:]) < 60 and len(subsource[:]) >= 2:
                if len(subtarget[:]) < 60 and len(subtarget[:]) >= 2:
                    mol_subsource, mol_subtarget = Chem.MolFromSmiles(subsource), Chem.MolFromSmiles(subtarget)
                    for atom in mol_subsource.GetAtoms():
                        atom.SetFormalCharge(0)
                        atom.SetNoImplicit(True)
                        atom.SetNumExplicitHs(0)
                    subsource_new = Chem.MolToSmiles(mol_subsource, isomericSmiles=False)
                    for atom in mol_subtarget.GetAtoms():
                        atom.SetFormalCharge(0)
                        atom.SetNoImplicit(True)
                        atom.SetNumExplicitHs(0)
                    subtarget_new = Chem.MolToSmiles(mol_subtarget, isomericSmiles=False)
                    if subsource_new != subtarget_new:
                        if Chem.MolFromSmiles(core_fragment) is not None and Chem.MolFromSmiles(core_fragment.replace('*', '')) is not None:
                            if Chem.MolFromSmiles(subsource_new) is not None and Chem.MolFromSmiles(subsource_new.replace('*', '')) is not None:
                                if Chem.MolFromSmiles(subtarget_new) is not None and Chem.MolFromSmiles(subtarget_new.replace('*', '')) is not None:
                                    if Chem.MolFromSmiles(subsource_new).GetNumAtoms() > 1 and Chem.MolFromSmiles(subtarget_new).GetNumAtoms() > 1:
                                        if Chem.MolToSmiles(Chem.MolFromSmiles(subsource_new.replace('*', '')),
                                                            canonical=True) != Chem.MolToSmiles(
                                                Chem.MolFromSmiles(subtarget_new.replace('*', '')), canonical=True):
                                         #   final_pairs.append([core_fragment, subsource_new, subtarget_new])
                                            final_pairs.append([data.iloc[i,0], data.iloc[i,1], core_fragment, subsource_new, subtarget_new])
final_pairs = [list(x) for x in set(tuple(x) for x in final_pairs)]  # 嵌套列表去重

print('pairs_num: ', len(final_pairs))  

with open('./data/pairs_new_cpd.txt', 'w') as f:  # 180008
    for pair in final_pairs:
        n = len(pair)
        for i in range(n):
            if i+1 == n:
                line = pair[i] + '\n'
                f.write(line)
            else:
                line = pair[i] + '   '
                f.write(line)
f.close()
'''

final_pairs = []
for pair in list(open('./data/pairs_new_cpd.txt')):
    pair_new = []
    pair_new.append(pair.split('  ')[2].strip())
    pair_new.append(pair.split('  ')[3].strip())
    pair_new.append(pair.split('  ')[4].strip())
    final_pairs.append(pair_new)
final_pairs = random.sample(final_pairs,100000)

'''
from rdkit.Chem import AllChem
for i in range(len(final_pairs)):
    if i == 10000:
        break
    pair = final_pairs[i]
    if Chem.MolFromSmiles(pair[1].replace('*','')) is not None and Chem.MolFromSmiles(pair[2].replace('*','')) is not None and Chem.MolFromSmiles(pair[1]) is not None and Chem.MolFromSmiles(pair[2]) is not None:
        rxn = AllChem.ReactionFromSmarts(">>")
        reactant = Chem.MolFromSmiles(pair[1].replace('*', ''))
        rxn.AddReactantTemplate(reactant)
        product = Chem.MolFromSmiles(pair[2].replace('*', ''))
        rxn.AddProductTemplate(product)
        with open('./check_pairs/pair_{0}.rxn'.format(i), 'w') as file:
            file.write(AllChem.ReactionToRxnBlock((rxn)))
'''

#3 encode generated fragment pairs
core_fragment_list, growing_site_list, subsource_data, subtarget_data = Encode_data(final_pairs)

#3 pretrain
args = {'batch_size':256,'lr':0.0005,'epochs':15,'split':0.8,'device':'cuda:2','out_path':'./result2'}

#split Data
train_split = int(args['split']*len(subsource_data[0]))
train_core_fragment = core_fragment_list[:train_split]
test_core_fragment = core_fragment_list[train_split:]
train_growing_site = growing_site_list[:train_split]
test_growing_site = growing_site_list[train_split:]
train_source_fragment = subsource_data[0][:train_split]
test_source_fragment = subsource_data[0][train_split:]
train_source_fragment_lens = subsource_data[1][:train_split]
test_source_fragment_lens = subsource_data[1][train_split:]
train_target_fragment = subtarget_data[0][:train_split]
test_target_fragment = subtarget_data[0][train_split:]
train_target_fragment_lens = subtarget_data[1][:train_split]
test_target_fragment_lens = subtarget_data[1][train_split:]

print('Start Training.........')
#清除缓存
torch.cuda.empty_cache()
#使用cudnn
torch.backends.cudnn.enabled = True

torch.manual_seed(0)
epochs = args['epochs']
model = MolOpt().to(args['device'])
optimizer = optim.Adam(model.parameters(),lr=args['lr'])

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# this code is very important! It initialises the parameters with a
# range of values that stops the signal fading or getting too big.

x_axis = []
train_loss_list = []
test_loss_list = []
train_loss_avg = 0
test_loss_avg = 0
plot_every = 1

for epoch in range(1, epochs + 1):
    begin = time.time()
    train_loss = training(args,model,optimizer,train_core_fragment,train_growing_site,train_source_fragment,train_source_fragment_lens,train_target_fragment,train_target_fragment_lens)
    test_loss = testing(args,model,test_core_fragment,test_growing_site,test_source_fragment,test_source_fragment_lens,test_target_fragment,test_target_fragment_lens)
    print('epoch: {}, train_loss: {:.2f}, test_loss: {:.2f}'.format(epoch,train_loss,test_loss))
    train_loss_avg += train_loss
    test_loss_avg += test_loss

    if epoch % plot_every == 0:
        train_loss_list.append(train_loss_avg / plot_every)
        test_loss_list.append(test_loss_avg / plot_every)
        train_loss_avg = 0
        test_loss_avg = 0
        x_axis.append(epoch)
    torch.save(model.state_dict(), args['out_path'] + '/prior-{:02d}.pth'.format(epoch))
    end = time.time()
    print('training time: {:.2f} h'.format((end-begin)/3600))

#Draw Loss
plot_loss(x_axis,train_loss_list,test_loss_list)

#Output training log
x_axis = np.array(x_axis)
train_loss_list = np.array(train_loss_list)
test_loss_list = np.array(test_loss_list)
out_log = np.vstack((x_axis,train_loss_list,test_loss_list))
out_log = np.transpose(out_log)
np.savetxt(args['out_path']+'/prior-log.txt', out_log,fmt='%f',delimiter=',')





