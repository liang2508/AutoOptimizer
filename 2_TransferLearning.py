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
from rdkit import Chem
import random
import re
from tqdm import tqdm
import os
os.chdir('/data/lliang/molecular optimization')


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
    subsource_data = utils.encode(source_fragment_list, pad_size=80)
    subtarget_data = utils.encode(target_fragment_list, pad_size=80)
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


def testing(args, model, test_core_fragment, test_growing_site, test_source_fragment, test_source_fragment_lens,
            test_target_fragment, test_target_fragment_lens):
    if len(test_source_fragment) % args['batch_size'] == 0:
        test_batches = int(len(test_source_fragment) / args['batch_size'])
    else:
        test_batches = int(len(test_source_fragment) / args['batch_size']) + 1

    model.eval()
    test_loss = 0
    for batch_idx in range(test_batches):
        fragment_batch = test_core_fragment[batch_idx * args['batch_size']:min((batch_idx + 1) * args['batch_size'],
                                                                               len(test_source_fragment_lens))]
        core_fragment_data = utils.EncodeCoreFragmentBatch(fragment_batch)
        growing_site = torch.tensor(test_growing_site[
                                    batch_idx * args['batch_size']:min((batch_idx + 1) * args['batch_size'],
                                                                       len(test_source_fragment_lens))]).long().to(
            args['device'])

        # Construct source and target for RNN
        data_source = test_source_fragment[batch_idx * args['batch_size']:min((batch_idx + 1) * args['batch_size'],
                                                                              len(test_source_fragment_lens))].to(
            args['device'])
        data_target = test_target_fragment[batch_idx * args['batch_size']:min((batch_idx + 1) * args['batch_size'],
                                                                              len(test_target_fragment_lens))].to(
            args['device'])
        lengths = test_target_fragment_lens[batch_idx * args['batch_size']:min((batch_idx + 1) * args['batch_size'],
                                                                               len(test_target_fragment_lens))]
        lengths = [i - 1 for i in lengths]  # the length of prev and nexts should reduce by 1
        prevs = data_source[:, :-1]
        nexts = data_target[:, 1:]
        nexts = rnn_utils.pack_padded_sequence(nexts, lengths, enforce_sorted=False,
                                               batch_first=True)  # return（batch_size,max_length）
        nexts, _ = rnn_utils.pad_packed_sequence(nexts, batch_first=True)
        outputs,hidden,_ = model(core_fragment_data, growing_site, data_source, prevs, lengths, hidden=None)

        # loss
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
    plt.savefig(args['out_path']+'/tl-loss.png',dpi=200)
    plt.show()
    return 

def smi_filter(smi):
    atoms = ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's','H', 'F', 'I', 'Cl','Br']
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if mol.GetNumAtoms() < 50:
            count = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in atoms:
                    count += 1
            if count == len(mol.GetAtoms()):
                return True
    return False

# encode fragment pairs collected from literatures
data = pd.read_csv('./data/yxy_all/yxy_substructure.csv',header=None)
final_pairs = []
for i in range(data.shape[0]):
    if smi_filter(data.iloc[i,0]) and smi_filter(data.iloc[i,1]):
        core_fragment = re.sub(r"/|\\", "", data.iloc[i,2])
        subsource = re.sub(r"/|\\", "", data.iloc[i,3])
        subtarget = re.sub(r"/|\\", "", data.iloc[i,4])
        if len(subsource[1:]) <= 60 and len(subsource[1:]) >= 3:
            if len(subtarget[1:]) <= 60 and len(subtarget[1:]) >= 3:
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
                    final_pairs.append([core_fragment, subsource, subtarget])
final_pairs = [list(x) for x in set(tuple(x) for x in final_pairs)]  # 嵌套列表去重
random.shuffle(final_pairs)
print('pairs_num: ', len(final_pairs))    # 4517

core_fragment_list, growing_site_list, subsource_data, subtarget_data = Encode_data(final_pairs)

args = {'model': './result/prior-15.pth','batch_size':256,'lr':0.0005,'epochs':25,'split':0.8,'device':'cuda:2','out_path':'./result2'}

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
model.load_state_dict(torch.load(args['model']))
# 对预训练模型进行微调，除了最后一个全连接层外预训练模型的所有参数均被冻结，在梯度反传的过程中这部分参数不会被更新，只有最后一个线性层的参数可以被调整。
# 通过微调的方式为模型赋予化学家的优化经验，从而增加合理优化分子的生成概率

for param in model.parameters():   # 冻结除最后一层线性层以外的所有参数
    param.requires_grad = False
for param in model.decoder_rnn.parameters():    # 解冻最后一层的参数
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(),lr=args['lr'])

x_axis = []
train_loss_list = []
test_loss_list = []
train_loss_avg = 0
test_loss_avg = 0
plot_every = 1

for epoch in range(1, epochs + 1):
    train_loss = training(args, model, optimizer, train_core_fragment, train_growing_site, train_source_fragment,
                          train_source_fragment_lens, train_target_fragment, train_target_fragment_lens)
    test_loss = testing(args, model, test_core_fragment, test_growing_site, test_source_fragment,
                        test_source_fragment_lens, test_target_fragment, test_target_fragment_lens)
    print('epoch: {}, train_loss: {:.2f}, test_loss: {:.2f}'.format(epoch,train_loss,test_loss))
    train_loss_avg += train_loss
    test_loss_avg += test_loss

    if epoch % plot_every == 0:
        train_loss_list.append(train_loss_avg / plot_every)
        test_loss_list.append(test_loss_avg / plot_every)
        train_loss_avg = 0
        test_loss_avg = 0
        x_axis.append(epoch)
    torch.save(model.state_dict(), args['out_path'] + '/tl-{:02d}.pth'.format(epoch))

#Draw Loss
plot_loss(x_axis,train_loss_list,test_loss_list)

#Output training log
x_axis = np.array(x_axis)
train_loss_list = np.array(train_loss_list)
test_loss_list = np.array(test_loss_list)
out_log = np.vstack((x_axis,train_loss_list,test_loss_list))
out_log = np.transpose(out_log)
np.savetxt(args['out_path']+'/tl-log.txt', out_log,fmt='%f',delimiter=',')





