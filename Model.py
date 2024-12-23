
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import utils
import numpy as np
import rdkit
from rdkit import Chem
import random

class MolOpt(nn.Module):
    def __init__(self,
                state_dim=64,
                n_node=60,
                n_steps=5, 
                encoder_hidden=128,
                decoder_hidden=512,
                num_layers=3,
                dropout=0.2,
                vocab_size=37):
        super(MolOpt, self).__init__()

        #encoder1--GGNN
        self.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        self.state_dim = state_dim
        self.n_node = n_node
        self.n_steps = n_steps
        self.encoder = GGNN(state_dim=self.state_dim, n_node=self.n_node, n_steps=self.n_steps, device=self.device)
        
        #encoder2--RNN
        self.vocab_size = self.input_size = self.output_size = vocab_size
        self.hidden_size = self.encoder_hidden = encoder_hidden
        self.hidden_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size,padding_idx=0)##padding
        self.encoder_rnn = nn.GRU(self.input_size, self.encoder_hidden,1,batch_first=True)

        #decorder--RNN
        self.num_layers = num_layers
        self.decoder_hidden = decoder_hidden
        self.linear1 = nn.Linear(self.state_dim+self.encoder_hidden,self.decoder_hidden)
        self.decoder_rnn = nn.GRU(self.input_size+self.encoder_hidden, self.decoder_hidden,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear2 = nn.Linear(self.decoder_hidden, self.output_size)


    def forward(self, core_fragment_data,growing_site,source_fragment_data,x_input, lengths, hidden=None,hidden2=None):
        if hidden == None:
            emb_node = self.graphGGNNEncoder(core_fragment_data)
            hidden1 = torch.gather(emb_node, 1, growing_site.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, emb_node.size(-1)))
            hidden1 = hidden1.squeeze(1)

            hidden2 = self.embedding(source_fragment_data)
            _, hidden2 = self.encoder_rnn(hidden2, None)
            hidden2 = hidden2[-1:]#hidden of last step
            hidden2 = torch.cat(hidden2.split(1), dim=-1).squeeze(0)
            hidden = torch.cat((hidden1,hidden2),dim=-1)
            hidden = F.relu(self.linear1(hidden))# (batch_size, hidden_size)
            hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)#(n_layers, batch_size, hidden_size)

            x_emb = self.embedding(x_input)
            hidden2 = hidden2.unsqueeze(1).repeat(1, x_emb.size(1), 1)##repeat the matrix to a time step (max_length) of 120
            x_input = torch.cat([x_emb, hidden2], dim=-1)
            x_input = pack_padded_sequence(x_input, lengths, enforce_sorted=False,batch_first=True)##return（batch_size,max_length）
            x_input, hidden = self.decoder_rnn(x_input, hidden)
            out, _ = pad_packed_sequence(x_input, batch_first=True)
            out = self.linear2(out)
            return out, hidden, hidden2
        else:
            hidden = torch.tensor(hidden).to(self.device)
            x_emb = self.embedding(x_input)
            x_input = torch.cat([x_emb, hidden2], dim=-1)
            x_input = pack_padded_sequence(x_input, lengths, enforce_sorted=False,batch_first=True)## return（batch_size,max_length）
            x_input, hidden = self.decoder_rnn(x_input, hidden)
            out, _ = pad_packed_sequence(x_input, batch_first=True)
            out = self.linear2(out)
            return out, hidden

    def graphGGNNEncoder(self,core_fragment_data):
        #encode the core fragment graph
        nodes, edges = core_fragment_data['node'], core_fragment_data['adj']
        nodes=torch.tensor(nodes).float().to(self.device)
        edges=torch.from_numpy(edges).float().to(self.device)
        padding = torch.zeros(nodes.size(0), self.n_node, self.state_dim-13).float().to(self.device)#self.n_node is the maximum number of node，13 is the number of node type 
        #initialize_state
        init_state = torch.cat((nodes, padding), 2)
        #the state after propogation
        emb_node=self.encoder(init_state, nodes, edges)
        return emb_node
    
    def optimization(self,n_batch,core_fragment,source_fragment,max_length=60,T=1):
        self.device=next(self.parameters()).device
        with torch.no_grad():            
            # initialize sequence
            seq = torch.tensor(np.zeros((n_batch,max_length+2))).long().to(self.device)
            seq[:,0] = 1

            #end mask
            end_smiles_list = [False for _ in range(n_batch)]

            #initialize hidden
            lens=[1]*n_batch
            core_fragment,growing_site=utils.ProcessCoreFragmentBatch([core_fragment]*n_batch)
            growing_site=torch.tensor(growing_site).long().to(self.device)
            core_fragment_data=utils.EncodeCoreFragmentBatch(core_fragment)

            source_fragment=Chem.MolToSmiles(Chem.MolFromSmiles(source_fragment))#standardize source fragment
            source_fragment=source_fragment[1:] #remove '*'
            source_fragment_data= utils.encode([source_fragment]*n_batch,pad_size=37)
            source_fragment_data= source_fragment_data[0].long().to(self.device)
            _, hidden, hidden2 = self.forward(core_fragment_data,growing_site,source_fragment_data,seq[:,0].unsqueeze(1),lens,hidden=None)
            hidden = hidden.detach().cpu().numpy().tolist()

            starts = seq[:, 0].unsqueeze(1)

            #generate word step by step
            for i in range(1,max_length+1):
                output,hidden = self.forward(core_fragment_data,growing_site,source_fragment_data,starts,lens,hidden=hidden,hidden2=hidden2)
                hidden=hidden.detach().cpu().numpy().tolist()
                # probabilities
                probs = [F.softmax(o/T, dim=-1) for o in output]                
                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):#j represent the jth object
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == 2:#2 is the word for ending
                            end_smiles_list[j] = True#update ending mask

                        seq[j][i] = top_elem

                starts = torch.tensor(ind_tops, dtype=torch.long,
                                      device=self.device).unsqueeze(1)# update starts
                    
            new_smiles_list=seq[:,:]
            return new_smiles_list

#********* https://github.com/chingyaoc/ggnn.pytorch/blob/master/model.py **********#
class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat
        return output

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self,state_dim,n_node,n_steps, device):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.annotation_dim = 13
        self.n_edge_types = 4
        self.n_node = n_node
        self.n_steps = n_steps
        self.mask_null = True
        self.device = device
        self.possible_atom_types = ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's','H', 'F', 'I', 'Cl','Br']
        self.possible_bond_types = np.array([Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE])

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh()
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)
        output = self.out(join_state)
        #output = output.sum(2)
        return output