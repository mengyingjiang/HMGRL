from layers import *
import torch
import torch.nn.functional as F

class HMGRL(torch.nn.Module):
    def __init__(self, input_dim0, R, n_drugs, N_Four_attribute, args):
        super(HMGRL, self).__init__()
        self.N_s = N_Four_attribute[0]
        self.N_t = N_Four_attribute[1]
        self.N_e = N_Four_attribute[2]
        self.targets_end = N_Four_attribute[0] + N_Four_attribute[1]
        self.N = n_drugs
        self.n_hop = args.n_hop
        self.dr = torch.nn.Dropout(args.dr)
        self.R = R

        self.RGCN = RGCN(R + 1, input_dim0 // 2, args.g_d)
        self.Mapping1 = torch.nn.Linear(args.g_d,args.g_d)
        self.Mapping2 = torch.nn.Linear(args.g_d,args.g_d)
        self.Mapping3 = torch.nn.Linear(args.g_d,args.g_d)

        self.AE1 = Encoder(2 * args.g_d, args.hd1, args.dr, args.layers)  # Joining together
        self.AE2 = Encoder(self.N*2, args.hd2, args.dr, args.layers)  # Joining together
        self.AE3 = Encoder(self.N*2, args.hd2, args.dr, args.layers)  # Joining together
        self.AE4 = Encoder(self.N*2, args.hd2, args.dr, args.layers)  # Joining together
        self.cnn_concat = CNN_concat(args.hd2, 'drug', **config)

        N_Fea = 4 * args.hd2 + args.hd1
        self.layer1 = MHDSC(N_Fea, N_Fea, args.heads, args.clusters,args.dr)
        self.layer2 = MHDSC(self.N_t, N_Fea, args.heads, args.clusters,args.dr)
        self.layer3 = MHDSC(self.N_e, N_Fea, args.heads, args.clusters,args.dr)
        self.layer4 = MHDSC(self.N_s, N_Fea, args.heads, args.clusters,args.dr)

        N_trans = 5
        self.l1 = torch.nn.Linear(N_trans * N_Fea, (N_trans * N_Fea + R))
        self.l2 = torch.nn.Linear((N_trans * N_Fea + R),R)

    def forward(self, label_graph, norm_adj, x_vector0, x_three_vector, ddi_edge, ddi_edge_mixup, lam, drug_coding):

        label_graph = F.relu(self.RGCN(x_vector0,label_graph))

        x_tar_all = x_enzy_all = x_sub_all = label_graph

        for i in range(self.n_hop):
            x_tar_all = torch.mm(norm_adj[0, :, :], x_tar_all)
            x_enzy_all = torch.mm(norm_adj[1, :, :], x_enzy_all)
            x_sub_all = torch.mm(norm_adj[2, :, :], x_sub_all)
        x_vector = x_tar_all+x_enzy_all+x_sub_all

        node_id = ddi_edge.T
        node_id_mixup = ddi_edge_mixup.T
        true_drug_s = x_vector0[node_id[0]]
        true_drug_t = x_vector0[node_id[1]]
        false_drug_s = x_vector0[node_id_mixup[0]]
        false_drug_t = x_vector0[node_id_mixup[1]]

        X_input_embeddings = lam * torch.cat([x_vector[node_id[0]], x_vector[node_id[1]]], dim=1) \
                           + (1 - lam) * torch.cat([x_vector[node_id_mixup[0]], x_vector[node_id_mixup[1]]], dim=1)

        x_smiles = lam * torch.cat([true_drug_s[:,0:self.N], true_drug_t[:,0:self.N]], dim=1) \
                           + (1 - lam) * torch.cat([false_drug_s[:,0:self.N], false_drug_t[:,0:self.N]], dim=1)

        x_targets = lam * torch.cat([true_drug_s[:,self.N:2*self.N], true_drug_t[:,self.N:2*self.N]], dim=1) \
                           + (1 - lam) * torch.cat([false_drug_s[:,self.N:2*self.N], false_drug_t[:,self.N:2*self.N]], dim=1)

        x_enzymes = lam * torch.cat([true_drug_s[:,2*self.N:], true_drug_t[:,2*self.N:]], dim=1) \
                           + (1 - lam) * torch.cat([false_drug_s[:,2*self.N:], false_drug_t[:,2*self.N:]], dim=1)

        smiles_string = lam * torch.cat([drug_coding[node_id[0]], drug_coding[node_id[1]]], dim=2) \
                + (1 - lam) * torch.cat([drug_coding[node_id_mixup[0]], drug_coding[node_id_mixup[1]]], dim=2)

        x_three_vector = lam * (x_three_vector[node_id[0]] + x_three_vector[node_id[1]]) + \
                         (1 - lam) * (x_three_vector[node_id_mixup[0]] + x_three_vector[node_id_mixup[1]])

        X1 = self.AE1(X_input_embeddings)
        X2 = self.AE2(x_smiles)
        X3 = self.AE3(x_targets)
        X4 = self.AE4(x_enzymes)
        X_smile = self.cnn_concat(smiles_string)
        X = torch.cat((X1, X2, X3, X4, X_smile), 1)

        X1,attn1,context1 = self.layer1(X, X)
        X2,attn2,context2 = self.layer2(x_three_vector[:, self.N_s:self.targets_end], X)
        X3,attn3,context3 = self.layer3(x_three_vector[:, self.targets_end:], X)
        X4,attn4,context4 = self.layer4(x_three_vector[:, :self.N_s], X)

        attn = torch.cat((attn1,attn2,attn3,attn4), 0)
        context = torch.cat((context1,context2,context3,context4), 0)

        X = torch.cat((X1,X2,X3,X4,X1+X2+X3+X4), 1)
        X = F.relu(self.dr(self.l1(X)))
        X = self.l2(X)

        return X, attn, context