import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
from torch.autograd import Variable
drug_encoding = 'CNN'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_concat(nn.Sequential):
    def __init__(self, out_dim, encoding,  **config):
        super(CNN_concat, self).__init__()
        if encoding == 'drug':
            in_ch = [64] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((64, 200))
            # n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, out_dim)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class Encoder(torch.nn.Module):  # Joining together
    def __init__(self, vector_size,dim_out,drop_out_rating,n_layers):
        super(Encoder, self).__init__()
        self.vector_size = vector_size
        self.l1 = torch.nn.Linear(self.vector_size,dim_out)
        self.att2s = torch.nn.ModuleList([EncoderLayer(dim_out) for _ in range(n_layers)])
        self.dr = torch.nn.Dropout(drop_out_rating)

    def forward(self, X):
        X = F.relu(self.dr(self.l1(X)))
        for att2 in self.att2s:
           X = att2(X)
        return X

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim):  #
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, 4)   #
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)   
        X = self.AN1(output + X)

        output = self.l1(X)  
        X = self.AN2(output + X)

        return X

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads  #
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim  #
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)   
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)   
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)   
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)  

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)  

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class RGCN(torch.nn.Module):
    def __init__(self, N_relation, in_features, out_features, bias=True):
        super(RGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N_relation = N_relation
        self.weight = torch.nn.Parameter(torch.FloatTensor(N_relation, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mapping = torch.nn.Linear(self.N_relation*self.out_features, self.out_features)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.matmul(adj, input)
        output = torch.matmul(input, self.weight)
        # output = self.mapping(output.view(-1, self.N_relation*self.out_features))
        output = torch.sum(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MHDSC(torch.nn.Module):
    def __init__(self, fea_dim, input_dim, n_head_gc ,n_clusters,drop_out_rating):
        super(MHDSC, self).__init__()
        self.d_k = input_dim// n_head_gc
        self.d_v = n_clusters
        self.n_heads = n_head_gc
        self.W_Q = torch.nn.Linear(fea_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc1 = torch.nn.Linear(self.n_heads * self.d_v, input_dim, bias=False)
        self.fc2 = torch.nn.Linear(input_dim,input_dim, bias=False)
        self.dr = torch.nn.Dropout(drop_out_rating)

    def forward(self, new_fe, features):
        Q = self.W_Q(new_fe).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(features).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, Q.transpose(-1, -2))
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = F.relu(torch.matmul(attn, V))

        output = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = F.relu(self.fc1(output))+features
        output = F.relu(self.fc2(output))+output
        return output,attn,context

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def generate_config(drug_encoding=None,
                    result_folder="./result/",
                    input_dim_drug=1024,
                    hidden_dim_drug=256,
                    cls_hidden_dims=[1024, 1024, 512],
                    mlp_hidden_dims_drug=[1024, 256, 64],
                    batch_size=256,
                    train_epoch=10,
                    test_every_X_epoch=20,
                    LR=1e-4,
                    mpnn_hidden_size=50,
                    mpnn_depth=3,
                    cnn_drug_filters=[32, 64, 96],
                    cnn_drug_kernels=[4, 6, 8],
                    num_workers=0,
                    cuda_id=None,
                    ):
    base_config = {'input_dim_drug': input_dim_drug,
                   'hidden_dim_drug': hidden_dim_drug,  # hidden dim of drug
                   'cls_hidden_dims': cls_hidden_dims,  # decoder classifier dim 1
                   'batch_size': batch_size,
                   'train_epoch': train_epoch,
                   'test_every_X_epoch': test_every_X_epoch,
                   'LR': LR,
                   'drug_encoding': drug_encoding,
                   'result_folder': result_folder,
                   'binary': False,
                   'num_workers': num_workers,
                   'cuda_id': cuda_id
                   }
    if not os.path.exists(base_config['result_folder']):
        os.makedirs(base_config['result_folder'])
    if drug_encoding == 'Morgan':
        base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug  # MLP classifier dim 1
    elif drug_encoding == 'CNN':
        base_config['cnn_drug_filters'] = cnn_drug_filters
        base_config['cnn_drug_kernels'] = cnn_drug_kernels
    # raise NotImplementedError
    elif drug_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct drug encoding available!")

    return base_config

config = generate_config(drug_encoding = drug_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 5,
                         LR = 0.001,
                         batch_size = 128,
                         hidden_dim_drug = 700,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3
                        )
