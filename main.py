import sqlite3   
import time
from sklearn.model_selection import StratifiedKFold
from HMGRL import *
from model import *
from accuracy import *
from utils import *
import warnings
import argparse
warnings.filterwarnings("ignore")

feature_list = ["smile","target","enzyme"]
parser = argparse.ArgumentParser("MSAGC")
parser.add_argument('--dataset', choices=['dataset1', 'dataset2'], default='dataset2', help='dataset to use')
parser.add_argument('--task', choices=['task1','task2','task3'], default='task1', help='dataset to use')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--ep', type=int, default=120, help='epoch number')
parser.add_argument('--n_hop', type=int, default=0, help='RaSE propagate steps')
parser.add_argument('--learn_rating', type=float, default= 0.00002, help='learning rate')
parser.add_argument('--weight_decay_rate', type=float, default=0.0001, help='weight_decay')
parser.add_argument('--batch_size', type=int, default=512, help='number of batch size')
parser.add_argument('--g_d', type=int, default=500, help='dimension of RaGSEs')
parser.add_argument('--hd1', type=int, default=1000, help='output dimension of hidden embedding')
parser.add_argument('--hd2', type=int, default=1000, help='output dimension of hidden attributes')
parser.add_argument('--layers',type=int, default=1, help='Transformer encoder')
parser.add_argument('--cross_ver_tim', type=int, default=5, help='cross_ver_tim')
parser.add_argument('--heads', type=int, default=4, help='n_head_gc')
parser.add_argument('--clusters', type=int, default=200, help='n_clusters')
parser.add_argument('--gc_weight', type=float, default=0.1, help='n_clusters')
parser.add_argument('--dr', type=float, default=0.3, help='dropout_rate')
args = parser.parse_args()

def main():
    if args.dataset=='dataset1':
        conn = sqlite3.connect("dataset1/event.db")
        df_drug = pd.read_sql('select * from drug;',conn)   
        extraction = pd.read_sql('select * from extraction;', conn)
        drug_smiles = pd.read_csv("dataset1/data.csv")
        mechanism = extraction['mechanism']
        action = extraction['action']
        drugA = extraction['drugA']
        drugB = extraction['drugB']
        drug_smile = drug_smiles['smile']
        drug_coding = drug_smile_coding(drug_smile, drug_encoding)
        new_label, drugA, drugB, event_num, X_vector, X_three_vector, DDI_edge, N_three_attribute,tensor_tempvec_multi = prepare(df_drug,feature_list,mechanism,action,drugA,drugB,args)
    else:
        new_label = np.load("dataset2/new_label.npy")
        drugA =np.load("dataset2/newdrugA.npy")
        drugB =np.load("dataset2/newdrugB.npy")
        X_vector = np.load("dataset2/X_vector.npy")
        X_vector = torch.tensor(X_vector, dtype=torch.float)
        drug_coding = np.load("dataset2/drug_coding.npy")
        drug_coding = torch.tensor(drug_coding)
        DDI_edge = np.load("dataset2/DDI_edge.npy")
        event_num = np.unique(new_label).shape[0]
        X_three_vector = np.load("dataset2/X_three_vector.npy")
        X_three_vector = torch.tensor(X_three_vector, dtype=torch.float)
        adj1 = (gcnnormalization(X_vector[:,:1000])).reshape(1, 1000, 1000)
        adj2 = (gcnnormalization(X_vector[:,1000:2000])).reshape(1, 1000, 1000)
        adj3 = (gcnnormalization(X_vector[:,2000:])).reshape(1, 1000, 1000)
        tensor_tempvec_multi = torch.cat((adj1, adj2, adj3),dim=0)
        N_three_attribute = torch.tensor([2033,1589,285],dtype=torch.int)

    print("dataset len", len(new_label))
    start=time.time()
    result_all =cross_val(new_label,drugA, drugB,event_num, X_vector, X_three_vector, DDI_edge, drug_coding, N_three_attribute,tensor_tempvec_multi)
    print("time used:", (time.time() - start) / 3600)

def cross_val(label, drugA,drugB, event_num, X_vector,X_three_vector,DDI_edge, drug_coding, N_three_attribute,tensor_tempvec_multi):

    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])
    cross_ver_tim = args.cross_ver_tim
    skf = StratifiedKFold(n_splits=args.cross_ver_tim)

    if args.task == "task2" or "task3":
        temp_drug1 = [[] for i in range(event_num)]
        temp_drug2 = [[] for i in range(event_num)]
        for i in range(len(label)):
            temp_drug1[label[i]].append(drugA[i])  
            temp_drug2[label[i]].append(drugB[i])  
        drug_cro_dict = {}
        for i in range(event_num):
            for j in range(len(temp_drug1[i])):   
                drug_cro_dict[temp_drug1[i][
                    j]] = j % cross_ver_tim   
                drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
        train_drug = [[] for i in range(cross_ver_tim)]
        test_drug = [[] for i in range(cross_ver_tim)]
        for i in range(cross_ver_tim):
            for dr_key in drug_cro_dict.keys():
                if drug_cro_dict[dr_key] == i:
                    test_drug[i].append(dr_key)
                else:
                    train_drug[i].append(dr_key)

    cross_ver = 0

    for train_index, test_index in skf.split(DDI_edge, label):
        if args.task == "task1":
            y_train, y_test = label[train_index], label[test_index]
            ddi_edge_train, ddi_edge_test = DDI_edge[train_index], DDI_edge[test_index]

        if args.task == "task2":
            y_train = [];
            y_test = []
            ddi_edge_train = [];
            ddi_edge_test = [];
            for i in range(len(drugA)):
                if (drugA[i] in np.array(train_drug[cross_ver])) and (
                        drugB[i] in np.array(train_drug[cross_ver])):
                    y_train.append(label[i])
                    ddi_edge_train.append(DDI_edge[i])

                if (drugA[i] not in np.array(train_drug[cross_ver])) and (
                        drugB[i] in np.array(train_drug[cross_ver])):
                    y_test.append(label[i])
                    ddi_edge_test.append(DDI_edge[i])

                if (drugA[i] in np.array(train_drug[cross_ver])) and (
                        drugB[i] not in np.array(train_drug[cross_ver])):
                    y_test.append(label[i])
                    ddi_edge_test.append(DDI_edge[i])

        if args.task == "task3":
            y_train = [];
            y_test = []
            ddi_edge_train = [];
            ddi_edge_test = [];
            for i in range(len(drugA)):
                if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                    y_train.append(label[i])
                    ddi_edge_train.append(DDI_edge[i])

                if (drugA[i] not in np.array(train_drug[cross_ver])) and (
                        drugB[i] not in np.array(train_drug[cross_ver])):
                    y_test.append(label[i])
                    ddi_edge_test.append(DDI_edge[i])

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        ddi_edge_train = np.array(ddi_edge_train, dtype=int)
        ddi_edge_test = np.array(ddi_edge_test, dtype=int)

        adj, n_drugs = adj_Heter_gene(ddi_edge_train, X_vector, event_num, y_train)
        model = HMGRL(len(X_vector[0]) * 2, event_num, n_drugs, N_three_attribute, args)

        print("train len", len(y_train))
        print("test len", len(y_test))

        pred_score = HMGRL_train(model, y_train, y_test,
                                 event_num, X_vector, X_three_vector, adj, ddi_edge_train, ddi_edge_test,
                                 drug_coding, tensor_tempvec_multi,
                                 args)

        pred_type = np.argmax(pred_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

        y_true = np.hstack((y_true, y_test))
        result_all_now, _ = evaluate(pred_type, pred_score, y_test, event_num)
        print(result_all_now)
        cross_ver = cross_ver + 1


        # break

    result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num)
    print('The final prediction accuracies:')
    print(result_all)


main()





