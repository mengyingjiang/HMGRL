from numpy.random import seed
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from radam import RAdam
from layers import *
import warnings
from accuracy import *
warnings.filterwarnings("ignore")

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

class DDIDataset(Dataset):  # [59622, 3432], [59622]
    def __init__(self, y, z):
        self.len = len(z)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z)

    def __getitem__(self, index):
        return self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.len

class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()
        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, targets_a, adj_matrix, assignment, weight):

        layers,_, C = assignment.size()
        # assignment = F.normalize(assignment, dim=1)
        eye = torch.eye(C).to(device)

        sas = torch.matmul(torch.matmul(assignment.transpose(1, 2), adj_matrix), assignment)
        trsas = torch.sum((sas*eye), dim=[1, 2])

        sds = torch.matmul(assignment.transpose(1, 2), assignment)
        trsds = torch.sum((sds * eye), dim=[1, 2])

        loss_gc = -(trsas/trsds).mean()

        sdsfn = torch.sum(sds.pow(2), dim=[1, 2]).sqrt()

        sds_sdsfn = torch.mul(sdsfn.reshape(layers, 1, 1).pow(-1), sds)
        dis = (sds_sdsfn - eye/(C**0.5))
        loss_orth = torch.sum(dis.pow(2), dim=[1, 2]).sqrt().mean()
        loss = self.criteria1(X, targets_a) + weight * (loss_gc + loss_orth)

        del sas; sds; dis
        gc.collect()

        return loss

def HMGRL_train(model, y_train, y_test, event_num, X_vector,X_three_vector, adj, ddi_edge_train,ddi_edge_test, drug_coding,norm_adj,args):

    model_optimizer = RAdam(model.parameters(), lr=args.learn_rating, weight_decay=args.weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    ddi_edge_train = np.vstack((ddi_edge_train,np.vstack((ddi_edge_train[:,1],ddi_edge_train[:,0])).T))
    y_train = np.hstack((y_train, y_train))

    N_edges = ddi_edge_train.shape
    index = np.arange(N_edges[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    y_train = y_train[index]
    ddi_edge_train = ddi_edge_train[index]

    len_train = len(y_train)
    len_test = len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))

    train_dataset = DDIDataset(ddi_edge_train, np.array(y_train) )  # [59622, 3432], [59622]
    test_dataset = DDIDataset(ddi_edge_test, np.array(y_test) )  # [7453, 3432], [7453]

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    x_vector = X_vector.to(device)
    drug_coding = drug_coding.to(device)
    X_three_vector = X_three_vector.to(device)
    adj = adj.to(device)
    norm_adj = norm_adj.to(device)

    for epoch in range(args.ep):
        my_loss = my_loss1()
        running_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            train_edge, train_edge_labels = data
            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(train_edge_labels.size()[0]).to(device)
            train_edge_labels = train_edge_labels.type(torch.int64).to(device)
            targets_a, targets_b = train_edge_labels, train_edge_labels[index]
            train_edge = torch.tensor(train_edge, dtype=torch.long)
            train_edge = train_edge.to(device)
            train_edge_mixup = train_edge[index, :]
            model_optimizer.zero_grad()
            X, attn, context = model(adj,norm_adj,x_vector, X_three_vector, train_edge, train_edge_mixup, lam, drug_coding)
            loss = lam*my_loss(X, targets_a, attn, context, args.gc_weight)\
                   +(1-lam)*my_loss(X, targets_b, attn, context, args.gc_weight)
            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()
            gc.collect()
        #
        model.eval()
        testing_loss = 0.0
        # pre_score = np.zeros((0, event_num), dtype=float)
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader, 0):
                test_edge, test_edge_labels = data
                test_edge = torch.tensor(test_edge, dtype=torch.long)
                test_edge = test_edge.to(device)
                lam = 1
                test_edge_labels = test_edge_labels.type(torch.int64).to(device)
                X,_,_ = model(adj, norm_adj, x_vector, X_three_vector, test_edge,test_edge,lam, drug_coding)
                # pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))
                loss = torch.nn.functional.cross_entropy(X, test_edge_labels)
                testing_loss += loss.item()
                del test_edge
                gc.collect()

        # pred_type = np.argmax(pre_score, axis=1)
        # result_all_now, _ = evaluate(pred_type, pre_score, y_test, event_num)
        # print(result_all_now)
        print('epoch [%d] trn_los: %.6f tet_los: %.6f ' % (
            epoch + 1, running_loss / len_train, testing_loss / len_test))

    pre_score = np.zeros((0, event_num), dtype=float)

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            test_edge, _= data
            test_edge = torch.tensor(test_edge, dtype=torch.long)
            test_edge = test_edge.to(device)
            lam = 1
            X,_,_ = model(adj, norm_adj, x_vector, X_three_vector, test_edge, test_edge,lam, drug_coding)
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))

    del model
    del X
    del model_optimizer
    del train_loader
    del test_loader
    del train_dataset
    del test_dataset
    gc.collect()

    return pre_score
