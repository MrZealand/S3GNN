import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping
import warnings
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def result(pre,test,thre=0.5):

    TP = FN = FP = TN = 0
    for (i, j) in zip(pre,test):
        if j == 1:  
            if i == 1:  
                TP += 1  
            else:
                FN += 1 
        else:  
            if i == 1:
                FP += 1  
            else:
                TN += 1  

    print('TP:' + str(TP) + ',FN:' + str(FN) + ',FP:' + str(FP) + ',TN:' + str(TN))
    if TP != 0 and TN != 0 and FP != 0 and FN != 0:
        P = float(TP) / (TP + FP)
        R = float(TP) / (TP + FN)
        D = float(FP) / (TN + FP)
        print("Test Precision: %.3f%%" % (P * 100.0),"Test Recall: %.3f%%" % (R * 100.0),"Test Disturb: %.3f%%" % (D * 100.0))
        print("KS:",R-D)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    result(prediction,labels)

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def main(args):
    g_train,features_train,labels_train,train_mask,val_mask=Load_traindataset()
    g_test,features_test,labels_test,num_classes,test_mask=Load_testdataset()

    features_train = features_train.to(args['device'])
    features_test = features_test.to(args['device'])

    labels_train = labels_train.to(args['device'])
    labels_test=labels_test.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    # if args['hetero']:
    from model_hetero import S3GNN
    model = S3GNN(meta_paths=[['ap', 'pa'],
                            ['am', 'ma'],
                            ['ac', 'ca'],
                            ['ai','ia'],
                            ['al','la'],
                            ['avc','vca'],
                            ['avn','vna'],
                            ['aad','ada']],
                in_size=features_train.shape[1],
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g_train, features_train)
        if epoch==0:
            print('nan', np.any(np.isnan(features_train.cpu().numpy())))
            print(logits)
            print(logits.shape)
            train_labels=pd.DataFrame(logits[train_mask])
            train_labels.to_csv('train_labels.csv')
        loss = loss_fcn(logits[train_mask], labels_train[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels_train[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g_train, features_train, labels_train, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g_test, features_test, labels_test, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse

    from utils import setup

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('S3GNN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)