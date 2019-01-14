import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    return acc, prc, rc, f1, auc_


def eval(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        all_predict, all_label = list(), list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code).cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code).detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_pred=all_predict, y_true=all_label)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        return acc, prc, rc, f1, auc_