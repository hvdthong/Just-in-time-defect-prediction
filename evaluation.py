import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluation_metrics(y_true, y_pred):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    return acc, prc, rc


def eval(data, model):
    with torch.no_grad():
        model.eval()
        all_predict, all_label = list(), list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code).cpu().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code).detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
        all_predict = [1 if p >= 0.5 else 0 for p in all_predict]
        acc, prc, rc = evaluation_metrics(y_pred=all_predict, y_true=all_label)
        return acc, prc, rc
