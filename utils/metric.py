import numpy as np
from sklearn import metrics

def get_test_metrics(total_pred, total_label):
    y_pred = np.concatenate(total_pred)
    y_true = np.concatenate(total_label)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # ap
    ap = metrics.average_precision_score(y_true, y_pred)

    # Accuracy (Acc)
    accuracy = metrics.accuracy_score(y_true, np.round(y_pred))

    return {'auc': auc, 'eer': eer, 'ap': ap, 'acc': accuracy}

if __name__=="__main__":
    # Example usage
    total_pred = [np.array([0.2, 0.8, 0.6]), np.array([0.4, 0.7, 0.1])]
    total_label = [np.array([0, 1, 1]), np.array([1, 1, 0])]

    metrics_dict = get_test_metrics(total_pred, total_label)

    # Access the metrics
    print("AUC:", metrics_dict['auc'])
    print("EER:", metrics_dict['eer'])
    print("AP:", metrics_dict['ap'])
    print("Accuracy:", metrics_dict['acc'])
    print("TDR at FDR of 0.01%:", metrics_dict['tdr_001'])
    print("TDR at FDR of 0.1%:", metrics_dict['tdr_01'])
    print("pAUC for FPR range [0, 0.1]:", metrics_dict['pauc_01'])