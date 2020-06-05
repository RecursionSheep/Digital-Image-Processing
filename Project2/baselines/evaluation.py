def single_label_accuracy(preds, labels, acc_result):
    if isinstance(preds, __import__("torch").Tensor):
        num_class = preds.size(1)
        preds_index = preds.max(dim=1)[1]
        size = labels.size(0)
    else:
        num_class = 10
        preds_index = preds
        size = labels.shape[0]
    if acc_result is None:
        acc_result = [{"TP": 0, "FN": 0, "FP": 0} for i in range(num_class)]
    for i in range(size):
        it_is, should_be = int(preds_index[i]), int(labels[i])
        if it_is == should_be:
            acc_result[it_is]["TP"] += 1
        else:
            acc_result[it_is]["FP"] += 1
            acc_result[should_be]["FN"] += 1
    return acc_result

def generate_metric(acc_result):
    size = len(acc_result)
    precision, recall, f1, total = [], [], [], []
    total = {"TP": 0, "FN": 0, "FP": 0}
    for item in acc_result:
        p, r, f = get_prf(item)
        precision.append(p)
        recall.append(r)
        f1.append(f)
        for key in item:
            total[key] += item[key]
    micro_precision, _, _ = get_prf(total)
    macro_precision = sum(precision) / size
    macro_recall = sum(recall) / size
    macro_f1 = sum(f1) / size
    return {
        "micro_precision": round(micro_precision, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4)
    }

def get_prf(item):
    if item["TP"] == 0:
        p, r, f = (1.0, 1.0, 1.0) if item["FP"] == 0 and item["FN"] == 0 else (0.0, 0.0, 0.0)
    else:
        p = item["TP"] / (item["TP"] + item["FP"])
        r = item["TP"] / (item["TP"] + item["FN"])
        f = 2 * p * r / (p + r)
    return p, r, f