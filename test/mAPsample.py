import numpy as np

def mean_average_precision(y_true, y_pred):
    average_precisions = []
    # クラス単位でAPを計算
    for i in range(len(y_true)):
        sort_idx = np.argsort(y_pred[i])[::-1]
        y_true_sorted = y_true[i][sort_idx]

        cumsum = np.cumsum(y_true_sorted)
        recall = cumsum / np.max(cumsum)
        precision = cumsum / np.arange(1, 1 + y_true[i].shape[0])

        # 代表点
        points_x = np.arange(11) / 10
        points_y = np.zeros(points_x.shape[0])
        for i in range(points_x.shape[0]-1, -1, -1):
            points_y[i] = np.max(precision[recall >= points_x[i]])

        average_precision = np.mean(points_y)
        average_precisions.append(average_precision)
    return sum(average_precisions)/len(average_precisions)

if __name__ == "__main__":
    # 1クラスの場合
    # Correct or not
    y_true = [np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1], dtype=np.float32)]
    # Confidence score
    y_pred = [np.array([.96, .92, .89, .88, .84, .83, .80, .78, .74, .72], dtype=np.float32)]
    # クラス単位のAverage Precision
    class_ap = mean_average_precision(y_true, y_pred)
    print(class_ap)
