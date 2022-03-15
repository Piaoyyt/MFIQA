import  pandas as pd
import  numpy as np

def calculate_correlation(first, second,file_path, kind="spearman",) -> float:
    data = pd.DataFrame({'first': first,
                         'second': second})
    if kind == "rmse":
        first = np.array(first).reshape(1, -1)
        second = np.array(second).reshape(1, -1)
        print(first[0][0:10])
        print(second[0][0:10])
        print(np.mean(np.square(first[0] - second[0])))
        rmse = round(np.sqrt(np.mean(np.square(first[0] - second[0]))), 3)
        with open(file_path, 'a+')  as f:
            f.write("RMSE-----{}".format(rmse))
        print("calculating rmse:{}".format(rmse))

        return rmse
    elif kind == "spearman":
        result = round(float(data.corr("spearman")["first"][1]), 3)
        with open(file_path, 'a+')  as f:
            f.write("Spearman Correlation-----{}".format(result))
        print("calculating spearman correlation coefficients:{}".format(result))
        return result
    elif kind == "kendall":
        result = round(float(data.corr("kendall")["first"][1]), 3)
        with open(file_path, 'a+')  as f:
            f.write("Kendall Correlation-----{}".format(result))
        print("calculating kendall correlation coefficients:{}".format(result))
        return result
    else:
        result = round(float(data.corr("pearson")["first"][1]), 3)
        with open(file_path, 'a+')  as f:
            f.write("Pearson Correlation-----{}".format(result))
        print("calculating pearson correlation coefficients:{}".format(result))
        return result
def correlation_analysis(label,predict,correlation_f):
    calculate_correlation(label,predict,file_path=correlation_f)
    calculate_correlation(label,predict,file_path=correlation_f, kind="kendall")
    calculate_correlation(label,predict,file_path=correlation_f, kind="pearson")
    calculate_correlation(label,predict,file_path=correlation_f, kind="rmse")