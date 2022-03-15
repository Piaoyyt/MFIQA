from MFace.net_struture.metricbone.metrics_bone import *

metrics_dict = {
    "add_margin": AddMarginProduct,
    "arc_margin": ArcMarginProduct,
    "sphere": SphereProduct,
}

metrics_paradict = {
    "arc_margin": {"in_features": 512, "out_features": 10517, 's': 30.0, 'm': 0.50, "easy_margin": False},
    "add_margin": {"in_features": 512, "out_features": 10517, 's': 30.0, 'm': 0.50},
    "sphere": {"in_features": 512, "out_features": 10517, 'm': 4},

}

class KeyError(BaseException):
    def __init__(self, key):
        self.key = key
    def __str__(self):
        return f"Metricbone Parameters Keys not exist!!!: {self.key}"
def load_metricsbone(Metric_name: str, **kwargs):
    for k in kwargs:
        if k not in metrics_paradict[Metric_name]: continue
        print(f"[{Metric_name}]-Metric Parameter {k} update : {metrics_paradict[Metric_name][k]} → {kwargs[k]}")
        metrics_paradict[Metric_name][k] = kwargs[k]
    Metricbone = metrics_dict[Metric_name](**metrics_paradict[Metric_name])

    return Metricbone
if __name__ == "__main__":
    backbone = load_metricsbone("add_margin", **{"in_features": 512, "out_features": 10517, 's': 30.0, 'm': 0.50, "easy_margin": False})