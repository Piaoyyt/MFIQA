from MFace.net_struture.backbone.resnet import *
from MFace.net_struture.net_sphere import sphere20a
from MFace.net_struture.backbone.ir_resnet_se import Backbone

backbone_dict = {
    "resnet18": resnet_face18,
    "resnet34": resnet_face34,
    "resnet50": resnet_face50,
    "resnet101": resnet_face101,
    "sphere20a": sphere20a,
    "resnet50_IR": Backbone,
    "resnet50_IR_SE": Backbone,

}

backbone_paradict = {
    "resnet18": {'use_se': False},
    "resnet34": {'use_se': False},
    "resnet50": {'use_se': False},
    "resnet101": {'use_se': False},
    "sphere20a": {},
    "resnet50_IR": {"num_layers": 50, "drop_ratio": 0.6, "mode": "ir"},
    "resnet50_IR_SE": {"num_layers": 50, "drop_ratio": 0.6, "mode": "ir_se"},
}
class KeyError(BaseException):
    def __init__(self, key):
        self.key = key
    def __str__(self):
        return f"Backbone Parameters Keys not exist!!!:{self.key}"

def load_backbone(Backbone_name: str, **kwargs):
    for k in kwargs:
        if k not in backbone_paradict[Backbone_name]: continue
        # raise KeyError(k)
        print(f"[{Backbone_name}]-Backbone Parameter {k} update : {backbone_paradict[Backbone_name][k]} → {kwargs[k]}")
        backbone_paradict[Backbone_name][k] = kwargs[k]
    backbone = backbone_dict[Backbone_name](**backbone_paradict[Backbone_name])

    return backbone

if __name__ == "__main__":
    backbone = load_backbone("resnet50_IR", **{'use_se': False, "num_layers": 50, "drop_ratio": 0.6})
    # print(backbone)
