
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
BACKBONES = {
            'Inception-Resnet':InceptionResnetV1
}


def get_backbone(name,**kwargs):
    return BACKBONES[name](**kwargs)