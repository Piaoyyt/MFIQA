import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
# import torch.functional as F
# # from config import device
# import  cv2
from PIL import Image
# from data_gen import data_transforms
from torchvision import transforms
# from pyIUA.dataset.mask_img_utils import detect_face
import numpy as np
import os
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval()
model = model.to(device)


def get_embedding(file_path, model):
    img = Image.open(file_path).convert('RGB')
    #resize = transforms.Resize((112, 112))
    # img = resize(img)
    transform = data_transforms['val']
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor,dim=0)

    img_device = img_tensor.to(device)
    output_feature = model(img_device)
    return np.squeeze(output_feature.detach().cpu().numpy())
mode = "Test"
base_embedding = np.load(f".\\Facenet_pytorch_{mode}_template_embedding.npy",allow_pickle=True).item()
file_path =r"E:\My_project\dataset\WDFace\WD_Input"+"\\"+mode
annotation = {}
annotation_method = "facenet_cosine_distance"
for file in os.listdir(file_path):
    pictures_path = os.path.join(file_path, file)
    base_feature = base_embedding[file]
    base_feature = torch.from_numpy(base_feature)
    for picture_name in os.listdir(pictures_path):
        img_path = os.path.join(pictures_path, picture_name)
        # img = Image.open(img_path)
        # if not detect_face(img):#如果检测不到人脸，就将可用性置为0
        #     annotation[file + "\\" + picture_name] = {}
        #     annotation[file + "\\" + picture_name][annotation_method] = 0.0
        # else:
        img_embedding = get_embedding(img_path, model= model)

        img_embedding =torch.from_numpy(img_embedding)
        # similarity = torch.sqrt(torch.sum((img_embedding-base_feature)**2))
        similarity = np.dot(img_embedding/np.linalg.norm(img_embedding), base_feature/np.linalg.norm(base_feature))
        annotation[file+"\\"+picture_name] = {}
        annotation[file+"\\"+picture_name][annotation_method] = round(float(similarity), 3)
np.save(f"Facenet_{mode}_quality_cosine_labels.npy", annotation)
# f1 = get_embedding("Abdel_Nasser_Assidi_0001.jpg", model =model)
# f2 = get_embedding("Abdel_Nasser_Assidi_0002.jpg", model = model)
# similarity = np.dot(f1/np.linalg.norm(f1), f2/np.linalg.norm(f2))
