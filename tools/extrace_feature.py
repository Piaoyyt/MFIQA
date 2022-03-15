import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
import  cv2
from PIL import Image
# from .data_gen import data_transforms
from torchvision import transforms
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
# filename ="Aaron_Peirsol_0001.jpg"
# img = cv2.resize(img,dsize=[112,112])
def get_embedding(file_path, model):
    img = Image.open(file_path).convert('RGB')
    #resize = transforms.Resize((160, 112))
    #img = resize(img)
    transform = data_transforms['val']
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor,dim=0)

    img_device = img_tensor.to(device)
    output_feature = model(img_device)
    return np.squeeze(output_feature.detach().cpu().numpy())
base_embedding ={}
mode = "Test"
file_path =r"E:\My_project\dataset\WDFace\WD_Template"+"\\"+mode
for file in os.listdir(file_path):
    pictures_path = os.path.join(file_path, file)
    for picture_name in os.listdir(pictures_path):
        img_path = os.path.join(pictures_path, picture_name)
        img_embedding = get_embedding(img_path, model= model)
        base_embedding[file] = img_embedding
np.save(f"Facenet_pytorch_{mode}_template_embedding.npy", base_embedding)
# f1 = get_embedding("Abdel_Nasser_Assidi_0001.jpg", model =model)
# f2 = get_embedding("Abdel_Nasser_Assidi_0002.jpg", model = model)
# similarity = np.dot(f1/np.linalg.norm(f1), f2/np.linalg.norm(f2))
