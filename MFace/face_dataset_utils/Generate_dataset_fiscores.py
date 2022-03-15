import os, sys
from tqdm import tqdm
project_path = "\\".join(os.getcwd().split("\\")[:-2]) if not sys.platform == "linux" \
    else "/".join(os.getcwd().split("/")[:-2])
sys.path.append(project_path)
print(project_path)
import logging
import numpy as np

from Evaluation.FIQA_model.FIQA import FIQA

from project_path.get_project_path import project_path
from torch.utils.data import Dataset,DataLoader
logger = logging.getLogger('globals')
class ImgPath_Dataset(Dataset):

    def __init__(self, root_path):
        self.root_path = root_path.replace("\\", "/") if sys.platform=="linux" else root_path.replace("/", "\\")
        self.imgs_path = []
        for human in os.listdir(self.root_path):
            for img_name in os.listdir(os.path.join(self.root_path, human)):
                self.imgs_path.append(os.path.join(self.root_path, human, img_name))
        print("The dataset contains totally %d pictures" % (len(self.imgs_path)))
    def __getitem__(self, item):
        return self.imgs_path[item]
    def __len__(self):
        return  len(self.imgs_path)

def dataset_score():
    dataset_path = {"My dataset": [project_path + "\\dataset\\human_face", 30000], \
                    "LFW": [project_path + "\\dataset\\human_face_original", 6000], \
                    "VGGface": [project_path + "\\dataset\\Public_dataset\\vggface2_test_alignment", 30000], \
                    "Celeba": [project_path + "\\dataset\\Public_dataset\\CelebA-Dataset", 30000], \
                    "CASIA": [project_path + "\\dataset\\Public_dataset\\CASIA_Face_alignment", 3000], \
                    "CASIA-Webface": [project_path + "\\dataset\\Public_dataset\\CASIA_WebFace_alignment", 30000], \
                    "WDFace_Train": [project_path + "\\dataset\\WDFace\\WD_Input\\Train", 30000], \
                    "CASIA_Webface_distor": [project_path + "\\dataset\\Public_dataset\\Distor_datasets\\CASIA-Webface",
                                             30000]
                    }
    fiqa_score_root = project_path + "\\Evaluation\\FIQA_score_store\\"
    if sys.platform == "linux":
        fiqa_score_root = fiqa_score_root.replace("\\", "/")

    dataset = ImgPath_Dataset(root_path = dataset_path["LFW"][0])
    FIQA_model = FIQA(method_name = "FIQA_mine_v2")
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)
    ans = dict()
    for data in tqdm(dataloader):
        img_name = '/'.join(data[0].split("/")[-2:])
        score = FIQA_model.predict(data[0])
        ans[img_name] = score
        print(score)
    np.save(os.path.join(fiqa_score_root, "CASIA_Webface_distor.npy"), ans)






if __name__ == "__main__":
    dataset_score()
