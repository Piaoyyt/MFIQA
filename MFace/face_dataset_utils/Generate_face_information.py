"""
Generate the face image informaiton including the face id, face quality for the Face Dataset Construction
"""
import os
import sys

import numpy as np
from Evaluation.FIQA_model.FIQA_mine import FIQA_mine_v2
#from Evaluation.FIQA_model.FaceQnet_v1 import FaceQnet_v1, FaceQnet_v0
#from Evaluation.FIQA_model.SER_FIQ import SER_FIQ
#from Evaluation.IQA_model.Brisque import Brisque
#from Evaluation.IQA_model.Niqe import NIQE
#from Evaluation.IQA_model.Piqe import PIQE
#from Evaluation.FIQA_model.FIQA_mine import FIQA_mine
from MFace.face_dataset_utils.Dataset import Face_Dataset
def load_FIQA_Model(fiqa_model="MFIQA"):
    """
    :param fiqa_model:The model that is used for calculate the face image score
    :return: a list that stores the face images scores
    """
    model_dict = {#"FaceQnet_v0": FaceQnet_v0, \
             #"FaceQnet_v1": FaceQnet_v1,
             #"FIQA_mine": FIQA_mine,
             "FIQA_mine_v2": FIQA_mine_v2}
             #"SER-FIQ": SER_FIQ, \
             #"BRISQUE": Brisque, \
             #"NIQE": NIQE, "PIQE": PIQE}
    model = model_dict[fiqa_model]
    return model

def generate_face_quality(root_path:str, dataset_name:str, face_images_paths:list, store_path = None, fiqa_model ="MFIQA")->dict:
    """
     :param root_path: The root path of the face image paths
     :param face_images_paths: The list that stores the face images paths
     :param fiqa_model:The name of  FIQA model   ["MFIQA","FaceQnet_v1","FaceQnet_v0","SER-FIQ"]
     :return: A dict that stores all face images' quality score
    """
    score_file_path = store_path +f"\\{fiqa_model}_{dataset_name}.npy"
    if sys.platform == "linux":
        score_file_path = score_file_path.replace("\\", "/")
    if os.path.exists(score_file_path):
        faces_score = np.load(score_file_path, allow_pickle=True)
        return faces_score
    model = load_FIQA_Model(fiqa_model)()
    faces_score = {}
    for face_img in face_images_paths:
        face_img_path = os.path.join(root_path,face_img)
        try:
            face_score = min(model.predict(image_path=face_img_path).cpu()[0][0],1)
            faces_score[face_img] = face_score
        except:
            print(face_img)


    np.save(f"{store_path}\\{fiqa_model}_Face_Quality_{dataset_name}.npy",faces_score)
    return faces_score

def generate_face_images_paths(root_path:str, bound = 10)->list:
    """
    :param root_path:The root path of the face image paths.
    :param bound: The maximum number of the face images under each face id.
    :return:
    """
    face_images_paths = []
    for face_name in os.listdir(root_path):
        start = 0
        for img_name in os.listdir(os.path.join(root_path,face_name)):
            if start >= bound:
                break
            if img_name.endswith(".jpg") or img_name.endswith(".bmp") \
                or img_name.endswith(".png") or img_name.endswith("jpeg"):
                face_images_paths.append(os.path.join(face_name,img_name))
                start+=1
    return face_images_paths
def generate_face_id(root_path:str)->dict:
    """
    :param root_path:
    :return:A dict that stores the face id
    """
    face_name_set = set()
    face_ids = {}
    for face_name in os.listdir(root_path):
        face_name_set.add(face_name)
    face_id = 0
    face_name_set=sorted(face_name_set)
    for face_name in face_name_set:
        face_ids[face_name]= face_id
        face_id+=1
    #print("The total Face ID numbers is %d" %(len(face_name_set)))
    return face_ids
if __name__=="__main__":
    root_path = ""
    faces_id = generate_face_id(root_path)
    faces_imgs_paths = generate_face_images_paths(root_path)
    faces_quality_score = generate_face_quality(root_path,faces_imgs_paths,fiqa_model="MFIQA")
    Dataset = Face_Dataset(root_path,faces_imgs_paths,faces_id,faces_quality_score,seed=1)
