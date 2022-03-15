from yacs.config import CfgNode as CN
import os, sys

project_path = "\\".join(os.getcwd().split("\\")[:-2]) if not sys.platform == "linux" \
    else "/".join(os.getcwd().split("/")[:-2])
sys.path.append(project_path)
from project_path.get_project_path import project_path

__C = CN()
cfg = __C

cfg.fiqa_score_path = os.path.join(project_path, 'Evaluation', 'FIQA_score_store')
cfg.MFIQ_path = os.path.join(project_path, "Evaluation", "FIQA_model", "models", "fiqa_mine_v2.ckpt")
cfg.face_dimension = 512 + 512  # 传入分类层的人脸特征维度
# *------------训练设置---------------#
cfg.Train = CN()
cfg.Train.method = "Arcface_quali1"  # 训练人脸分类模型的方法["MFaceSphereface",..."Sphereface", "MFaceSphereface_v2"]
cfg.Train.backbone = "resnet50"  # resnet50_IR_SE,resnet50_IR
cfg.Train.metric = "arc_margin"  # "add_margin"，'arc_margin'，'sphere'
cfg.Train.loss = "focal_loss"  # focal_loss,angle_loss
cfg.Train.quality_loss = False  # 是否加入质量分数指导的损失函数 QualiIndicLoss（）
cfg.Train.quality_constraint = True  # 是否加入质量分数指导的损失函数Uq_constrloss()
cfg.Train.backbone_epoch = 0  #主干网络的训练起始epoch
cfg.Train.metricbone_epoch = 0 #分类层的训练起始epoch
cfg.Train.test_ratio = 0  # 测试集划分比例，
cfg.Train.feature_fusion = True  # 是否将人脸特征和质量特征进行融合
cfg.Train.face_score = True  #
cfg.Train.use_se = False
cfg.Train.img_input = [160, 160]
# cfg.Train.root_path = project_path+ "\\dataset\\Public_dataset\\CASIA_WebFace_alignment"
cfg.Train.dataset = "CASIA_WebFace"  # WDFace
cfg.Train.pretrained = None  # [project_path + "/MFace/result_models/Arcface_quali1
# /2022_03_06_16_CASIA_WebFace_distor/resnet50_19.ckpt", \ project_path +
# "/MFace/result_models/Arcface_quali1/2022_03_06_16_CASIA_WebFace_distor/resnet50_arc_margin_19.ckpt"]#project_path
# + "/MFace/result_models/Arcface/2022_01_04_11_LFW_fake/resnet50_arc_margin_89.ckpt"]#"..\\result_models\\Sphereface
# \\2021_08_12_13\\model_48.ckpt"
cfg.Train.epochs = 30
cfg.Train.batch_size = 256
cfg.Train.dataloader = {"shuffle": True, "pin_memory": True, "num_workers": 8, "drop_last": True}
# 主干网络的训练设置
cfg.Train.backbone_lr = 0.001
cfg.Train.backbone_ldecay = 0.1
cfg.Train.backbone_lstep = [10, 20]
cfg.Train.backbone_wdecay = 5e-4
cfg.Train.backbone_epoch_start = 0
# 分类层的训练设置
cfg.Train.metricbone_lr = 1e-3
cfg.Train.metricbone_ldecay = 0.1
cfg.Train.metricbone_lstep = 4
cfg.Train.metricbone_wdecay = 5e-4
cfg.Train.metricbone_epoch_start = 0
cfg.Train.momentum = 0.9
cfg.Train.optimizer = "adam"

cfg.Train.qualalign_lr = 1e-3
cfg.Train.qualalign_ldecay = 0.1
cfg.Train.qualalign_lstep = 4
cfg.Train.qualalign_wdecay = 5e-4
cfg.Train.fiqa = "FIQA_mine_v2"  # [FIQA_mine \\ FaceQnet_v1 \\FaceQnet_v0\\ SER-FIQ \\FIQA_mine_v2]
# cfg.Train.loss = "focal_loss"#[""
cfg.Train.imgs_root = CN()
cfg.Train.imgs_root.DDFace = project_path + "\\dataset\\WDFace\\WD_Input\\Train"
cfg.Train.imgs_root.CelebA = project_path + "\\dataset\\Public_dataset\\CelebA-alignment"
cfg.Train.imgs_root.LFW = project_path + "\\dataset\\human_face_original"
cfg.Train.imgs_root.LFW_fake = project_path + "\\dataset\\fake_lfw"
cfg.Train.imgs_root.VGGFace2 = project_path + "\\dataset\\Public_dataset\\vggface2_train_align"
cfg.Train.imgs_root.CASIA_WebFace = project_path + "\\dataset\\Public_dataset\\CASIA_WebFace_alignment"

cfg.Train.imgs_root.CASIA_WebFace_distor = project_path + "\\dataset\\Public_dataset\\Distor_datasets\\CASIA-Webface"
# cfg.Train.loss_minimum = 5.9095#预测准确时的最小损失，这里因为采用了softmax之后再每一个log，所以导致不是为0，不过不影响。

cfg.Train.store_epoch = 2
cfg.Train.store_path = project_path + "\\MFace\\result_models"
# *------------验证设置---------------#
cfg.Valid = CN()
cfg.Valid.step = 2

# *------------测试设置---------------#
cfg.Test = CN()
cfg.Test.method = "Sphereface"
cfg.Test.model_path = project_path + "\\MFace\\result_models\\Sphereface\\2021_09_23_09_CASIA_WebFace\\model_98.ckpt"
cfg.Test.root_path = project_path + "\\dataset\\human_face"
cfg.Test.dataset = "LFW_original"
cfg.Test.fiqa = "FIQA_mine_v2"  # [FIQA_mine \\ FaceQnet_v1 \\FaceQnet_v0\\ SER-FIQ \\FIQA_mine_v2]
cfg.Test.batch_size = 8

# *------------测试设置---------------#
cfg.Eval = CN()
cfg.Eval.size = [160, 160, 3]
cfg.Eval.face_recoginition = "Arcface"
cfg.Eval.model_path = project_path + "\\MFace\\result_models\\Arcface\\2022_02_21_10_CASIA_WebFace_distor\\resnet50_3.ckpt"
cfg.Eval.model_name = "Arcface"  # Sphereface  Insighface   MFaceSphereface Sphere_ori
cfg.Eval.dataset = "CelebA_Hard1_distor"  # [CelebA, LFW, VGGFace2, CASIA_WebFace, WDFace]
"""
[CelebA, LFW, VGGFace2, CASIA_WebFace, WDFace,LFW_original]→ 
[LFW_original_Hard1_distor→LFW原始图像对的图像，每一对图像加入Hard失真类型生成新的一对
 LFW_original_Easy1_distor→LFW原始图像对的图像，每一对图像加入Easy失真类型生成新的一对
 CelebA_Hard1_distor→CelebA原始图像对的图像，每一对图像加入Hard失真类型生成新的一对
 CelebA_Easy1_distor→CelebA原始图像对的图像，每一对图像加入Easy失真类型生成新的一对
"""
cfg.Eval.feature_align = False  # take the face feature and the quality feature as the face representation?
cfg.Eval.imgs_pairs = CN()
# 原始的图像对
cfg.Eval.imgs_pairs.DDFace = project_path + "\\Evaluation\\images_pairs\\WDFace_images_pairs.npy"
cfg.Eval.imgs_pairs.CelebA = project_path + "\\Evaluation\\images_pairs\\Celeba_images_pairs.npy"
cfg.Eval.imgs_pairs.LFW = project_path + "\\Evaluation\\images_pairs\\LFW_images_pairs.npy"
cfg.Eval.imgs_pairs.LFW_original = project_path + "\\Evaluation\\images_pairs\\LFW_pairs.npy"
cfg.Eval.imgs_pairs.VGGFace2 = project_path + "\\Evaluation\\images_pairs\\VGGface_images_pairs.npy"
cfg.Eval.imgs_pairs.CASIA_WebFace = project_path + "\\Evaluation\\images_pairs\\CASIA-Webface_images_pairs.npy"
# 加入失真后的图像对
cfg.Eval.imgs_pairs.LFW_original_Hard1_distor = project_path + "\\Evaluation\\images_pairs\\LFW_original_Hard_1_distor_pairs.npy"
cfg.Eval.imgs_pairs.LFW_original_Easy1_distor = project_path + "\\Evaluation\\images_pairs\\LFW_original_Easy_1_distor_pairs.npy"

cfg.Eval.imgs_pairs.CelebA_Hard1_distor = project_path + "\\Evaluation\\images_pairs\\CelebA_Hard_1_distor_pairs.npy"
cfg.Eval.imgs_pairs.CelebA_Easy1_distor = project_path + "\\Evaluation\\images_pairs\\CelebA_Easy_1_distor_pairs.npy"

cfg.Eval.imgs_root = CN()
# 原始的图像对所在的根路径
cfg.Eval.imgs_root.DDFace = project_path + "\\dataset\\WDFace\\WD_Input\\Test"
cfg.Eval.imgs_root.CelebA = project_path + "\\dataset\\Public_dataset\\CelebA-alignment"
cfg.Eval.imgs_root.LFW = project_path + "\\dataset\\human_face_original"
cfg.Eval.imgs_root.LFW_original = project_path + "\\dataset\\Public_dataset\\lfw_align"
cfg.Eval.imgs_root.VGGFace2 = project_path + "\\dataset\\Public_dataset\\vggface2_test_alignment"
cfg.Eval.imgs_root.CASIA_WebFace = project_path + "\\dataset\\Public_dataset\\CASIA_WebFace_alignment"
# 失真的图像对所在的根路径
cfg.Eval.imgs_root.LFW_original_Hard1_distor = project_path + "\\dataset\\Public_dataset\\Distor_test_imgs\\LFW_original"
cfg.Eval.imgs_root.LFW_original_Easy1_distor = project_path + "\\dataset\\Public_dataset\\Distor_test_imgs\\LFW_original"
cfg.Eval.imgs_root.CelebA_Hard1_distor = project_path + "\\dataset\\Public_dataset\\Distor_test_imgs\\CelebA"
cfg.Eval.imgs_root.CelebA_Easy1_distor = project_path + "\\dataset\\Public_dataset\\Distor_test_imgs\\CelebA"
cfg.Eval.fiqa = "FIQA_mine_v2"  # [FIQA_mine \\ FaceQnet_v1 \\FaceQnet_v0\\ SER-FIQ\\ BRISQUE\\ PIQE\\ NIQE]

cfg.Eval.store_path = project_path + "\\MFace\\Eval_results"
if __name__ == "__main__":
    # for attr in cfg.__dict__:
    #    if attr
    print(cfg)
