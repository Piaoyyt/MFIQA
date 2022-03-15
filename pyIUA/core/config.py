from yacs.config import  CfgNode as CN
from project_path.get_project_path import project_path
import sys
__C = CN()
cfg = __C
__C.CUDA =True
__C.Config_path =project_path+"\\experiments\\IUA-Inception_resnet\\config.yaml"
# __C.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#----------------------------------------------------
#Dataset options
#----------------------------------------------------
__C.Dataset = CN()

__C.Dataset.train_annotation_path = project_path+ "\\dataset\\WDFace\\Annotation_file\\InsightFace_Train_quality_labels.npy"
#__C.Dataset.root_path = project_path+"\\dataset\\human_face"
__C.Dataset.root_path = project_path+"\\dataset\\WDFace\\WD_Input"
__C.Dataset.test_annotation_path = project_path+"\\dataset\\WDFace\\Annotation_file\\InsightFace_Test_quality_labels.npy"
#__C.Dataset.test_annotation_path = project_path+"\\dataset\\\\Annotations\\human_annotation_sphereface_merge.npy"
#__C.Dataset.test_root_path = r"..\\dataset\\WDFace\\input_face\\Train"
__C.Dataset.annotation_method = 'insightface_cosine_distance'#['sphereface_euclidean_distance','sphereface_euclidean_score','sphereface_cos_score','insightface_euclidean_distance']
__C.Dataset.label_type = "continuous"  # 设置标签的类型，可选的参数["binary", "continuous"]
__C.Dataset.label_normalize = True#Whether the score has been normalized to 0-1
__C.Dataset.thresh = 0  # 设定二分类时的阈值,仅在label_type为"binary"时有效
__C.Dataset.landmark_path=project_path+"\\dataset\\landmark_face.txt"
#print("本次选择的人脸识别分数为:{}".format(__C.Dataset.annotation_method))

#--------------------------------------------------
# Training options
#--------------------------------------------------
__C.TRAIN = CN()
#损失函数和优化器的设置
__C.TRAIN.optim_name = 'Adam'#["SGD","Adam"]
__C.TRAIN.loss_function = "mean_square_loss"#["binary_cross_entropy""mean_square_loss""contrast_loss""huber_loss"]
# 参数的设置
__C.TRAIN.learning_rate = 0.001  # 学习率
__C.TRAIN.batch_size = 64  # batch大小
__C.TRAIN.iter_times = 20  # 迭代次数
__C.TRAIN.decay_steps = 4  # 学习率衰减步长
__C.TRAIN.decay_rate = 0.9  # 学习率衰减速率
__C.TRAIN.weight_decay = 1e-4#正则化参数
__C.TRAIN.print_loss_gap = 1  # 打印损失间隔
# 计算均方误差时的损失函数构成  total_loss = mse_lambda * mse_loss + rank_lambda * rank_loss
__C.TRAIN.mse_lambda = 1
__C.TRAIN.rank_lambda = 2
__C.TRAIN.ckpt_store_gap = 2  # 存储模型的步长
cfg.TRAIN.dataset = "WDFace"
__C.TRAIN.store_root =  project_path+"\\experiments\\model_result"
__C.TRAIN.layer = 15#"final"#选择主干网络Inception-resnet里面从哪一层截断，可选["low_integrated","low3_high_1","low2_high_1","high_integrated",15,"final"]
__C.TRAIN.pretrained_model =None#project_path+"\\experiments\\model_result\\2021_08_29_20_WDFace\\model_file\\model_19.ckpt"# None#project_path+"\\experiments\\model_result\\2021_08_27_20_WDFace\\model_file\\model_8.ckpt"  #None#设置预训练模型
__C.TRAIN.pre_dropout =0.2#设置训练主干网络的dropout值
__C.TRAIN.full_dropout =0#设置后面全连接层的dropout值
#设置先验知识（手工特征）
__C.TRAIN.prior = None#"brisque"
__C.TRAIN.prior_feature = ""
__C.TRAIN.prior_dimension =36

#----------------------------------------------------
#Backbone options
#----------------------------------------------------
__C.BACKBONE = CN()

__C.BACKBONE.TYPE ="Inception-Resnet"
__C.BACKBONE.PRETRAINED ='vggface2'#vggface2#Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.(default: {None})
__C.BACKBONE.DROPOUT = 0.2
__C.BACKBONE.KWARGS =CN(new_allowed=True)

__C.Tailnet =CN()
__C.Tailnet.TYPE ="Full-connection"
__C.Tailnet.layer_number = 1
__C.Tailnet.layer_set = [512*2,32]

#------------------------------------------------
#Testing options
#------------------------------------------------
__C.TEST = CN()
__C.TEST.Model_path = project_path+"/experiments/model_result/2021_09_04_20_WDFace/model_file/model_12.ckpt"
__C.TEST.Batch_size = 64

if sys.platform == "linux":
    __C.Config_path =__C.Config_path.replace("\\", "/")
    __C.Dataset.train_annotation_path = __C.Dataset.train_annotation_path.replace("\\", "/")
    __C.Dataset.root_path = __C.Dataset.root_path.replace("\\", "/")
    __C.Dataset.test_annotation_path = __C.Dataset.test_annotation_path.replace("\\", "/")
    __C.Dataset.landmark_path = __C.Dataset.landmark_path.replace("\\", "/")
    __C.TRAIN.store_root = __C.TRAIN.store_root.replace("\\", "/")
    #__C.TEST.Model_path = __C.TEST.Model_path.replace("\\", "/")
