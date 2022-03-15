from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from train_project.database import ImgDatabase, get_image_path
from correlation import  calculate_correlation
import  numpy as np
import  torch
import  random
import  gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

model_path = "../train_project/Incep_Resnet_1_2_3_conca_1ow_level_4/model_19.ckpt"
predict_model = torch.load(model_path).cuda()

annotation_path = r"D:\Yinyangtao\Image utility research of low-quality human face\annotation\annotation_result.npy"
annotaion = np.load(annotation_path).item()
root_path = r"D:\Yinyangtao\Image utility research of low-quality human face\Face_recognition\deepface-master\samples\human_face"
train_paths, valid_paths, test_paths = get_image_path(root_path)
test_database = ImgDatabase(img_paths = test_paths, annotation_path = annotation_path, mode="train", annotation_method="deep_face_score")
# total_number = 12556
test_number = 5
# batch = 256
# batch_time = test_number// batch #向下取整
test_index = [i for i in range(test_number)]
random.shuffle(test_index)
test_img, test_lable = test_database.getbatch(test_index[0:test_number])
predict_result = []
# for i in range(batch_time):
#     if i == batch_time-1:
#         test_batch = test_img[i * batch:test_number]
#     else:
#         test_batch = test_img[i*batch:(i+1)*batch]
with torch.no_grad():
    x_batch_1, x_batch_2, x_batch_3 = resnet(test_img.cuda(), layer="low_123")
    output = np.squeeze(predict_model(x_batch_1.cuda(), x_batch_2.cuda(), x_batch_3.cuda()).detach().cpu().numpy())
    predict_result = predict_result + list(output)
    del  x_batch_1,x_batch_2,x_batch_3,output
    gc.collect()

test_lable = np.squeeze(test_lable.numpy())
output = np.squeeze(predict_result)
print(output.shape)
print(test_lable.shape)
print("True score:", test_lable)
print("Predicted score:", output)
calculate_correlation(output, test_lable)
calculate_correlation(output, test_lable, "kendall")
calculate_correlation(output, test_lable, "pearson")
calculate_correlation(output, test_lable, "rmse")
