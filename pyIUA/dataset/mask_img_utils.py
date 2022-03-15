import os,sys

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from project_path.get_project_path import project_path
from landmark_detection.Retinaface import Retinaface
from landmark_detection.common.utils import BBox, drawLandmark_multiple
from landmark_detection.models.mobilefacenet import MobileFaceNet
from landmark_detection.utils.align_trans import warp_and_crop_face, get_reference_facial_points
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
crop_size = 160
out_size = 112
# crop_size = 160
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
def load_model():
    '''
    加载人脸检测模型
    :return:
    '''

    model = MobileFaceNet([112, 112], 136)
    model_path = project_path+'\\landmark_detection\\checkpoint\\mobilefacenet_model_best.pth.tar'
    if sys.platform == "linux":
        model_path = model_path.replace("\\", "/")
    checkpoint = torch.load(model_path, map_location=map_location)
    #print('Use MobileFaceNet as backbone')

    model.load_state_dict(checkpoint['state_dict'])

    return model
def add_mask(img, landmark, face_exist):
    '''
    :param img:输入的人脸
    :param landmark: 检测出来的关键点
    :param face_exist: 是否检测出脸
    :return:掩膜人脸
    '''
    if face_exist:
        img = np.squeeze(img)
        height, width, _ = img.shape
        left_eye_x_min = max(min(landmark[36:42, 0]) - 2, 0)
        left_eye_x_max = min(max(landmark[36:42, 0]) + 2, width)
        left_eye_y_min = max(min(landmark[36:42, 1]) - 2, 0)
        left_eye_y_max = min(max(landmark[36:42, 1]) + 2, height)
        right_eye_x_min = max(min(landmark[42:48, 0]) - 2, 0)
        right_eye_x_max = min(max(landmark[42:48, 0]) + 2, width)
        right_eye_y_min = max(min(landmark[42:48, 1]) - 2, 0)
        right_eye_y_max = min(max(landmark[42:48, 1]) + 2, height)
        nose_x_min = max(min(landmark[29:36, 0]) - 2, 0)
        nose_x_max = min(max(landmark[29:36, 0]) + 2, height)
        nose_y_min = max(min(landmark[29:36, 1]) - 2, 0)
        nose_y_max = min(max(landmark[29:36, 1]) + 2, height)
        mouse_x_min = max(min(landmark[48:60, 0]) - 2, 0)
        mouse_x_max = min(max(landmark[48:60, 0]) + 2, width)
        mouse_y_min = max(min(landmark[48:60, 1]) - 2, 0)
        mouse_y_max = min(max(landmark[48:60, 1]) + 2, height)
        for i in range(height):
            for j in range(width):
                if(i>left_eye_y_min and i<left_eye_y_max and j>left_eye_x_min and j<left_eye_x_max)\
                    or (i>right_eye_y_min and i<right_eye_y_max and j>right_eye_x_min and j<right_eye_x_max)\
                    or (i>nose_y_min and i<nose_y_max and j>nose_x_min and j<nose_x_max)\
                    or (i>mouse_y_min and i<mouse_y_max and j>mouse_x_min and j<mouse_x_max):
                    img[i,j,:] = 0
    else:
        img[:,:,:] = 0
    return img


def process_faces(img, faces, model, draw_lanmark=False, save_align=False):
    '''
    :param faces:人脸检测器检测出来的结果
    :return:
    '''
    model = model.cuda().eval()
    face_exist = True
    height, width,_ = img.shape
    #org_img = img.copy()
    if len(faces) == 0:
        face_exist =False
        #print('NO face is detected!')
        # landmark = None
        # mask_img = add_mask(img, landmark, face_exist)
        # mask_img = cv2.resize(mask_img, dsize=(160, 160))
        mask_img = cv2.resize(img, dsize=(160, 160))
        return mask_img

    face = faces[0]
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))
    new_bbox = BBox(new_bbox)
    cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    # if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
    #     continue
    test_face = cropped_face.copy()
    test_face = test_face / 255.0
    # if args.backbone == 'MobileNet':
    #     test_face = (test_face - mean) / std
    test_face = test_face.transpose((2, 0, 1))
    test_face = test_face.reshape((1,) + test_face.shape)
    input = torch.from_numpy(test_face).float()
    input = torch.autograd.Variable(input)
    landmark = model(input.cuda())[0].cpu().data.numpy()
    landmark = landmark.reshape(-1, 2)
    landmark = new_bbox.reprojectLandmark(landmark)
    if draw_lanmark:
        img = drawLandmark_multiple(img, new_bbox, landmark)
    mask_img = add_mask(img,landmark ,face_exist)
    mask_img = cv2.resize(mask_img,dsize=(160,160))

    return mask_img


def gen_mask_img(img):
    '''
    :param img: 输入的人脸图像
    :return: 掩膜人脸图像
    '''
    img = np.array(img)
    model = load_model()
    retinaface = Retinaface.Retinaface()#print注释掉
    #img_m = img.copy()
    faces = retinaface(img)
    mask_img = process_faces(img, faces, model)
    mask_img = Image.fromarray(mask_img.astype('uint8')).convert('RGB')
    return mask_img
def detect_face(img):
    img = np.array(img)
    retinaface = Retinaface.Retinaface()
    faces = retinaface(img)
    if len(faces)==0:
        return False
    else:
        return True

if __name__ =="__main__":
    img = Image.open("./0051_01.jpg")
    mask_img = gen_mask_img(img)
    plt.ion()
    plt.axis("off")
    plt.imshow(mask_img)
    plt.pause(4)
    plt.imshow(img)
    plt.pause(4)
    plt.ioff()
    plt.clf()
    plt.close()
    # mask_img.show()
    # cv2.imshow("ss",mask_img)
    # cv2.waitKey(0)
    print(type(mask_img))
   # print(mask_img.shape)
    # cv2.waitKey(0)
