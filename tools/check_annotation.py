from project_path.get_project_path import project_path
import numpy as np
import matplotlib.pyplot as plt
#annotation_path = project_path+ "\\Image utility research of low-quality human face\\Face_recognition\\InsightFace_v2_master\\InsightFace_Train_quality_euclidean_labels.npy"
annotation_path = "./Facenet_Test_quality_cosine_labels.npy"
file = np.load(annotation_path,allow_pickle=True).item()
quality_label = []
person_quality ={}
for img_name in file.keys():
    quality_label.append(file[img_name]["insightface_euclidean_distance"])
    person_quality[img_name] = file[img_name]["insightface_euclidean_distance"]
# person_quality_sort = sorted(person_quality.items(),key=lambda x:x[1])
quality_max = max(quality_label)
quality_min = min(quality_label)
#quality_label = np.array(quality_label)
def get_distribution_graph(quality, gap_num):
    quality_min = min(quality)
    quality_max = max(quality)
    ratio = 100/gap_num
    quality_gap = 1
    quality_bar = [i*ratio for i in range(0, int(100/ratio+1), quality_gap)]
    score_count = len(quality_bar) *[0]
    for quality_score in quality:
        for j in range(len(quality_bar)-1):
            if 100*quality_score/quality_max>=quality_bar[j] and 100*quality_score/quality_max<quality_bar[j+1]:
                score_count[j]+=1
    plt.plot(quality_bar, score_count,'r-')
    plt.savefig("euclidean_norm_train_dis.jpg",dpi=200)
def normalize_score(file , score_max, score_min):
    for img_name in file.keys():
        file[img_name]["insightface_euclidean_distance"] = \
        round(1- (file[img_name]["insightface_euclidean_distance"]-score_min)/(score_max-score_min),3)
    np.save("InsightFace_Train_quality_euclidean_norm_labels.npy",file)
#normalize_score(file, quality_max, quality_min)
get_distribution_graph(quality_label, gap_num=100)
print()