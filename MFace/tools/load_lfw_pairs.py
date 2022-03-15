import os
import numpy as np

def load_LFW_pairs(root_path, path:str):
    result = []
    with open(path, 'r') as f:
        for each in f.readlines():
            split_str = each.strip().split("\t")
            if len(split_str)==3:
                name = split_str[0]
                imgs_list = sorted(os.listdir(os.path.join(root_path,name)))
                try:
                    id_1 =  imgs_list[int(split_str[1])-1]
                    id_2 = imgs_list[int(split_str[2])-1]
                    result.append([name, id_1, name, id_2])
                except:
                    print(name, split_str[1], split_str[2])
                
                # print("same")
                # same +=1
            elif len(split_str) == 4:
                name_1 = split_str[0]
                imgs_list_1 = sorted(os.listdir(os.path.join(root_path, name_1)))
                name_2 = split_str[2]
                imgs_list_2 = sorted(os.listdir(os.path.join(root_path, name_2)))
                try:
                    id_1 = imgs_list_1[int(split_str[1])-1]
                    id_2 = imgs_list_2[int(split_str[3])-1]
                    result.append([name_1, id_1, name_2, id_2])
                except:
                    print(name_1, id_1)
                    print(name_2, id_2)
                #result.append([name, id_1, name, id_2])
    np.save("LFW_pairs.npy", result)


if __name__ == "__main__":
    pairs =  load_LFW_pairs(root_path= "/media/yyt/My Passport/My_project/dataset/Public_dataset/lfw_align",path="./lfw_pairs.txt")
