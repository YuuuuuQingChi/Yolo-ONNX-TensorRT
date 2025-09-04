# 将图片和标注数据按比例切分为 训练集\测试集\验证集
import shutil
import random
import os


image_original_path = (
    "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/RawData/imgs/"
)
label_original_path = (
    "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/RawData/labels/"
)

train_image_path = "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/images/train"
train_label_path = "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/labels/train"

val_image_path = "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/images/val"
val_label_path = "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/labels/val"

test_image_path = "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/images/test"
test_label_path = "/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/Data/labels/test"


train_percent = 0.75
val_percent = 0.15
test_percent = 0.15


def confirm_dir_existence(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    confirm_dir_existence(train_image_path)
    confirm_dir_existence(train_label_path)
    confirm_dir_existence(val_image_path)
    confirm_dir_existence(val_label_path)
    confirm_dir_existence(test_image_path)
    confirm_dir_existence(test_label_path)

    total_txt = os.listdir(label_original_path)
    print("total_txt==", total_txt)
    num_txt = len(total_txt)
    print("num_txt==", num_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    num_test = num_txt - num_train - num_val

    train = random.sample(list_all_txt, num_train)
    val_and_test = [i for i in list_all_txt if not i in train]
    val = random.sample(val_and_test, num_val)
    test = [i for i in val_and_test if not i in val]

    for i in list_all_txt:
        name = total_txt[i][:-4]
    
        srcImage = image_original_path + "/" + name + ".jpg"
        srcLabel = label_original_path + "/" + name + ".txt"
 
        if i in train:
            dst_train_Image = train_image_path + "/" + name + ".jpg"
            dst_train_Label = train_label_path + "/" + name + ".txt"
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        elif i in val:
            dst_val_Image = val_image_path + "/" + name + ".jpg"
            dst_val_Label = val_label_path + "/" + name + ".txt"
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
        elif i in test:
            dst_test_Image = test_image_path + "/" + name + ".jpg"
            dst_test_Label = test_label_path + "/" + name + ".txt"
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)


if __name__ == "__main__":
    main()
