import warnings
import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录（Src目录）
current_dir = os.path.dirname(current_file)
# 获取上级目录（项目根目录，包含yolov13的目录）
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到Python路径中
sys.path.append(project_root)

# 现在忽略警告并导入YOLO
warnings.filterwarnings("ignore")
from yolov13.ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(
        model='/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/yolov13/ultralytics/cfg/models/v13/yolov13.yaml'
    )
    model.load('/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/yolov13n.pt')
    # 一般情况下不加载预训练权重。
    model.train(
        data=r'/home/yuqingchi/Code/yolo+ONNXruntime+TensorRT/dataset.yaml',
        imgsz=640,
        epochs=30,
        batch=4,
        workers=2,
        device="0",
        optimizer="SGD",
        close_mosaic=10,
        resume=False,
        project="runs/train",
        name="exp",
        single_cls=False,
        cache=False,
    )