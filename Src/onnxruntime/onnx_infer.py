import cv2
import numpy as np
import onnxruntime 
import argparse#解析命令行参数

Classes = ["cup", "dog", "cat"]
Colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
Input_size = (640, 640)


def preprocess(img, input_size=(640, 640)):
    # 图像处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = cv2.normalize(
        img.astype(np.float32), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    img = np.transpose(img,(2,0,1))
    img = img[np.newaxis, ...].astype(np.float32)  

    return img    

def NMS():
    
    return 0

def main(args):
    #这个是onnx的核心加载模型的函数
    session = onnxruntime.InferenceSession(args.model)
    #获得输入和输出
    input_name = session.get_inputs()[0].name
    #你会获得[NodeArg(name='input_tensor', type='tensor(float)', shape=[1, 3, 224, 224])]
    #但
    output_name = session.get_outputs()[0]