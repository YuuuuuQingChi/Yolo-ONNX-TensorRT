from yolov13.ultralytics import YOLO

model = YOLO('/home/yuqingchi/Code/Yolo-ONNX-TensorRT/runs/train/exp8/weights/best.pt')
model.export(format="onnx",half =True)