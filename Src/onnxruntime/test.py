import cv2
import numpy as np
import onnxruntime
import argparse

Classes = ["cup", "dog", "cat"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
Input_size = (640, 640)

def preprocess(img, input_size=(640, 640)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...].astype(np.float32)
    return img

def NMS(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def draw_detections(image, boxes, scores, class_ids, conf_thres):
    for i in range(len(boxes)):
        if scores[i] < conf_thres:
            continue
            
        box = boxes[i]
        class_id = int(class_ids[i])  # 确保转换为整数
        x, y, w, h = box
        
        # 检查类别ID是否有效
        if class_id < 0 or class_id >= len(Classes):
            continue
            
        color = COLORS[class_id % len(COLORS)]
        label = f"{Classes[class_id]}: {scores[i]:.2f}"
        
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (int(x), int(y) - th), (int(x) + tw, int(y)), color, -1)
        cv2.putText(
            image,
            label,
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

# 加载模型
session = onnxruntime.InferenceSession(
    "/home/yuqingchi/Code/Yolo-ONNX-TensorRT/runs/train/exp8/weights/best.onnx"
)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# 打印模型信息
print(f"输入名称: {input_name}")
print(f"输出名称: {output_names}")
print(f"输入形状: {session.get_inputs()[0].shape}")
print(f"输出形状: {session.get_outputs()[0].shape}")

camera = cv2.VideoCapture(2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break
        
    orig_h, orig_w = frame.shape[:2]
    
    # 预处理
    input_data = preprocess(frame)
    
    # 推理
    outputs = session.run(output_names, {input_name: input_data})
    
    # 后处理 - 修正版本
    output = outputs[0]  # 形状 [1, 7, 8400]
    
    # 打印输出信息用于调试
    print(f"输出形状: {output.shape}")
    
    # 转置为 [8400, 7]
    output = output.squeeze(0).transpose(1, 0)
    
    # 尝试不同的输出格式解析
    # 方法1: 假设格式为 [x_center, y_center, w, h, obj_score, class_id, class_score]
    boxes = output[:, :4]  # x_center, y_center, w, h
    obj_scores = output[:, 4]  # 对象置信度
    class_ids = output[:, 5]  # 类别ID
    class_scores = output[:, 6]  # 类别置信度
    
    # 计算最终得分 = 对象置信度 * 类别置信度
    scores = obj_scores * class_scores
    
    # 如果类别ID范围异常，尝试其他格式
    if np.max(class_ids) > 100:
        print("检测到类别ID异常，尝试其他格式...")
        # 方法2: 假设格式为 [x_center, y_center, w, h, class_id, obj_score, class_score]
        boxes = output[:, :4]
        class_ids = output[:, 4]
        obj_scores = output[:, 5]
        class_scores = output[:, 6]
        scores = obj_scores * class_scores
    
    # 确保类别ID是整数
    class_ids = class_ids.astype(np.int32)
    
    # 打印类别ID分布
    unique_classes = np.unique(class_ids)
    print(f"检测到的类别ID: {unique_classes}")
    
    # 过滤低置信度
    conf_thres = 0.2
    valid_indices = scores > conf_thres
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]
    
    # 应用NMS和绘制结果
    if len(boxes) > 0:
        # 将中心坐标转换为左上角坐标
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        
        # NMS
        indices = NMS(boxes, scores, 0.2)
        
        # 缩放回原始图像尺寸
        boxes[:, [0, 2]] *= orig_w / 640
        boxes[:, [1, 3]] *= orig_h / 640
        
        # 绘制检测结果
        draw_detections(frame, boxes[indices], scores[indices], class_ids[indices], conf_thres)
    
    # 显示结果
    cv2.imshow("YOLOv13 ONNX Inference - Camera", frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
camera.release()
cv2.destroyAllWindows()