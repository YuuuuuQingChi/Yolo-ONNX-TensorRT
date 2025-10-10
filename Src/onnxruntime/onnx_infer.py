import cv2
import numpy as np
import onnxruntime
import argparse  # 解析命令行参数

Classes =  ['cat', 'bottle', 'dog']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
Input_size = (640, 640)


def preprocess(img, input_size=(640, 640)):
    # 图像处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    dst = np.zeros_like(img)
    img = cv2.normalize(
        img.astype(np.float32),dst=dst,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    # img = img / 255.0  # 归一化
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, ...].astype(np.float32)

    return img


def NMS(boxs, scores, line):
    if len(boxes) == 0:
        return []
    x1 = boxs[:, 0]
    y1 = boxs[:, 1]
    x2 = x1 + boxs[:, 2]
    y2 = y1 + boxs[:, 3]

    areas = boxs[:, 2] * boxs[:, 3]
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

        Iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(Iou < line)[0]
        # 这个where返回的是元组，但是我们只想要第一个数组
        order = order[inds + 1]

    return keep


def draw_detections(image, boxes, scores, class_idx, conf_thres):
    # image: 要绘制检测结果的原始图像

    # boxes: 检测到的边界框，格式为 [x, y, width, height]

    # scores: 每个检测框的置信度分数

    # class_ids: 每个检测框的类别ID

    # conf_thres: 置信度阈值，低于此值的检测结果不绘制
    for i in range(len(boxes)):
        if scores[i] < conf_thres:
            continue

        box = boxes[i]
        class_id = class_idx[i]
        x, y, w, h = box
        # color: 根据类别ID选择颜色（使用取模确保不越界）
        color = COLORS[ int(class_id) % len(COLORS)]
        # f"..."：f-string格式化字符串
        label = f"{Classes[int(class_id)]}: {scores[i]:.2f}"

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


# 这个是onnx的核心加载模型的函数
session = onnxruntime.InferenceSession(
    "/home/yuqingchi/Code/Yolo-ONNX-TensorRT/runs/train/exp8/weights/best.onnx"
)
# 获得输入和输出
input_name = session.get_inputs()[0].name
# 你会获得[NodeArg(name='input_tensor', type='tensor(float)', shape=[1, 3, 224, 224])]
output_names = [o.name for o in session.get_outputs()]
camera = cv2.VideoCapture(2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

while camera.isOpened():
    success, frame = camera.read()
    # ret：一个布尔值，表示是否成功读取帧。如果读取成功，则为True；如果没有帧可读（例如视频结束），则为False。
    # frame：读取到的视频帧，是一个图像数组（numpy数组）。如果读取失败，frame可能会是None。
    orig_h, orig_w = frame.shape[:2]

    input_data = preprocess(frame)
    outputs = session.run(output_names, {input_name: input_data})
    # 后处理
    output = outputs[0].squeeze(0)  # 假设输出为 (1, 8400, 84) -> (8400, 84)

    # 转置以便每个检测框是一行
    output = output.transpose(1, 0)

    print("输出矩阵的前几行:")
    print(output[:5, :])  # 打印前5个检测框的7个值
    boxes = output[:, :4]
    scores = np.max(output[:, 4:], axis=1)
    class_ids = np.argmax(output[:, 4:], axis=1)

    print(class_ids)

    # 过滤低置信度
    valid_indices = scores > 0.2
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]

    # NMS
    # print(" shape:", boxes.shape) 
    # 缩放回原始图像尺寸
    if len(boxes) > 0:
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x = x_center - width/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y = y_center - height/2
        indices = NMS(boxes, scores, 0.2)
        boxes[:, [0, 2]] *= orig_w / 640
        boxes[:, [1, 3]] *= orig_h / 640

        # 绘制检测结果
        draw_detections(frame, boxes[indices], scores[indices], class_ids[indices], 0.2)

        # 显示结果
    cv2.imshow("YOLOv13 ONNX Inference - Camera", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 释放资源
camera.release()
cv2.destroyAllWindows()
