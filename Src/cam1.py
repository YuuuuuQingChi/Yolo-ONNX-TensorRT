# -*- coding: utf-8 -*-
# import sys
# import os
# current_file = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file)
# project_root = os.path.dirname(current_dir)

# sys.path.append(project_root)
# sys.path.append(os.path.join(project_root, 'yolov13'))
# print(sys.path)
import cv2
import time
from yolov13.ultralytics import YOLO


model = YOLO('/home/yuqingchi/Code/Yolo-ONNX-TensorRT/runs/train/exp8/weights/best.pt')

video_path = ""
cap = cv2.VideoCapture(0)  # 更改数字，切换不同的摄像头

# loop
while cap.isOpened():
    success, frame = cap.read()

    if success:

        # 记录推理时间
        start = time.perf_counter()

        # Run YOLOv13 inference on the frame
        results = model(frame)

        # Plot the results
        annotated_frame = results[0].plot()

        # 计算 FPS
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        # 在图像右上角显示 FPS
        # 设置文本参数
        fps_text = f"FPS: {fps:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # 绿色
        line_type = 2

        # 获取文本大小（用于右上角对齐）
        text_size, _ = cv2.getTextSize(fps_text, font, font_scale, line_type)
        text_width, text_height = text_size
        margin = 10

        # 图像宽度
        frame_width = annotated_frame.shape[1]

        # 文字起始位置（右上角）
        org = (frame_width - text_width - margin, text_height + margin)

        # 在图像上绘制 FPS 文本
        cv2.putText(annotated_frame, fps_text, org, font, font_scale, font_color, line_type)

        # 显示检测结果
        cv2.imshow("YOLOv13 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()