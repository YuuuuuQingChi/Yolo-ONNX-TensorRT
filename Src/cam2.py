#coding:utf-8
import cv2
from ultralytics import YOLO
import time

# 加载YOLO模型
modle = YOLO('/home/bpz/yolo系列网课/yolov13-main/runs/train/exp2/weights/best.pt')

# 视频路径（留空表示使用摄像头）
video_path = ""
# 初始化摄像头（0表示默认摄像头）
cap = cv2.VideoCapture(2)  

# 主循环
while cap.isOpened():
    # 读取视频帧
    success, frame = cap.read()

    if success:
        # 记录开始时间（用于性能计算）
        start = time.perf_counter()

        # 使用YOLOv8进行目标检测
        results = modle(frame)

        # 记录结束时间
        end = time.perf_counter()

        # 计算处理耗时和帧率
        total_time = end - start
        fps = 1 / total_time

        # 在图像上绘制检测结果
        annotated_frame = results[0].plot()

        # 显示带检测结果的画面
        cv2.imshow("YOLOv:", annotated_frame)

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # 视频读取失败时退出
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()