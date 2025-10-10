# cv2.cvtColor(src,code)
第一个参数是转换的原始图像，第二个是转换的代码

```py
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
为什么用？
cv.imread会将图片读取成BGR格式的，但我们一般喜欢处理RGB，或者神经网络是RGB的
# img = cv2.resize(img,input_size)
这个不多说

# img = cv2.normalize(img.astype(np.float32), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
这个是归一化，alpha是下限 beta是上限

# img = np.transpose（img，（2,0,1））
cv里面没有提供hcw转化cwh等等函数，需要通过transpose来实现

它的逻辑是，引用原输入的图像的轴的索引，新的顺序，以你填入顺序为准

注意：cv2.transpose是矩阵的转置

# img_with_batch = img[np.newaxis, ...]
在开头添加一个新的维度

# cv.rectangle
这个用来绘制矩形框
```py
cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
```
第一个参数是图像

后续是位置、颜色、线条粗细

# cv2.putText(image, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

image：要绘制的图像

label：要绘制的文本

(int(x), int(y) - 5)：文本起始位置

cv2.FONT_HERSHEY_SIMPLEX：字体类型

0.6：字体大小

(255, 255, 255)：文本颜色（白色）

1：文本线条粗细