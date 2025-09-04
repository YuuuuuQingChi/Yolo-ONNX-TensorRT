# 安装显卡驱动
```
ubuntu-drivers devices
```
找到带有recommended

```
sudo apt install 带有recommended的
例如sudo apt install nvidia-driver-575 - distro non-free

```

# 下载anaconda
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

这个清华源，下载anaconda3

使用如下命令

```
bash 文件路径
```

# 安装cuda+torch
这个版本
torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl
```
pip install torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl
```
# 安装labelIMg
安装libelImg
conda create -n use_labelimg python=3.6

conda activate use_labelimg

pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple

执行命令打开：labelImg

# YoloV13
https://github.com/iMoonLab/yolov13
按照这个教程走就行

但是里面的torch和torchvision要手动安装，避免它自动安装错误