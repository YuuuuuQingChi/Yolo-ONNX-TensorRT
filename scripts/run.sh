#!/bin/bash

# 获取当前脚本的绝对路径
current_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# 获取当前脚本所在目录
current_dir=$(dirname "$current_file")
# 获取项目根目录（当前目录的父目录）
project_root=$(dirname "$current_dir")

# 设置PYTHONPATH环境变量（仅对当前终端有效）
export PYTHONPATH="$project_root:$project_root/yolov13:$PYTHONPATH"

echo "Successfully set PYTHONPATH to include project root and yolov13 directory."
