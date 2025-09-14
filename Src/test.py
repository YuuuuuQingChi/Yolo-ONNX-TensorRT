import torch
import onnxruntime
test = torch.tensor([[1, 2], [3, 4]])  # 形状 [2, 2]
example = torch.ones(3, 3, 3)          # 形状 [3, 3, 3]

# 解决方案：先添加一个维度，再扩展
test = test.unsqueeze(0)  # 形状变为 [1, 2, 2]
test = test.expand(3, 3, 3)  # 现在可以扩展为 [3, 3, 3]
