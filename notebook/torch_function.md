# expand_as
```py
tensor = tensor.expand_as(example)
```
expand和expand_as只能扩展1个大小的维度

将tensor扩展成example的形状，可以补充维度，例如：将[2,2]扩展成[3,2,2]，他会将原来的数据进行重复，但是必须非扩展维度必须相同