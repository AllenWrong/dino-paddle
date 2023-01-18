# dino-paddle
Dino paddle implemetion


<a href="https://pan.baidu.com/s/1WzDp_N2QfnkRrSvPYxpv9A?pwd=uyt7 ">full_checkpoint</a>  提取码：uyt7

<a href="https://pan.baidu.com/s/1j6mfxRfZ9Dwu5QqyFSaWIw?pwd=3j5t ">linear_checkpoint</a>  提取码：3j5t

## 使用

### 训练backbone

在`train_vit.sh`设置如下参数：

- --gpus：gpu ids
- --data_path：训练集路径
- --output_dir：日志、模型保存的目录

### 训练分类层

在`train_linear.sh`设置如下参数：

- --gpus：gpu ids

- --data_path：训练集路径
- --pretrained_weights：backbone的预训练模型。默认即可
- --output_dir：日志、模型的保存目录
- --batch_size：32，训练分类层、评估模型时都是32。四卡的话就是4*32=128

### 评估模型

在`eval.sh`中设置如下参数：

- --data_path：验证集路径
- --pretrained_weights：backbone的预训练模型
- --pretrained_linear：分类层的预训练模型

