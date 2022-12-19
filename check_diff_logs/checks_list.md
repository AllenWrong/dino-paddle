## 文件结构

- `bkb_diff.log`：模型加载预训练参数后，对模型的参数进行检查。（这块为了检查参数加载时是否出现问题，意义不大）

- `sub_m_diff.log`：追踪了前向过程中某几个中间层的输出。

- `diff.log`：模型组网前向对齐检查日志。追踪了线性层的输出和损失值。

- `dino_loss_diff.log`：dino损失函数的前向对齐检查日志。追踪了loss值和center值。

- `clf_diff.log`：分类层反向过程检查日志。追踪了线性层输出，预测值，损失值，线性层的梯度。
- `backward_diff.log`：vit反向对齐日志。追踪了student的输出，teacher的输出，loss值，loss中的center值。（2 epochs）。

- `clf_lr_diff.log`：线性分类层训练时的学习率对齐检查日志。

- `lr_diff.log`：vit训练时学习率、权重衰减、动量值对齐检查日志。



## vit backbone部分的检查

- 前向
  - 代码：pipline/step_1
  - 效果：
    - logits（output层）差异为1e-6
    - 固定loss的输入。loss前向差异为：1e-7。
    - 使用模型的输出计算loss: 差异为1e-5

- 前向传播时，中间层检查
  - 代码：pd/check_backward.py: check_sub_m_out
  - 效果：中间层差异为1e-7，但是output层差异为1e-6
  - Note：有理由认为是LayerNorm造成的误差导致output(logits)和loss的误差较大

- 反向
  - 梯度检查  待做
  - 效果：

## 线性分类层部分的检查

- 前向
  - 代码：pipline/step_1
  - 效果：logits（Backbone的输出）：1e-6，loss：1e-5

- 反向、梯度检查
  - 代码：pd/check_backward.py: check_clf_backward
  - 效果：
    - 学习率：0差异
    - 梯度：0差异
    - 损失：loss差异较小
    - 线性层输出：1e-6（考虑主要是因为LayerNorm带来的误差）
    - 预测结果：1e-5，随着训练的进行，差异在增加
