

==== pretrain backbone ====
- 前向
  - 代码：pipline/step_1
  - 效果：
    - logits（output层）差异为1e-6
    - 固定loss的输入。loss前向差异为：1e-7。
    - 使用模型的输出计算loss: 差异为1e-5

- 前向传播时，中间层检查
  - 代码：pd/check_backward.py: check_sub_m_out
  - 效果：中间层差异为1e-7，但是output层差异为1e-6（主要是LayerNorm带来的误差）
  - Note：有理由认为是LayerNorm造成的误差导致output(logits)和loss的误差较大

- 反向
  - 梯度检查  待做
  - 效果：


==== linear clf ====
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
