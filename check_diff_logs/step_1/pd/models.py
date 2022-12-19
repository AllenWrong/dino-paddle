from paddle import nn
import paddle


class LinearClassifier(nn.Layer):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight = paddle.create_parameter(
            shape=self.linear.weight.shape,
            dtype=self.linear.weight.dtype,
            default_initializer=paddle.nn.initializer.Normal(mean=0, std=0.01)
        )
        self.linear.bias = paddle.create_parameter(
            shape=self.linear.bias.shape,
            dtype=self.linear.bias.dtype,
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )

    def forward(self, x):
        # flatten
        x = x.reshape((x.shape[0], -1))

        # linear layer
        return self.linear(x)