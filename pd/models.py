from paddle import nn
import paddle
import numpy as np


class MultiCropWrapper(nn.Layer):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]

        idx_crops = paddle.cumsum(paddle.unique_consecutive(
            paddle.to_tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output = 0, paddle.empty((0,))
        for end_idx in idx_crops:
            _out = self.backbone(paddle.concat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]

            # accumulate outputs
            output = paddle.concat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        return self.head(output)


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

class DINOHead(nn.Layer):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1D(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1D(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias_attr=False), dim=1)
        self.last_layer.weight_g.set_value(
            np.ones(self.last_layer.weight_g.shape, dtype="float32")
        )
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = paddle.create_parameter(
                shape=m.weight.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)
            )
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(m.bias.shape, dtype="float32"))

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, axis=-1, p=2)
        x = self.last_layer(x)
        return x
