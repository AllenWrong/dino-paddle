import paddle
import paddle.nn as nn

def vits8():
    return ViT(patch_size=8)

class ViT(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,):
        super(ViT, self).__init__()
        #creat patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(image_size, 
                                              patch_size, 
                                              in_channels, 
                                              embed_dim, 
                                              dropout)

        #creat multi head self-attention layers encoder
        self.encoder = Encoder( embed_dim,
                                num_heads, 
                                qkv_bias,
                                mlp_ratio,
                                dropout, 
                                attention_dropout,
                                depth )

        #classifier head for num classes
        self.classifier = Classify(embed_dim, dropout, num_classes)

    def forward(self, x):
        # input [N, C, H', W']
        x = self.patch_embedding(x) #[N, C * H + 1, embed_dim]
        x = self.encoder(x)         #[N, C * H + 1, embed_dim]
        # x = self.classifier(x[:, 0, :])      #[N, num_classes]

        return x


class PatchEmbedding(nn.Layer):
    def __init__(self,
                image_size = 224,
                patch_size = 16,
                in_channels = 3,
                embed_dim = 768,
                dropout = 0.):
        super(PatchEmbedding, self).__init__()

        n_patches = (image_size // patch_size) * (image_size // patch_size) #14 * 14 = 196(个)

        self.patch_embedding = nn.Conv2D(in_channels = in_channels,
                                         out_channels = embed_dim,
                                         kernel_size = patch_size,
                                         stride = patch_size)
        
        self.dropout=nn.Dropout(dropout)

        #add class token
        self.cls_token = paddle.create_parameter(
                                        shape = [1, 1, embed_dim],
                                        dtype = 'float32',
                                        default_initializer = paddle.nn.initializer.Constant(0)
                                        #常量初始化参数，value=0， shape=[1, 1, 768]
                                        )

        #add position embedding
        self.position_embeddings = paddle.create_parameter(
                                        shape = [1, n_patches + 1, embed_dim],
                                        dtype = 'float32',
                                        default_initializer = paddle.nn.initializer.TruncatedNormal(std = 0.02)
                                        #随机截断正态（高斯）分布初始化函数
                                        )

    def forward(self, x):
        x = self.patch_embedding(x) #[N, C, H', W',]  to  [N, embed_dim, H, W]卷积层
        x = x.flatten(2)            #[N, embed_dim, H * W]
        x = x.transpose([0, 2, 1])  #[N, H * W, embed_dim]

        cls_token = self.cls_token.expand((x.shape[0], -1, -1)) #[N, 1, embed_dim]
        x = paddle.concat((cls_token, x), axis = 1)             #[N, H * W + 1, embed_dim]
        x = x + self.position_embeddings                        #[N, H * W + 1, embed_dim]
        x = self.dropout(x)

        return x


class Encoder(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads, 
                 qkv_bias,
                 mlp_ratio,
                 dropout, 
                 attention_dropout,
                 depth):
        super(Encoder, self).__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                        num_heads, 
                                        qkv_bias,
                                        mlp_ratio,
                                        dropout, 
                                        attention_dropout)
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)# or nn.Sequential(*layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class EncoderLayer(nn.Layer):
    def __init__(self, 
                 embed_dim,
                 num_heads, 
                 qkv_bias,
                 mlp_ratio,
                 dropout, 
                 attention_dropout
                 ):
        super(EncoderLayer, self).__init__()
        #Multi Head Attention & LayerNorm
        w_attr_1, b_attr_1 = self._init_weights()
        self.attn_norm = nn.LayerNorm(embed_dim, 
                                      weight_attr = w_attr_1,
                                      bias_attr = b_attr_1,
                                      epsilon = 1e-6)
        self.attn = Attention(embed_dim,
                              num_heads,
                              qkv_bias,
                              dropout,
                              attention_dropout)

        #MLP & LayerNorm
        w_attr_2, b_attr_2 = self._init_weights()
        self.mlp_norm = nn.LayerNorm(embed_dim,
                                     weight_attr = w_attr_2,
                                     bias_attr = b_attr_2,
                                     epsilon = 1e-6)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x                   #[N, H * W + 1, embed_dim]
        x = self.attn_norm(x)   #Attention LayerNorm
        x = self.attn(x)        #[N, H * W + 1, embed_dim]
        x = h + x               #Add

        h = x                   #[N, H * W + 1, embed_dim]
        x = self.mlp_norm(x)    #MLP LayerNorm
        x = self.mlp(x)         #[N, H * W + 1, embed_dim]
        x = h + x               #[Add]
        return x


class Attention(nn.Layer):
    def __init__(self,
                 embed_dim, 
                 num_heads, 
                 qkv_bias, 
                 dropout, 
                 attention_dropout):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(embed_dim / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads
        self.scales = self.attn_head_size ** -0.5

        #calculate qkv
        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim, 
                             self.all_head_size * 3, # weight for Q K V
                             weight_attr = w_attr_1,
                             bias_attr = b_attr_1 if qkv_bias else False)

        #calculate proj
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(embed_dim,
                              embed_dim, 
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        #input size  [N, ~, embed_dim]
        new_shape = x.shape[0:2] + [self.num_heads, self.attn_head_size]
        #reshape size[N, ~, head, head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        #transpose   [N, head, ~, head_size]
        return x

    def forward(self, x):
        #input x = [N, H * W + 1, embed_dim]
        qkv = self.qkv(x).chunk(3, axis = -1)           #[N, ~, embed_dim * 3]  list
        q, k, v = map(self.transpose_multihead, qkv)    #[N, head, ~, head_size]
        
        attn = paddle.matmul(q, k, transpose_y = True)  #[N, head, ~, ~]
        attn = self.softmax(attn * self.scales)         #softmax(Q*K/(dk^0.5))
        attn = self.attn_dropout(attn)                  #[N, head, ~, ~]
        
        z = paddle.matmul(attn, v)                      #[N, head, ~, head_size]
        z = z.transpose([0, 2, 1, 3])                   #[N, ~, head, head_size]
        new_shape = z.shape[0:2] + [self.all_head_size]
        z = z.reshape(new_shape)                        #[N, ~, embed_dim]
        z = self.proj(z)                                #[N, ~, embed_dim]
        z = self.proj_dropout(z)                        #[N, ~, embed_dim]

        return z

class Mlp(nn.Layer):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout):
        super(Mlp, self).__init__()
        #fc1
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim, 
                            int(embed_dim * mlp_ratio), 
                            weight_attr = w_attr_1, 
                            bias_attr = b_attr_1)
        #fc2
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                            embed_dim, 
                            weight_attr = w_attr_2, 
                            bias_attr = b_attr_2)

        self.act = nn.GELU()#GELU > ELU > ReLU > sigmod
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())  
            #XavierNormal正态分布的所有层梯度一致，XavierUniform均匀分布的所有成梯度一致。
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=1e-6)) #正态分布的权值和偏置
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)         #[N, ~, embed_dim]
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)         #[N, ~, embed_dim]
        x = self.dropout2(x)
        return x


class Classify(nn.Layer):
    def __init__(self, embed_dim, dropout, num_classes):
        super(Classify, self).__init__()
        #fc1
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim, 
                            embed_dim,
                            weight_attr = w_attr_1,
                            bias_attr = b_attr_1)
        #fc2
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(embed_dim, 
                            num_classes,
                            weight_attr = w_attr_2,
                            bias_attr = b_attr_2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()  

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x