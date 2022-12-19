from vision_transformer import vit_small, DINOHead
from models import LinearClassifier
import numpy as np
import paddle
from reprod_log import ReprodLogger
from utils import MultiCropWrapper
from pipeline.lib import get_eval_args, get_sub_dict_pd, get_args_from
from loss import DINOLoss


def forward_linear():
    reprod_logger = ReprodLogger()

    args = get_eval_args()
    vits16 = vit_small(16)
    embed_dim = vits16.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    lin = LinearClassifier(embed_dim, args.num_labels)

    fake_data = np.load("../../fake_data/fake_data.npy")
    x = paddle.Tensor(fake_data)
    y = paddle.zeros([1,], dtype=paddle.int32)

    ckp_path = "../../../weights/dino_deitsmall16_pretrain_full_checkpoint.pdparams"
    backbone_dict, _ = get_sub_dict_pd(ckp_path)

    vits16.load_dict(backbone_dict)

    lin_ckp_path = "../../../weights/dino_deitsmall16_linearweights.pdparams"
    lin_dict = paddle.load(lin_ckp_path)['state_dict']
    lin_dict = {
        k.replace("module.", "") : v for k, v in lin_dict.items()
    }
    lin.load_dict(lin_dict)
    vits16.eval()
    lin.eval()

    interm_output = vits16.get_intermediate_layers(x, args.n_last_blocks)
    output = paddle.concat(
        [x[:, 0] for x in interm_output],
        axis=-1
    )
    out = lin(output)
    reprod_logger.add("logits", out.numpy())

    loss = paddle.nn.CrossEntropyLoss()(out, y)
    reprod_logger.add("loss", loss.numpy())
    reprod_logger.save("forward_pd.npy")


def forward_dino_loss():
    losses_log = ReprodLogger()

    # load fake mv data
    stu_input = np.load('../../fake_data/stu_input.npy')
    tea_input = np.load('../../fake_data/tea_input.npy')
    stu_input = paddle.Tensor(stu_input)
    tea_input = paddle.Tensor(tea_input)

    # get args
    args = get_args_from("../../../args.json")

    # build loss
    loss_f = DINOLoss(
        args.out_dim,
        10,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    )

    # compute loss
    iter_num = stu_input.shape[0]
    losses = []
    epoch = 4
    for e in range(epoch):
        loss: paddle.Tensor = 0
        for i in range(iter_num):
            loss += loss_f(stu_input[i], tea_input[i], e)
        loss /= iter_num
        losses.append(loss.item())

    losses_log.add("dino_losses", np.array(losses))
    losses_log.add("center", loss_f.center.numpy())
    losses_log.save("forward_dino_loss_pd.npy")


if __name__ == "__main__":
    paddle.seed(10)
    reprod_logger = ReprodLogger()
    forward_linear()
    # forward_dino_loss()
