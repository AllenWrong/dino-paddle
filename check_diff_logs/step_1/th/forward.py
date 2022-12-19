from vision_transformer import vit_small, DINOHead
from dino.eval_linear import LinearClassifier
from pipeline.seed_set import setup_seed_th
import numpy as np
import torch
from torch import nn
from reprod_log import ReprodLogger
from pipeline.lib import get_eval_args, get_args_from
import utils
from pipeline.lib import get_sub_dict
from dino.main_dino import DINOLoss


def forward_lin():
    reprod_logger = ReprodLogger()

    args = get_eval_args()
    vits16 = vit_small(16)
    embed_dim = vits16.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    lin = LinearClassifier(embed_dim, args.num_labels)

    fake_data = np.load("../../fake_data/fake_data.npy")
    x = torch.tensor(fake_data).cuda()
    y = torch.tensor([0]).cuda()

    ckp_path = "../../../weights/dino_deitsmall16_pretrain_full_checkpoint.pth"
    backbone_dict, _ = get_sub_dict(ckp_path)

    vits16.load_state_dict(backbone_dict)

    lin_state_dict = torch.load("../../../weights/dino_deitsmall16_linearweights.pth")['state_dict']
    lin_state_dict = {k.replace("module.", "") : v for k, v in lin_state_dict.items()}
    lin.load_state_dict(lin_state_dict)
    vits16.eval()
    lin.eval()
    vits16.cuda()
    lin.cuda()

    intermediate_output = vits16.get_intermediate_layers(x, args.n_last_blocks)
    output = torch.cat(
        [x[:, 0] for x in intermediate_output], dim=-1
    )

    output = lin(output)
    reprod_logger.add("logits", output.cpu().detach().numpy())

    loss = nn.CrossEntropyLoss()(output, y)
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")


def forward_dino_loss():
    losses_log = ReprodLogger()

    # load fake mv data
    stu_input = np.load('../../fake_data/stu_input.npy')
    tea_input = np.load('../../fake_data/tea_input.npy')
    stu_input = torch.tensor(stu_input)
    tea_input = torch.tensor(tea_input)

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
    losses = []
    iter_num = stu_input.shape[0]
    epoch = 4
    for e in range(epoch):
        loss: torch.Tensor = 0
        for i in range(iter_num):
            loss += loss_f(stu_input[i], tea_input[i], e)
        loss /= iter_num
        losses.append(loss.item())

    losses_log.add("dino_losses", np.array(losses))
    losses_log.add("center", loss_f.center.detach().cpu().numpy())
    losses_log.save("forward_dino_loss_th.npy")


if __name__ == "__main__":
    setup_seed_th(10)
    reprod_logger = ReprodLogger()
    forward_lin()
    # forward_dino_loss()
