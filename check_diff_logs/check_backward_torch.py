import pickle
from vision_transformer import vit_small, VisionTransformer
from lib import get_args_from, get_args_parser
import utils
from main_dino import DINOHead, DINOLoss
from eval_linear import LinearClassifier
import torch
import os
from reprod_log import ReprodLogger
import warnings
warnings.filterwarnings('ignore')
import numpy as np


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint[key].items()}
                msg = value.load_state_dict(state_dict, strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


# load data
with open("../data/8_images.pkl", "rb") as f:
    images: list = pickle.load(f)

# convert to tensor
images = [torch.tensor(it) for it in images]
LEN_DATALOADER = 1

# laod args
args = get_args_from("./args.json", get_args_parser())
args.epochs = 40

# build backbone
student_bkb = vit_small(drop_path_rate=args.drop_path_rate).cuda()
teacher_bkb = vit_small().cuda()
embed_dim = student_bkb.embed_dim
utils.fix_random_seeds(10)

# wrapper
student = utils.MultiCropWrapper(student_bkb, DINOHead(
    embed_dim,
    args.out_dim,
    use_bn=args.use_bn_in_head,
    norm_last_layer=args.norm_last_layer,
))
teacher = utils.MultiCropWrapper(
    teacher_bkb,
    DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
)
# move networks to gpu
student, teacher = student.cuda(), teacher.cuda()

teacher_without_ddp = teacher
teacher_without_ddp.load_state_dict(student.state_dict())
# no backward on teacher
for p in teacher.parameters():
    p.requires_grad = False

dino_loss = DINOLoss(
    args.out_dim,
    args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
    args.warmup_teacher_temp,
    args.teacher_temp,
    args.warmup_teacher_temp_epochs,
    args.epochs,
).cuda()

params_groups = utils.get_params_groups(student)
optimizer = torch.optim.AdamW(params_groups)
lr_schedule = utils.cosine_scheduler(
    args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
    args.min_lr,
    args.epochs, LEN_DATALOADER,
    warmup_epochs=args.warmup_epochs,
)
wd_schedule = utils.cosine_scheduler(
    args.weight_decay,
    args.weight_decay_end,
    args.epochs, LEN_DATALOADER,
)
momentum_schedule = utils.cosine_scheduler(
    args.momentum_teacher, 1,
    args.epochs, LEN_DATALOADER
)

to_restore = {"epoch": 0}
restart_from_checkpoint(
    "../weights/dino_deitsmall16_pretrain_full_checkpoint.pth",
    run_variables=to_restore,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    fp16_scaler=None,
    dino_loss=dino_loss,
)
start_epoch = 0

def log_state():
    bkb_log = ReprodLogger()
    i = 0
    for k, v in teacher_bkb.state_dict().items():
        bkb_log.add(f"tea_state_{i}", v.detach().cpu().numpy())
        i += 1

    i = 0
    for k, v in student_bkb.state_dict().items():
        bkb_log.add(f"stu_state_{i}", v.detach().cpu().numpy())
        i += 1
    bkb_log.save("../reprod_out/th_bkb_state.npy")


def log_lr():
    lr_log = ReprodLogger()
    lr_log.add("lr", lr_schedule)
    lr_log.add("wd", wd_schedule)
    lr_log.add("momentum_schedule", momentum_schedule)
    lr_log.save("../reprod_out/lr_th.npy")


def backbone_one_epoch(epoch, data, reprod_log):
    it = LEN_DATALOADER * epoch + 0
    
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedule[it]
        # if i == 0:  # only the first group is regularized
        #     param_group["weight_decay"] = wd_schedule[it]

    images = [it.cuda(non_blocking=True) for it in data]
    teacher_output = teacher(images[:2])    
    student_output = student(images)

    # for i in range(2):
    #     arr = teacher_bkb(images[i]).detach().cpu().numpy()
    #     reprod_log.add(f"bkb_fea_{i}", arr)
    #
    # for i in range(2, 12):
    #     arr = student_bkb(images[i]).detach().cpu().numpy()
    #     reprod_log.add(f"bkb_stu_{i}", arr)

    for i in range(len(teacher_output)):
        arr = teacher_output[i].detach().cpu().numpy()
        reprod_log.add(f"tea_out_{i}", arr)

    for i in range(len(student_output)):
        arr = student_output[i].detach().cpu().numpy()
        reprod_log.add(f"stu_out_{i}", arr)

    loss = dino_loss(student_output, teacher_output, epoch)
    reprod_log.add(f"center_epoch_{epoch}", dino_loss.center.detach().cpu().numpy())
    reprod_log.add(f"loss_epoch_{epoch}", loss.detach().cpu().numpy())

    optimizer.zero_grad()
    loss.backward()
    if args.clip_grad:
        param_norms = utils.clip_gradients(student, args.clip_grad)
    utils.cancel_gradients_last_layer(epoch, student,
                                      args.freeze_last_layer)

    # log grad
    # iter = student.parameters().__iter__()
    # idx = 0
    # for p in iter:
    #     if p.grad is not None:
    #         print("idx", idx)
    #         reprod_log.add(f"grad_{idx}_epoch_{epoch}", p.grad.detach().cpu().numpy())
    #         idx += 1
    #     if idx == 156:
    #         break
    #
    # last_layer_iter = student.head.last_layer.parameters().__iter__()
    # last_1 = last_layer_iter.__next__()
    # last_2 = last_layer_iter.__next__()
    #
    # if last_2.grad is not None:
    #     print("idx", idx)
    #     reprod_log.add(f"grad_{idx}_epoch_{epoch}", last_2.grad.detach().cpu().numpy())
    #     idx += 1
    # else:
    #     print("idx", idx)
    #     reprod_log.add(f"grad_{idx}_epoch_{epoch}", np.zeros(last_2.shape))
    #     idx += 1
    #
    # if last_1.grad is not None:
    #     print("idx", idx)
    #     reprod_log.add(f"grad_{idx}_epoch_{epoch}", last_1.grad.detach().cpu().numpy())
    #     idx += 1
    # else:
    #     print("idx", idx)
    #     reprod_log.add(f"grad_{idx}_epoch_{epoch}", np.zeros(last_1.shape))
    #     idx += 1

    optimizer.step()

    # EMA update for the teacher
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    print(f"epoch: {epoch}, loss: {loss.item()}")


# ============================================================
num_labels = 1000
lr = 0.01
batch_size = 1024
epochs = 40
last_blocks = 4

linear_classifier = LinearClassifier(embed_dim * last_blocks, num_labels=num_labels).cuda()
# set optimizer
clf_opt = torch.optim.SGD(
    linear_classifier.parameters(),
    lr * batch_size / 256., # linear scaling rule
    momentum=0.9,
    weight_decay=0, # we do not apply weight decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clf_opt, epochs, eta_min=0)
model = vit_small(patch_size=16)
utils.load_pretrained_weights(
    model,
    "../weights/dino_deitsmall16_pretrain_full_checkpoint.pth",
    "teacher",
    "vit_small",
    16
)
lin_state = torch.load("../weights/dino_deitsmall16_linearweights.pth")['state_dict']
lin_state = {k.replace("module.", "") : v for k, v in lin_state.items()}
linear_classifier.load_state_dict(lin_state)

model.cuda()
model.eval()

label = np.load("../data/8_labels.npy")
label = torch.tensor(label).cuda()

hook_module_idx = 0  # used for logger


def check_sub_m_out():
    main_log = ReprodLogger()

    def clf_one_epoch(epoch, data):
        images = data[0].cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(images, last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            main_log.add(f"output_epoch_{epoch}", output.detach().cpu().numpy())

    for epoch in range(5):

        def hook(module, input, output):
            global hook_module_idx
            if isinstance(output, tuple):
                for it in output:
                    main_log.add(f"module_{hook_module_idx}", it.detach().cpu().numpy())
            else:
                main_log.add(f"module_{hook_module_idx}", output.detach().cpu().numpy())
            hook_module_idx += 1

        handles = []
        # set hooks
        handles.append(model.patch_embed.register_forward_hook(hook))
        handles.append(model.blocks[0].register_forward_hook(hook))
        handles.append(model.blocks[5].register_forward_hook(hook))
        handles.append(model.blocks[11].register_forward_hook(hook))

        clf_one_epoch(epoch, images)

        # remove handle
        for hd in handles:
            hd.remove()

    main_log.save("../reprod_out/sub_m_th.npy")


def check_clf_backward():
    lr_log = ReprodLogger()
    main_log = ReprodLogger()

    def clf_one_epoch(epoch, data):
        linear_classifier.train()
        images = data[0].cuda(non_blocking=True)
        lr_log.add(f"lr_epoch_{epoch}", np.array([clf_opt.param_groups[0]['lr']]))

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(images, last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

        output = linear_classifier(output)
        main_log.add(f"output_epoch_{epoch}", output.detach().cpu().numpy())
        main_log.add(f"pred_epoch_{epoch}", output.detach().cpu().numpy().max(axis=1))

        # compute cross entropy loss
        loss = torch.nn.CrossEntropyLoss()(output, label)
        print(f"epoch: {epoch}, loss: {loss.item()}")
        main_log.add(f"loss_epoch_{epoch}", loss.detach().cpu().numpy())

        # compute the gradients
        clf_opt.zero_grad()
        loss.backward()

        # grad
        iter = linear_classifier.parameters().__iter__()
        idx = 0
        for p in iter:
            main_log.add(f"grad_{idx}_epoch_{epoch}", p.grad.detach().cpu().numpy())
            idx += 1

        # step
        clf_opt.step()

    for epoch in range(5):
        clf_one_epoch(epoch, images)

    lr_log.save("../reprod_out/clf_lr_th.npy")
    main_log.save("../reprod_out/clf_log_th.npy")


def check_backbone_backward():
    student.eval()
    teacher.eval()
    teacher_without_ddp.eval()

    reprod_log = ReprodLogger()
    for epoch in range(10):
        backbone_one_epoch(epoch, images, reprod_log)
    reprod_log.save(f"../reprod_out/th_backward.npy")


if __name__ == "__main__":
    # check_clf_backward()
    check_backbone_backward()
    # check_sub_m_out()
