import argparse
import json

import paddle
from paddle import nn
import paddle.distributed as dist
from vision_transformer import vit_small
import utils
from models import LinearClassifier
from paddle.vision import transforms
import os
from utils import MetricLogger
from paddle.vision import DatasetFolder


def eval_linear(args):
    dist.init_parallel_env()

    # ============ building student and teacher networks  ============
    # only support vit_s8 and vit_s16 currently
    model = vit_small(
        patch_size=args.patch_size, num_classes=0
    )
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    print(f"vit_s{args.patch_size} has build!")
    model.eval()

    # load weight to evaluate
    utils.load_pretrained_weights(model, args.checkpoint_key, args.pretrained_weights)

    linear_clf = LinearClassifier(embed_dim, args.num_labels)
    linear_clf = paddle.DataParallel(linear_clf)

    # ============ preparing data ... ============
    val_transform = transforms.Compose([
        transforms.Resize(size=256, interpolation='bicubic'),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # dataset_val = ImageNet2012Dataset(args.data_path, mode="val", transform=val_transform)
    dataset_val = DatasetFolder(args.data_path, transform=val_transform)
    val_loader = paddle.io.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.evaluate:
        utils.load_linear_clf_weights(linear_clf, args.pretrained_linear)
        test_stats = valid(val_loader, model, linear_clf, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    # train
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = DatasetFolder(args.data_path, transform=train_transform)
    sampler = paddle.io.DistributedBatchSampler(dataset_train, args.batch_size)
    train_loader = paddle.io.DataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=args.num_workers
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    base_lr = args.lr * (args.batch_size / 256)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=args.epochs, eta_min=0)
    optimizer = paddle.optimizer.Momentum(
        parameters=linear_clf.parameters(),
        learning_rate=scheduler,
        momentum=0.9,
        weight_decay=0.0
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_clf,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_stats = train(model, linear_clf, optimizer, train_loader, epoch,
                            args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = valid(val_loader, model, linear_clf, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if dist.get_rank() == 0:
                with open("valid_log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_clf.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                }

                paddle.save(save_dict, os.path.join(args.output_dir, "ckp.pdparams"))

        print("Training of the supervised linear classifier on frozen features completed.\n"
              "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


@paddle.no_grad()
def valid(valid_loader, model, linear_clf, n_last_blocks, avgpool_patchtokens):
    linear_clf.eval()
    metrics_logger = MetricLogger(" ")
    acc1_f = paddle.metric.Accuracy(topk=(1,))
    acc5_f = paddle.metric.Accuracy(topk=(5,))
    for images, y in valid_loader:

        # forward
        with paddle.no_grad():
            intermediate_output = model.get_intermediate_layers(images, n_last_blocks)
            output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
            if avgpool_patchtokens:
                output = paddle.concat(
                    (output.unsqueeze(-1), paddle.mean(intermediate_output[-1][:, 1:], axis=1).unsqueeze(-1)),
                    axis=-1
                )
                output = output.reshape((output.shape[0], -1))

        output = linear_clf(output)
        loss = nn.CrossEntropyLoss()(output, y)

        metrics_logger.update(loss=loss.item())
        acc1 = acc1_f.compute(pred=output, label=y)
        acc5 = acc5_f.compute(pred=output, label=y)
        acc1_f.update(acc1)
        acc5_f.update(acc5)

    acc1 = acc1_f.accumulate()
    acc5 = acc5_f.accumulate()

    log_info = f"Acc @1 {acc1:.4f}, " \
               f"Acc @5 {acc5:.4f}, " \
               f"loss {metrics_logger.loss.global_avg:.4f}"
    print(log_info)
    return {
        "acc1": acc1,
        "acc5": acc5,
    }


def train(model, linear_clf, optimizer, loader, epoch, n, avgpool):
    linear_clf.train()
    metrics_logger = MetricLogger(" ")
    metrics_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    for image, y in loader:

        # forward
        with paddle.no_grad():
            intermediate_output = model.get_intermediate_layers(image, n)
            output = paddle.concat([x[:, 0] for x in intermediate_output], axis=-1)
            if avgpool:
                output = paddle.concat(
                    (output.unsqueeze(-1), paddle.mean(intermediate_output[-1][:, 1:], axis=1).unsqueeze(-1)),
                    axis=-1
                )
                output = output.reshape((output.shape[0], -1))

        output = linear_clf(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, y)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        paddle.device.cuda.synchronize()
        metrics_logger.update(loss=loss.item())
        metrics_logger.update(lr=optimizer._learning_rate.last_lr)

    # gather the stats from all processes
    # metrics_logger.synchronize_between_processes()
    print("Averaged stats: ", metrics_logger)
    return {k: meter.global_avg for k, meter in metrics_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=bool,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../weights/dino_deitsmall16_pretrain_full_checkpoint.pdparams',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--pretrained_linear', default='../weights/dino_deitsmall16_linearweights.pdparams',
                        type=str, help='Path to pretrained linear clf weights.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='../data/small', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)