
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    eval_linear.py --patch_size 16 \
    --epochs 100 \
    --data_path ... \
    --pretrained_weights ./out/dino_deitsmall16_pretrain_full_ckp.pdparams \
    --checkpoint_key teacher \
    --batch_size 32 \
    --output_dir ./out
