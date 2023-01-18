
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    train.py --patch_size 16 --epochs 100 \
    --data_path ... \
    --batch_size 1024 \
    --output_dir ./out