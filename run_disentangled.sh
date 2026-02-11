#!/bin/bash

# 时空解耦双塔模型训练脚本
# 使用方法: bash run_disentangled.sh

echo "=========================================="
echo "启动时空解耦双塔模型训练"
echo "=========================================="

python train_disentangled.py \
    --data-train dataset/NYC/NYC/NYC_train.csv \
    --data-val dataset/NYC/NYC/NYC_val.csv \
    --data-adj-mtx dataset/NYC/NYC/graph_A.csv \
    --data-node-feats dataset/NYC/NYC/graph_X.csv \
    --time-units 48 \
    --time-feature norm_in_day_time \
    --poi-embed-dim 128 \
    --user-embed-dim 128 \
    --time-embed-dim 32 \
    --cat-embed-dim 32 \
    --node-attn-nhid 128 \
    --dual-tower-hidden-dim 256 \
    --num-fourier-freq 10 \
    --num-pref-heads 4 \
    --num-pref-layers 2 \
    --num-spatial-layers 2 \
    --initial-sigma 1.0 \
    --ortho-loss-weight 0.01 \
    --transformer-dropout 0.1 \
    --batch 16 \
    --epochs 200 \
    --lr 0.001 \
    --name disentangled_exp1

echo "=========================================="
echo "训练完成！"
echo "结果保存在: runs/train/disentangled_exp1/"
echo "=========================================="
