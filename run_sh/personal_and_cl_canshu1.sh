cd /data/xwx/code/GETNext

CUDA_VISIBLE_DEVICES=2 python train.py \
  --name nyc_quick_validation \
  --data-train dataset/NYC/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/NYC/graph_X.csv \
  --personalized-prior \
  --prior-gate-hidden 64 \
  --prior-gate-dropout 0.1 \
  --ssl \
  --ssl-weight 0.05 \
  --ssl-edge-drop 0.2 \
  --ssl-feat-mask 0.2 \
  --ssl-temp 0.2 \
  --ssl-batch-nodes 512 \
  --batch 32 \
  --epochs 50 \
  --lr 0.001 \
#   --weight-decay 5e-4 \
  --name personal_and_cl_batch_size_16_canshu1 \