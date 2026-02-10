cd /data/xwx/code/GETNext

CUDA_VISIBLE_DEVICES=3 python train.py \
  --name nyc_run_ssl_personalized \
  --data-train dataset/NYC/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/NYC/graph_X.csv \
  --ssl \
  --ssl-weight 0.1 \
  --ssl-edge-drop 0.1 \
  --ssl-feat-mask 0.1 \
  --ssl-temp 0.2 \
  --ssl-batch-nodes 1024 \
  --name only_cl \
  --batch 16