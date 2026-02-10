cd /data/xwx/code/GETNext

CUDA_VISIBLE_DEVICES=0 python train.py \
  --name nyc_run_ssl_personalized \
  --data-train dataset/NYC/NYC/NYC_train.csv \
  --data-val dataset/NYC/NYC/NYC_val.csv \
  --data-adj-mtx dataset/NYC/NYC/graph_A.csv \
  --data-node-feats dataset/NYC/NYC/graph_X.csv \
  --personalized-prior \
  --prior-gate-hidden 64 \
  --prior-gate-dropout 0.3 \
  --ssl \
  --ssl-weight 0.3 \
  --ssl-edge-drop 0.15 \
  --ssl-feat-mask 0.15 \
  --ssl-temp 0.2 \
  --ssl-batch-nodes 1024 \
  --name personal_and_cl_batch_size_16_canshu \
  --batch 16