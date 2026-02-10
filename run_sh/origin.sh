cd /data/xwx/code/GETNext


python train.py --data-train dataset/NYC/NYC_train.csv
                  --data-val dataset/NYC/NYC_val.csv
                  --time-units 48 --time-feature norm_in_day_time
                  --poi-embed-dim 128 --user-embed-dim 128 
                  --time-embed-dim 32 --cat-embed-dim 32
                  --node-attn-nhid 128    
                  --transformer-nhid 1024
                  --transformer-nlayers 2 --transformer-nhead 2
                  --batch 16 --epochs 200 --name exp1