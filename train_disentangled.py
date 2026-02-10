"""
使用时空解耦双塔模型的训练脚本
完整实现论文方法论
"""
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings
from disentangled_model import DisentangledDualTowerModel, orthogonality_loss
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]
    
    # POI idx to coordinates
    poi_idx2coords_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx = poi_id2idx_dict[row['node_name/poi_id']]
        lat = row[args.feature3]
        lon = row[args.feature4]
        poi_idx2coords_dict[poi_idx] = (lat, lon)

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # Print user-trajectories count
    traj_list = list(set(train_df['trajectory_id'].tolist()))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()
                
                # 获取经纬度坐标
                latitudes = traj_df[args.feature3].to_list()
                longitudes = traj_df[args.feature4].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i], latitudes[i], longitudes[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1], latitudes[i + 1], longitudes[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()
                latitudes = traj_df[args.feature3].to_list()
                longitudes = traj_df[args.feature4].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i], latitudes[i], longitudes[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1], latitudes[i + 1], longitudes[i + 1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model (GCN for global collaborative information)
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)

    # Node Attn Model (for graph-based adjustment)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    # User embedding model
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # Time Model
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # Disentangled Dual-Tower Model
    logging.info("=" * 50)
    logging.info("使用时空解耦双塔模型 (Disentangled Dual-Tower Model)")
    logging.info("=" * 50)
    
    dual_tower_model = DisentangledDualTowerModel(
        num_pois=num_pois,
        num_cats=num_cats,
        poi_embed_dim=args.poi_embed_dim,
        cat_embed_dim=args.cat_embed_dim,
        time_embed_dim=args.time_embed_dim,
        user_embed_dim=args.user_embed_dim,
        gcn_poi_embed_dim=args.poi_embed_dim,
        hidden_dim=args.dual_tower_hidden_dim,
        num_fourier_freq=args.num_fourier_freq,
        num_pref_heads=args.num_pref_heads,
        num_pref_layers=args.num_pref_layers,
        num_spatial_layers=args.num_spatial_layers,
        initial_sigma=args.initial_sigma,
        dropout=args.transformer_dropout
    )
    
    logging.info(f"双塔模型参数量: {sum(p.numel() for p in dual_tower_model.parameters()):,}")

    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(dual_tower_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def prepare_batch_data(batch, poi_embeddings):
        """准备批次数据"""
        batch_input_seqs = []
        batch_seq_lens = []
        batch_poi_indices = []
        batch_cat_indices = []
        batch_coords = []
        batch_time_embeds = []
        batch_gcn_poi_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_time = []
        batch_seq_labels_cat = []
        batch_user_embeds = []

        for sample in batch:
            traj_id = sample[0]
            input_seq = sample[1]
            label_seq = sample[2]
            
            # 提取POI索引
            poi_idxs = [each[0] for each in input_seq]
            # 提取时间特征
            time_features = [each[1] for each in input_seq]
            # 提取坐标
            coords = [(each[2], each[3]) for each in input_seq]  # (lat, lon)
            
            # 标签
            label_poi_idxs = [each[0] for each in label_seq]
            label_time_features = [each[1] for each in label_seq]
            label_cat_idxs = [poi_idx2cat_idx_dict[each] for each in label_poi_idxs]
            
            # 输入类别
            input_cat_idxs = [poi_idx2cat_idx_dict[each] for each in poi_idxs]
            
            # 用户嵌入
            user_id = traj_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            
            # 时间嵌入
            time_embeds = []
            for t in time_features:
                t_embed = time_embed_model(torch.tensor([t], dtype=torch.float).to(device=args.device))
                time_embeds.append(t_embed.squeeze(0))
            time_embeds = torch.stack(time_embeds)  # (seq_len, time_embed_dim)
            
            # GCN POI嵌入
            gcn_poi_embeds = []
            for poi_idx in poi_idxs:
                gcn_poi_embeds.append(poi_embeddings[poi_idx])
            gcn_poi_embeds = torch.stack(gcn_poi_embeds)  # (seq_len, poi_embed_dim)
            
            # 保存到batch
            batch_input_seqs.append(poi_idxs)
            batch_seq_lens.append(len(poi_idxs))
            batch_poi_indices.append(torch.LongTensor(poi_idxs))
            batch_cat_indices.append(torch.LongTensor(input_cat_idxs))
            batch_coords.append(torch.FloatTensor(coords))
            batch_time_embeds.append(time_embeds)
            batch_gcn_poi_embeds.append(gcn_poi_embeds)
            batch_seq_labels_poi.append(torch.LongTensor(label_poi_idxs))
            batch_seq_labels_time.append(torch.FloatTensor(label_time_features))
            batch_seq_labels_cat.append(torch.LongTensor(label_cat_idxs))
        
        # Pad sequences
        batch_poi_indices_padded = pad_sequence(batch_poi_indices, batch_first=True, padding_value=-1)
        batch_cat_indices_padded = pad_sequence(batch_cat_indices, batch_first=True, padding_value=-1)
        batch_coords_padded = pad_sequence(batch_coords, batch_first=True, padding_value=0.0)
        batch_time_embeds_padded = pad_sequence(batch_time_embeds, batch_first=True, padding_value=0.0)
        batch_gcn_poi_embeds_padded = pad_sequence(batch_gcn_poi_embeds, batch_first=True, padding_value=0.0)
        label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
        label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
        label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
        
        return {
            'poi_indices': batch_poi_indices_padded.to(device=args.device, dtype=torch.long),
            'cat_indices': batch_cat_indices_padded.to(device=args.device, dtype=torch.long),
            'coords': batch_coords_padded.to(device=args.device, dtype=torch.float),
            'time_embeds': batch_time_embeds_padded.to(device=args.device, dtype=torch.float),
            'gcn_poi_embeds': batch_gcn_poi_embeds_padded.to(device=args.device, dtype=torch.float),
            'label_poi': label_padded_poi.to(device=args.device, dtype=torch.long),
            'label_time': label_padded_time.to(device=args.device, dtype=torch.float),
            'label_cat': label_padded_cat.to(device=args.device, dtype=torch.long),
            'seq_lens': batch_seq_lens,
            'input_seqs': batch_input_seqs
        }

    def adjust_pred_prob_by_graph(y_pred_poi, batch_input_seqs, batch_seq_lens):
        """使用图注意力调整预测概率"""
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    dual_tower_model = dual_tower_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    train_epochs_ortho_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    val_epochs_ortho_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        dual_tower_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        train_batches_ortho_loss_list = []
        
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            # Get GCN POI embeddings
            poi_embeddings = poi_embed_model(X, A)
            
            # Prepare batch data
            batch_data = prepare_batch_data(batch, poi_embeddings)
            
            # Generate causal mask
            seq_len = batch_data['poi_indices'].shape[1]
            mask = dual_tower_model.generate_square_subsequent_mask(seq_len, args.device)
            
            # Forward pass through dual-tower model
            poi_logits, time_pred, cat_logits, pref_features, spatial_features, gate_values = dual_tower_model(
                poi_indices=batch_data['poi_indices'],
                cat_indices=batch_data['cat_indices'],
                coords=batch_data['coords'],
                time_embeddings=batch_data['time_embeds'],
                gcn_poi_embeddings=batch_data['gcn_poi_embeds'],
                mask=mask
            )
            
            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(
                poi_logits,
                batch_data['input_seqs'],
                batch_data['seq_lens']
            )
            
            # Calculate losses
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), batch_data['label_poi'])
            loss_time = criterion_time(torch.squeeze(time_pred), batch_data['label_time'])
            loss_cat = criterion_cat(cat_logits.transpose(1, 2), batch_data['label_cat'])
            
            # Orthogonality constraint loss
            loss_ortho = orthogonality_loss(pref_features, spatial_features)
            
            # Total loss
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat + args.ortho_loss_weight * loss_ortho
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = batch_data['label_poi'].detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            batch_pred_times = time_pred.detach().cpu().numpy()
            batch_pred_cats = cat_logits.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_data['seq_lens']):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            train_batches_ortho_loss_list.append(loss_ortho.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = poi_logits.detach().cpu().numpy()
                
                # 获取当前批次的sigma值
                sigma_values = []
                for layer in dual_tower_model.spatial_tower.spatial_attention_layers:
                    sigma_values.append(torch.exp(layer.log_sigma).item())
                
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_ortho_loss:{np.mean(train_batches_ortho_loss_list):.4f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'spatial_sigma_values:{[f"{s:.4f}" for s in sigma_values]}\n'
                             f'gate_mean:{gate_values.mean().item():.4f}, gate_std:{gate_values.std().item():.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_data["seq_lens"][sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_data["seq_lens"][sample_idx]])} \n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_data["seq_lens"][sample_idx]])} \n'
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_data["seq_lens"][sample_idx]])} \n' +
                             '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        dual_tower_model.eval()
        
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        val_batches_ortho_loss_list = []
        
        for vb_idx, batch in enumerate(val_loader):
            with torch.no_grad():
                # Get GCN POI embeddings
                poi_embeddings = poi_embed_model(X, A)
                
                # Prepare batch data
                batch_data = prepare_batch_data(batch, poi_embeddings)
                
                # Generate causal mask
                seq_len = batch_data['poi_indices'].shape[1]
                mask = dual_tower_model.generate_square_subsequent_mask(seq_len, args.device)
                
                # Forward pass through dual-tower model
                poi_logits, time_pred, cat_logits, pref_features, spatial_features, gate_values = dual_tower_model(
                    poi_indices=batch_data['poi_indices'],
                    cat_indices=batch_data['cat_indices'],
                    coords=batch_data['coords'],
                    time_embeddings=batch_data['time_embeds'],
                    gcn_poi_embeddings=batch_data['gcn_poi_embeds'],
                    mask=mask
                )
                
                # Graph Attention adjusted prob
                y_pred_poi_adjusted = adjust_pred_prob_by_graph(
                    poi_logits,
                    batch_data['input_seqs'],
                    batch_data['seq_lens']
                )
                
                # Calculate losses
                loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), batch_data['label_poi'])
                loss_time = criterion_time(torch.squeeze(time_pred), batch_data['label_time'])
                loss_cat = criterion_cat(cat_logits.transpose(1, 2), batch_data['label_cat'])
                loss_ortho = orthogonality_loss(pref_features, spatial_features)
                loss = loss_poi + loss_time * args.time_loss_weight + loss_cat + args.ortho_loss_weight * loss_ortho

                # Performance measurement
                top1_acc = 0
                top5_acc = 0
                top10_acc = 0
                top20_acc = 0
                mAP20 = 0
                mrr = 0
                batch_label_pois = batch_data['label_poi'].detach().cpu().numpy()
                batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
                batch_pred_times = time_pred.detach().cpu().numpy()
                batch_pred_cats = cat_logits.detach().cpu().numpy()
                for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_data['seq_lens']):
                    label_pois = label_pois[:seq_len]
                    pred_pois = pred_pois[:seq_len, :]
                    top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                    top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                    top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                    top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                    mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                    mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
                val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
                val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
                val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
                val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
                val_batches_mrr_list.append(mrr / len(batch_label_pois))
                val_batches_loss_list.append(loss.detach().cpu().numpy())
                val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
                val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
                val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
                val_batches_ortho_loss_list.append(loss_ortho.detach().cpu().numpy())

                # Report validation progress
                if (vb_idx % (args.batch * 2)) == 0:
                    sample_idx = 0
                    batch_pred_pois_wo_attn = poi_logits.detach().cpu().numpy()
                    logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                                 f'val_batch_loss:{loss.item():.2f}, '
                                 f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                                 f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                                 f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                                 f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                                 f'val_move_ortho_loss:{np.mean(val_batches_ortho_loss_list):.4f} \n'
                                 f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                                 f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                                 f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                                 f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                                 f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                                 f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                                 f'traj_id:{batch[sample_idx][0]}\n'
                                 f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_data["seq_lens"][sample_idx]])} \n'
                                 f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_data["seq_lens"][sample_idx]])} \n' +
                                 '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_train_ortho_loss = np.mean(train_batches_ortho_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)
        epoch_val_ortho_loss = np.mean(val_batches_ortho_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_ortho_loss_list.append(epoch_train_ortho_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_ortho_loss_list.append(epoch_val_ortho_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_ortho_loss:{epoch_train_ortho_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_time_loss: {epoch_val_time_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_ortho_loss: {epoch_val_ortho_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'dual_tower_state_dict': dual_tower_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                'poi_idx2coords_dict': poi_idx2coords_dict,
                'node_attn_map': node_attn_model(X, A),
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_ortho_loss': epoch_train_ortho_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_mrr': epoch_train_mrr
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_ortho_loss': epoch_val_ortho_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_mrr': epoch_val_mrr
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_ortho_loss_list={[float(f"{each:.4f}") for each in train_epochs_ortho_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_ortho_loss_list={[float(f"{each:.4f}") for each in val_epochs_ortho_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)


if __name__ == '__main__':
    args = parameter_parser()
    # Select device (support --no-cuda and handle CUDA arch incompatibility gracefully)
    if getattr(args, "no_cuda", False):
        args.device = torch.device("cpu")
    else:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                major, minor = torch.cuda.get_device_capability()
                arch = f"sm_{major}{minor}"
                arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
                if arch_list and arch not in arch_list:
                    print(
                        f"[WARN] GPU arch {arch} is not supported by this PyTorch build ({arch_list}). "
                        f"Falling back to CPU. (Tip: install a newer PyTorch build with CUDA support for {arch}.)"
                    )
                    use_cuda = False
            except Exception as e:
                print(f"[WARN] CUDA probing failed ({e}); falling back to CPU.")
                use_cuda = False
        args.device = torch.device("cuda") if use_cuda else torch.device("cpu")
    
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    
    # Update data paths to correct location
    args.data_adj_mtx = 'dataset/NYC/NYC/graph_A.csv'
    args.data_node_feats = 'dataset/NYC/NYC/graph_X.csv'
    args.data_train = 'dataset/NYC/NYC/NYC_train.csv'
    args.data_val = 'dataset/NYC/NYC/NYC_val.csv'
    
    train(args)
