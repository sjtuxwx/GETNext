#!/usr/bin/env python3
"""
检查过拟合程度的脚本
"""

import sys
import os

def read_metrics(file_path):
    """读取metrics文件"""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    metrics[key] = eval(value)
        return metrics
    except:
        return None

def check_overfit(exp_name):
    """检查指定实验的过拟合情况"""
    base_path = f'runs/train/{exp_name}'
    
    train_metrics = read_metrics(f'{base_path}/metrics-train.txt')
    val_metrics = read_metrics(f'{base_path}/metrics-val.txt')
    
    if not train_metrics or not val_metrics:
        print(f'❌ 无法读取实验 {exp_name} 的metrics文件')
        return
    
    print(f'\n{"="*60}')
    print(f'实验: {exp_name}')
    print(f'{"="*60}')
    
    epochs = len(train_metrics['train_epochs_top1_acc_list'])
    
    print(f'\n总共训练了 {epochs} 个 epochs\n')
    
    # 打印每个epoch的对比
    for epoch in range(epochs):
        train_top1 = train_metrics['train_epochs_top1_acc_list'][epoch]
        val_top1 = val_metrics['val_epochs_top1_acc_list'][epoch]
        gap = train_top1 - val_top1
        
        train_top5 = train_metrics['train_epochs_top5_acc_list'][epoch]
        val_top5 = val_metrics['val_epochs_top5_acc_list'][epoch]
        gap5 = train_top5 - val_top5
        
        print(f'Epoch {epoch+1}:')
        print(f'  Top-1: 训练 {train_top1:.4f} vs 验证 {val_top1:.4f} | 差距 {gap:.4f} ({gap*100:.2f}%)')
        print(f'  Top-5: 训练 {train_top5:.4f} vs 验证 {val_top5:.4f} | 差距 {gap5:.4f} ({gap5*100:.2f}%)')
        
        # 判断过拟合程度
        if gap > 0.15:
            print(f'  ⚠️  严重过拟合！')
        elif gap > 0.10:
            print(f'  ⚠️  中度过拟合')
        elif gap > 0.05:
            print(f'  ⚠️  轻微过拟合')
        else:
            print(f'  ✅ 拟合良好')
        print()
    
    # 最后一个epoch的总结
    last_epoch = epochs - 1
    final_train_top1 = train_metrics['train_epochs_top1_acc_list'][last_epoch]
    final_val_top1 = val_metrics['val_epochs_top1_acc_list'][last_epoch]
    final_gap = final_train_top1 - final_val_top1
    
    print(f'{"="*60}')
    print(f'最终状态 (Epoch {epochs}):')
    print(f'{"="*60}')
    print(f'训练集 Top-1: {final_train_top1:.4f} ({final_train_top1*100:.2f}%)')
    print(f'验证集 Top-1: {final_val_top1:.4f} ({final_val_top1*100:.2f}%)')
    print(f'差距: {final_gap:.4f} ({final_gap*100:.2f}%)')
    
    if final_gap > 0.15:
        print(f'\n结论: ❌ 严重过拟合！需要增强正则化')
    elif final_gap > 0.10:
        print(f'\n结论: ⚠️  中度过拟合，建议调整参数')
    elif final_gap > 0.05:
        print(f'\n结论: ⚠️  轻微过拟合，可以接受')
    else:
        print(f'\n结论: ✅ 拟合良好！')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        # 默认检查最近的实验
        exp_name = 'temporal_rotary_exp-5'
    
    check_overfit(exp_name)
    
    # 也可以同时检查多个实验对比
    print(f'\n\n{"#"*60}')
    print('对比其他实验:')
    print(f'{"#"*60}')
    
    for exp in ['exp1-10', 'temporal_rotary_exp-5']:
        check_overfit(exp)
