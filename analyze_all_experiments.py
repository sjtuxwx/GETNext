#!/usr/bin/env python3
"""
全面分析所有实验，找出问题所在
"""
import os

experiments = [
    ('exp1-10', '基准模型 (TransformerModel)'),
    ('temporal_rotary_exp-5', '时间旋转 (无正则化)'),
    ('temporal_rotary_exp-6', '时间旋转 (强正则化)'),
    ('temporal_rotary_exp-7', '时间旋转 (平衡正则化)'),
]

print("="*80)
print("全面实验对比分析")
print("="*80)

results = []

for exp_name, desc in experiments:
    train_path = f'runs/train/{exp_name}/metrics-train.txt'
    val_path = f'runs/train/{exp_name}/metrics-val.txt'
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        continue
    
    # 读取metrics
    train_metrics = {}
    val_metrics = {}
    
    with open(train_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                train_metrics[key] = eval(value)
    
    with open(val_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                val_metrics[key] = eval(value)
    
    # 获取最后一个epoch的结果
    epochs = len(val_metrics['val_epochs_top1_acc_list'])
    
    # 找到验证集最好的epoch
    best_val_idx = val_metrics['val_epochs_top1_acc_list'].index(
        max(val_metrics['val_epochs_top1_acc_list'])
    )
    
    result = {
        'name': exp_name,
        'desc': desc,
        'epochs': epochs,
        'best_epoch': best_val_idx + 1,
        'best_val_top1': val_metrics['val_epochs_top1_acc_list'][best_val_idx],
        'best_val_top5': val_metrics['val_epochs_top5_acc_list'][best_val_idx],
        'best_val_top10': val_metrics['val_epochs_top10_acc_list'][best_val_idx],
        'best_val_map': val_metrics['val_epochs_mAP20_list'][best_val_idx],
        'best_train_top1': train_metrics['train_epochs_top1_acc_list'][best_val_idx],
        'final_val_top1': val_metrics['val_epochs_top1_acc_list'][-1],
        'final_train_top1': train_metrics['train_epochs_top1_acc_list'][-1],
    }
    result['gap'] = result['best_train_top1'] - result['best_val_top1']
    
    results.append(result)

# 按照最佳验证集top1排序
results.sort(key=lambda x: x['best_val_top1'], reverse=True)

print(f"\n{'排名':<4} {'实验':<30} {'最佳Epoch':<10} {'验证Top-1':<12} {'训练Top-1':<12} {'差距':<10}")
print("-"*80)

for i, r in enumerate(results, 1):
    print(f"{i:<4} {r['desc']:<30} {r['best_epoch']:<10} "
          f"{r['best_val_top1']*100:>6.2f}%     {r['best_train_top1']*100:>6.2f}%     "
          f"{r['gap']*100:>5.2f}%")

print("\n" + "="*80)
print("详细对比（最佳epoch的完整指标）")
print("="*80)

baseline = results[0]
print(f"\n🏆 最佳模型: {baseline['desc']}")
print(f"   最佳Epoch: {baseline['best_epoch']}")
print(f"   Top-1:  {baseline['best_val_top1']*100:.2f}%")
print(f"   Top-5:  {baseline['best_val_top5']*100:.2f}%")
print(f"   Top-10: {baseline['best_val_top10']*100:.2f}%")
print(f"   mAP@20: {baseline['best_val_map']*100:.2f}%")
print(f"   过拟合差距: {baseline['gap']*100:.2f}%")

for r in results[1:]:
    print(f"\n📊 {r['desc']}")
    print(f"   最佳Epoch: {r['best_epoch']}")
    print(f"   Top-1:  {r['best_val_top1']*100:.2f}% (vs 基准: {(r['best_val_top1']-baseline['best_val_top1'])*100:+.2f}%)")
    print(f"   Top-5:  {r['best_val_top5']*100:.2f}% (vs 基准: {(r['best_val_top5']-baseline['best_val_top5'])*100:+.2f}%)")
    print(f"   Top-10: {r['best_val_top10']*100:.2f}% (vs 基准: {(r['best_val_top10']-baseline['best_val_top10'])*100:+.2f}%)")
    print(f"   mAP@20: {r['best_val_map']*100:.2f}% (vs 基准: {(r['best_val_map']-baseline['best_val_map'])*100:+.2f}%)")
    print(f"   过拟合差距: {r['gap']*100:.2f}%")

print("\n" + "="*80)
print("核心问题诊断")
print("="*80)

# 找出时间旋转模型
rotary_models = [r for r in results if 'rotary' in r['name']]
baseline_model = [r for r in results if r['name'] == 'exp1-10'][0]

best_rotary = max(rotary_models, key=lambda x: x['best_val_top1'])

print(f"\n最佳时间旋转模型: {best_rotary['desc']}")
print(f"vs 基准模型差距: {(best_rotary['best_val_top1'] - baseline_model['best_val_top1'])*100:.2f}%")

if best_rotary['best_val_top1'] < baseline_model['best_val_top1']:
    gap_percent = abs((best_rotary['best_val_top1'] - baseline_model['best_val_top1']) / baseline_model['best_val_top1'] * 100)
    print(f"\n❌ 时间旋转模型落后 {gap_percent:.1f}%")
    print("\n可能的原因:")
    print("  1. 模型架构设计问题")
    print("  2. 超参数未调优")
    print("  3. 训练不充分")
    print("  4. 数据不适合时间旋转方法")
    print("  5. 实现有bug")
else:
    print(f"\n✅ 时间旋转模型超越基准 {(best_rotary['best_val_top1'] - baseline_model['best_val_top1'])*100:.2f}%")

print("\n" + "="*80)
print("建议")
print("="*80)

print("\n基于当前实验结果，建议:")

if best_rotary['best_val_top1'] < baseline_model['best_val_top1'] * 0.9:
    print("\n⚠️  严重问题 - 性能差距超过10%")
    print("\n优先行动:")
    print("  1. 检查模型实现是否有bug")
    print("  2. 对比基准模型和时间旋转模型的输入输出维度")
    print("  3. 简化时间旋转模型（移除某些复杂组件）")
    print("  4. 增加训练epoch到50+")
    
elif best_rotary['best_val_top1'] < baseline_model['best_val_top1']:
    print("\n⚠️  需要改进 - 性能略低于基准")
    print("\n建议尝试:")
    print("  1. 调整学习率 (0.001 → 0.0005 或 0.002)")
    print("  2. 调整time_embed_dim (128 → 64 或 256)")
    print("  3. 增加训练epoch")
    print("  4. 调整theta参数 (10000 → 5000 或 20000)")
    
else:
    print("\n✅ 表现良好 - 继续优化")
    print("\n可以尝试:")
    print("  1. 更多epoch")
    print("  2. 更大的模型")
    print("  3. 数据增强")
