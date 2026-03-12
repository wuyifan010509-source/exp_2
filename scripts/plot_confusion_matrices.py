"""
生成三个混淆矩阵的可视化图表
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 混淆矩阵数据
semantic_routing = np.array([
    [16, 9, 3],
    [14, 24, 0],
    [8, 4, 25]
])

slm_distillation = np.array([
    [14, 16, 3],
    [4, 28, 1],
    [0, 1, 32]
])

llm_32b = np.array([
    [14, 16, 3],
    [1, 32, 0],
    [0, 15, 18]
])

labels = ['Low', 'Mid', 'High']

# 创建图形 - 使用GridSpec来更好地控制布局
fig = plt.figure(figsize=(16, 4.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.3)

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# 颜色映射
cmap = plt.cm.Blues

# 1. 语义路由
im1 = axes[0].imshow(semantic_routing, interpolation='nearest', cmap=cmap, vmin=0, vmax=32)
axes[0].set_title('Semantic Routing', fontsize=14, fontweight='bold', pad=20)
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True', fontsize=12)
axes[0].set_xticks(np.arange(len(labels)))
axes[0].set_yticks(np.arange(len(labels)))
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# 添加数值标注
for i in range(len(labels)):
    for j in range(len(labels)):
        text = axes[0].text(j, i, semantic_routing[i, j],
                           ha="center", va="center", color="black" if semantic_routing[i, j] < 20 else "white",
                           fontsize=14, fontweight='bold')

# 2. SLM 蒸馏
im2 = axes[1].imshow(slm_distillation, interpolation='nearest', cmap=cmap, vmin=0, vmax=32)
axes[1].set_title('SLM (Ours)', fontsize=14, fontweight='bold', pad=20)
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('True', fontsize=12)
axes[1].set_xticks(np.arange(len(labels)))
axes[1].set_yticks(np.arange(len(labels)))
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        color = "white" if slm_distillation[i, j] > 15 else "black"
        text = axes[1].text(j, i, slm_distillation[i, j],
                           ha="center", va="center", color=color,
                           fontsize=14, fontweight='bold')

# 3. 32B 大模型
im3 = axes[2].imshow(llm_32b, interpolation='nearest', cmap=cmap, vmin=0, vmax=32)
axes[2].set_title('32B LLM Baseline', fontsize=14, fontweight='bold', pad=20)
axes[2].set_xlabel('Predicted', fontsize=12)
axes[2].set_ylabel('True', fontsize=12)
axes[2].set_xticks(np.arange(len(labels)))
axes[2].set_yticks(np.arange(len(labels)))
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        color = "white" if llm_32b[i, j] > 15 else "black"
        text = axes[2].text(j, i, llm_32b[i, j],
                           ha="center", va="center", color=color,
                           fontsize=14, fontweight='bold')

# 添加颜色条 - 单独放在右侧
cbar_ax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(im3, cax=cbar_ax)
cbar.set_label('Count', rotation=270, labelpad=15, fontsize=12)

plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrices_comparison.pdf', bbox_inches='tight')
print("混淆矩阵对比图已保存: confusion_matrices_comparison.png")
print("混淆矩阵对比图已保存: confusion_matrices_comparison.pdf")
