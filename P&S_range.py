import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
human_df = pd.read_csv('/Users/jiechenli/Desktop/Word_sentences.csv')
machine_df = pd.read_csv('/Users/jiechenli/Desktop/G_sentences.csv')

# 计算总句子数（所有句子，包括极性列和主观性列，数据框行数即句子总数）
human_total = len(human_df)
machine_total = len(machine_df)

print(f"Human total sentences: {human_total}")
print(f"Machine total sentences: {machine_total}")

# 定义极性刻度（-1.0 到 1.0，步长 0.1），转为字符串列表
polarity_ticks = np.arange(-1.0, 1.05, 0.1)
polarity_labels = [f'{tick:.1f}' for tick in polarity_ticks]

# 定义主观性刻度（0.0 到 1.0，步长 0.1），转为字符串
subjectivity_ticks = np.arange(0.0, 1.05, 0.1)
subjectivity_labels = [f'{tick:.1f}' for tick in subjectivity_ticks]

def count_by_round(df, column, labels):
    """将数据四舍五入到一位小数，转为字符串，统计每个字符串的出现次数。返回 Series，索引为 labels。"""
    rounded = df[column].round(1).astype(str)
    counts = rounded.value_counts()
    full_counts = pd.Series(0, index=labels)
    for label in labels:
        if label in counts:
            full_counts[label] = counts[label]
    return full_counts

# 计算极性频数
human_pol_counts = count_by_round(human_df, 'polarity', polarity_labels)
machine_pol_counts = count_by_round(machine_df, 'polarity', polarity_labels)

# 计算主观性频数
human_sub_counts = count_by_round(human_df, 'subjectivity', subjectivity_labels)
machine_sub_counts = count_by_round(machine_df, 'subjectivity', subjectivity_labels)

# 绘图样式
plt.style.use('seaborn-v0_8-whitegrid')

# 辅助函数：添加总句子数文本
def add_total_text(ax, human_total, machine_total):
    ax.text(0.98, 0.95, f'Human: {human_total} sentences\nMachine: {machine_total} sentences',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 图1：极性分布
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = np.arange(len(polarity_labels))
width = 0.35
bars1 = ax1.bar(x - width/2, human_pol_counts, width, label='Human', edgecolor='black', alpha=0.8)
bars2 = ax1.bar(x + width/2, machine_pol_counts, width, label='Machine', edgecolor='black', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(polarity_labels, rotation=45)
ax1.set_xlabel('Polarity Value')
ax1.set_ylabel('Number of Sentences')
ax1.set_title('Polarity Distribution (Rounded to 0.1)')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 添加总句子数文本
add_total_text(ax1, human_total, machine_total)

# 可选：添加数值标签（避免拥挤，仅当柱子高度大于0时）
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

# 图2：主观性分布
fig2, ax2 = plt.subplots(figsize=(14, 6))
x_sub = np.arange(len(subjectivity_labels))
bars1 = ax2.bar(x_sub - width/2, human_sub_counts, width, label='Human', edgecolor='black', alpha=0.8)
bars2 = ax2.bar(x_sub + width/2, machine_sub_counts, width, label='Machine', edgecolor='black', alpha=0.8)
ax2.set_xticks(x_sub)
ax2.set_xticklabels(subjectivity_labels, rotation=45)
ax2.set_xlabel('Subjectivity Value')
ax2.set_ylabel('Number of Sentences')
ax2.set_title('Subjectivity Distribution (Rounded to 0.1)')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# 添加总句子数文本
add_total_text(ax2, human_total, machine_total)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()