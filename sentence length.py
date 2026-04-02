import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def compute_stats(df, label):
    # ... 与之前相同，省略 ...
    df = df.copy()
    df['word_count'] = df['sentence'].astype(str).apply(lambda x: len(x.split()))
    grouped = df.groupby('word_count').agg(
        count=('polarity', 'size'),
        mean_polarity=('polarity', 'mean'),
        std_polarity=('polarity', 'std'),
        mean_subjectivity=('subjectivity', 'mean'),
        std_subjectivity=('subjectivity', 'std')
    ).reset_index()
    grouped['se_polarity'] = grouped['std_polarity'] / np.sqrt(grouped['count'])
    grouped['se_subjectivity'] = grouped['std_subjectivity'] / np.sqrt(grouped['count'])
    grouped['source'] = label
    return grouped


def plot_combined(human_stats, machine_stats, variable='polarity', use_std=True, alpha_shadow=0.3):
    if variable == 'polarity':
        mean_col = 'mean_polarity'
        std_col = 'std_polarity'
        se_col = 'se_polarity'
        ylabel = 'Polarity'
        title = 'Polarity vs Sentence Length'
        colors = {'Human': 'steelblue', 'Machine': 'coral'}
    else:
        mean_col = 'mean_subjectivity'
        std_col = 'std_subjectivity'
        se_col = 'se_subjectivity'
        ylabel = 'Subjectivity'
        title = 'Subjectivity vs Sentence Length'
        colors = {'Human': 'forestgreen', 'Machine': 'darkorange'}

    fig, ax = plt.subplots(figsize=(10, 6))

    for stats, label in [(human_stats, 'Human'), (machine_stats, 'Machine')]:
        color = colors[label]
        if use_std:
            lower = stats[mean_col] - stats[std_col]
            upper = stats[mean_col] + stats[std_col]
            shadow_label = '±1 SD'
        else:
            lower = stats[mean_col] - stats[se_col]
            upper = stats[mean_col] + stats[se_col]
            shadow_label = '±1 SE'

        ax.fill_between(stats['word_count'], lower, upper,
                        alpha=alpha_shadow, color=color, label=f'{label} {shadow_label}')
        ax.plot(stats['word_count'], stats[mean_col],
                color=color, linewidth=2, label=f'{label} mean')

    # 设置刻度：y轴每0.1显示一个数值标签
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 主刻度间隔0.1
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 显示一位小数
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))  # 可选：更细的次要刻度（不显示标签）

    # x轴：主刻度间隔10，小刻度间隔1（不显示标签）
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, color='gray')
    ax.tick_params(axis='y', which='minor', length=3, color='lightgray')

    # 避免y轴标签重叠，可旋转或调整字体大小
    ax.tick_params(axis='y', labelsize=9, rotation=45)  # 旋转45度，字体稍小

    ax.set_xlabel('Sentence Length (number of words)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


# 读取数据
human_df = pd.read_csv('/Users/jiechenli/Desktop/Word_sentences.csv')
machine_df = pd.read_csv('/Users/jiechenli/Desktop/G_sentences.csv')

human_stats = compute_stats(human_df, 'Human')
machine_stats = compute_stats(machine_df, 'Machine')

print("Human 各句长统计（前10行）")
print(human_stats[['word_count', 'count', 'mean_polarity', 'mean_subjectivity']].head(10))
print("\nMachine 各句长统计（前10行）")
print(machine_stats[['word_count', 'count', 'mean_polarity', 'mean_subjectivity']].head(10))

plot_combined(human_stats, machine_stats, variable='polarity', use_std=True)
plot_combined(human_stats, machine_stats, variable='subjectivity', use_std=True)