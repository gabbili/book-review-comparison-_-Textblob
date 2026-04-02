import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

human = pd.read_csv('/Users/jiechenli/Desktop/Word_sentences.csv')
machine = pd.read_csv('/Users/jiechenli/Desktop/G_sentences.csv')

human['group'] = 'Human'
machine['group'] = 'Machine'

# 只保留需要的列
human_sub = human[['polarity', 'subjectivity', 'group']]
machine_sub = machine[['polarity', 'subjectivity', 'group']]

df = pd.concat([human_sub, machine_sub], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 极性箱形图
sns.boxplot(x='group', y='polarity', data=df, ax=axes[0])
axes[0].set_title('Polarity Distribution')
axes[0].set_ylabel('Polarity')
axes[0].set_xlabel('')

# 主观性箱形图
sns.boxplot(x='group', y='subjectivity', data=df, ax=axes[1])
axes[1].set_title('Subjectivity Distribution')
axes[1].set_ylabel('Subjectivity')
axes[1].set_xlabel('')

plt.tight_layout()
plt.show()

plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')