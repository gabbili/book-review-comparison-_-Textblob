import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

HUMAN_FILE = '/Users/jiechenli/Desktop/Word_sentences.csv'
MACHINE_FILE = '/Users/jiechenli/Desktop/G_sentences.csv'
VARIABLE = 'polarity'

human_df = pd.read_csv(HUMAN_FILE)
machine_df = pd.read_csv(MACHINE_FILE)

human_data = human_df[VARIABLE].dropna()
machine_data = machine_df[VARIABLE].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(human_data, dist="norm", plot=axes[0])
axes[0].set_title(f'Human {VARIABLE} Q-Q Plot')
axes[0].set_xlabel('Theoretical Quantiles')
axes[0].set_ylabel('Sample Quantiles')

stats.probplot(machine_data, dist="norm", plot=axes[1])
axes[1].set_title(f'Machine {VARIABLE} Q-Q Plot')
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()