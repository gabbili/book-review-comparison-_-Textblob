import pandas as pd
import numpy as np
from scipy import stats

def variance_ttest(data1, data2, variable_name, alpha_var=0.05):
    d1 = data1.dropna()
    d2 = data2.dropna()
    n1, n2 = len(d1), len(d2)
    mean1, mean2 = d1.mean(), d2.mean()
    std1, std2 = d1.std(), d2.std()
    print(f'\n{variable_name}')
    print(f'human: n={n1}, mean={mean1:.4f}, std={std1:.4f}')
    print(f'machine: {n2}, mean={mean2:.4f}, std={std2:.4f}')

    stat_levene, p_levene = stats.levene(d1, d2, center='median')
    print(f'levene: statistics={stat_levene:.4f}, p_levene={p_levene:.4f}')

    if p_levene > alpha_var:
        res = stats.ttest_ind(d1, d2, equal_var=True)
        test_type = '等方差'
        t_stat = res.statistic
        p_value = res.pvalue
        df = n1 + n2 -2
        pooled_sd = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / pooled_sd
        print(f'方差齐(p>{alpha_var}), 采用students t')
    else:
        res = stats.ttest_ind(d1, d2, equal_var=False)
        test_type = 'welchs test（方差不等）'
        t_stat = res.statistic
        p_value = res.pvalue
        if hasattr(res, 'df'):
            df = res.df
        else:
            s1_sq, s2_sq = std1 ** 2, std2 ** 2
            df = (s1_sq / n1 + s2_sq / n2) ** 2 / ((s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1))
        pooled_sd = np.sqrt((std1**2 + std2**2) / 2)
        cohen_d = (mean1 - mean2) / pooled_sd
        print(f'方差不齐（p<={alpha_var}), 用welchs test')

    print(f'{test_type}: t={t_stat:.4f}, p={p_value:.4f}, df={df:.2f}')
    print(f'cohen_d={cohen_d:.4f}')
    return t_stat, p_value, test_type, cohen_d

if __name__ == '__main__':
    human_df = pd.read_csv('/Users/jiechenli/Desktop/Word_sentences.csv')
    machine_df = pd.read_csv('/Users/jiechenli/Desktop/G_sentences.csv')

    for col in ['polarity','subjectivity']:
        if col not in human_df.columns or col not in machine_df.columns:
            print(f'erro: absence of {col}')
            exit( )
    variance_ttest(human_df['polarity'], machine_df['polarity'], 'polarity')
    variance_ttest(human_df['subjectivity'], machine_df['subjectivity'], 'subjectivity')

