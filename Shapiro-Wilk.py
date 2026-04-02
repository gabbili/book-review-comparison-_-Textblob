import pandas as pd
import numpy as np
from scipy import stats


HUMAN_FILE = '/Users/jiechenli/Desktop/Word_sentences.csv'
MACHINE_FILE = '/Users/jiechenli/Desktop/G_sentences.csv'
VARIABLE = 'subjectivity'
ALPHA = 0.05


def describe_and_test(group1, group2, var_name):
    """
    对两组数据进行描述统计、正态性检验和差异检验
    """
    print(f"\n===== 变量：{var_name} =====")

    for name, data in zip(['Human', 'Machine'], [group1, group2]):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        median = np.median(data)
        print(f"{name}: n={n}, mean={mean:.5f}, std={std:.5f}, median={median:.5f}")

    _, p1 = stats.shapiro(group1)
    _, p2 = stats.shapiro(group2)
    print(f"\nShapiro-Wilk: p value: Human={p1:.7f}, Machine={p2:.7f}")

    normal = (p1 > ALPHA) and (p2 > ALPHA)

    if normal:
        _, p_levene = stats.levene(group1, group2)
        print(f"Levene  p value: {p_levene:.4f}")

        if p_levene > ALPHA:
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)
            print(f"独立样本 t 检验 (等方差): t={t_stat:.4f}, p={p_val:.4f}")
            # Cohen's d
            pooled_sd = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + (len(group2)-1)*np.var(group2, ddof=1)) /
                                (len(group1)+len(group2)-2))
            d = (np.mean(group1) - np.mean(group2)) / pooled_sd
            print(f"Cohen's d = {d:.4f}")
        else:
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            print(f"Welch 检验 (不等方差): t={t_stat:.4f}, p={p_val:.4f}")
            # 近似 Cohen's d
            d = (np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1))/2)
            print(f"Cohen's d (近似) = {d:.4f}")
    else:
        # Mann-Whitney U 检验
        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        print(f"Mann-Whitney U 检验: U={u_stat:.1f}, p={p_val:.4f}")
        # 效应量 r = Z / sqrt(N)
        n1, n2 = len(group1), len(group2)
        # 计算近似的 Z 值
        z = (u_stat - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)
        r = z / np.sqrt(n1 + n2)
        print(f"效应量 r = {r:.4f}")

    # 最终结论
    if p_val < ALPHA:
        print(f"结论：在 α={ALPHA} 水平上，两组 {var_name} 存在显著差异。")
    else:
        print(f"结论：在 α={ALPHA} 水平上，两组 {var_name} 不存在显著差异。")

def main():
    try:
        human_df = pd.read_csv(HUMAN_FILE)
        machine_df = pd.read_csv(MACHINE_FILE)
    except FileNotFoundError as e:
        print(f"文件读取错误: {e}")
        return

    if VARIABLE not in human_df.columns or VARIABLE not in machine_df.columns:
        print(f"错误：列 '{VARIABLE}' 不存在于文件中。")
        print("请检查文件列名，或修改变量名。")
        return

    human_data = human_df[VARIABLE].dropna()
    machine_data = machine_df[VARIABLE].dropna()

    print(f"人类句子数: {len(human_data)}, 机器句子数: {len(machine_data)}")

    describe_and_test(human_data, machine_data, VARIABLE)

if __name__ == '__main__':
    main()
