import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from sklearn.metrics import r2_score
# 假设这些模型导入是正确的
from tp_model import raw_model
from tp_model import t_model
from tp_model import f_model
from tp_model import tf_model
from tp_model import SATP_model


# =================================================================
# 1. 增加目标函数计算的鲁棒性，防止出现 NaN/Inf 导致优化崩溃
# =================================================================

# R2: Coefficient of Determination (Note: This function calculates r^2)
def calculate_r2(y_true, y_pred):
    # 检查真值是否恒定（方差为零）
    if np.std(y_true) < 1e-6:
        return 0.0  # 如果真值恒定，r2返回0
        
    numerator = np.sum((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denominator = np.sqrt(np.sum((y_true - y_true.mean())**2) * np.sum((y_pred - y_pred.mean())**2))
    
    # 添加极小的epsilon防止分母为零
    if denominator < 1e-10:
        return 0.0
        
    r = numerator / denominator
    return r**2

# NSE: Nash-Sutcliffe Efficiency
def calculate_nse(y_true, y_pred):
    # TSS: Total Sum of Squares (分母)
    TSS = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # 添加极小的epsilon防止分母为零
    if TSS < 1e-10:
        # 返回一个强惩罚值，确保优化器远离这个参数组合
        return -100.0 
        
    # RSS: Residual Sum of Squares (分子)
    RSS = np.sum((y_true - y_pred) ** 2)
    
    return 1 - (RSS / TSS)

# Define optimization problem using pymoo
class TPOptimizationProblem(Problem):
    def __init__(self, discharge, avg_temp, pcp_grow, GEI, tp):
        
        # 2. 扩大参数搜索空间，以克服系统性低估问题
        super().__init__(n_var=8, n_obj=2, n_constr=0, xl=[-50]*8, xu=[50]*8)
        self.discharge = discharge
        self.avg_temp = avg_temp
        self.pcp_grow = pcp_grow
        self.GEI = GEI
        self.tp = tp

    def _evaluate(self, x, out, *args, **kwargs):
        a1, a2, a3, b1, b2, b3, b4, b5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6],x[:, 7]
        
        y_pred = SATP_model(self.discharge, self.avg_temp, self.pcp_grow, self.GEI, a1[:, None], a2[:, None], a3[:, None], b1[:, None], b2[:, None], b3[:, None], b4[:, None], b5[:, None])
        
        y_pred [y_pred < 0] = 0
        
        r2 = np.array([calculate_r2(self.tp, y) for y in y_pred])
        nse = np.array([calculate_nse(self.tp, y) for y in y_pred])
        
        # 确保目标函数返回值中不含 NaN 或 Inf
        r2[np.isnan(r2) | np.isinf(r2)] = 0.0
        nse[np.isnan(nse) | np.isinf(nse)] = -100.0 
        
        out["F"] = -np.column_stack([r2, nse])  # Negative because we want to maximize


def plot_pareto_front(nse_values, r2_values, title="Pareto Front", xlabel="NSE", ylabel="R2"):
    """
    Function to plot the Pareto front for optimization results.
    """
    valid_indices = np.argsort(nse_values)
    nse_sorted = nse_values[valid_indices]
    r2_sorted = r2_values[valid_indices]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(nse_sorted, r2_sorted, color="blue", label="Pareto Solutions", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


file_path = './input/input_data_TP_2022_2024.csv'
raw_df = pd.read_csv(file_path)

# --- 特征工程部分 (保持原样) ---
raw_df['temperature'] = raw_df['temperature'] - 273.15
raw_df['8_day_avg_temp'] = raw_df['temperature'].rolling(window=1, min_periods=1).mean()
avg_temp_min = raw_df['8_day_avg_temp'].min()
avg_temp_max = raw_df['8_day_avg_temp'].max()
raw_df['8_day_avg_temp'] = (raw_df['8_day_avg_temp'] - avg_temp_min) / (avg_temp_max - avg_temp_min)

raw_df['2_day_rainfall_growth'] = raw_df['precipitation'].diff(periods=5).apply(lambda x: x if x > 0 else 0)
avg_p_min = raw_df['2_day_rainfall_growth'].min()
avg_p_max = raw_df['2_day_rainfall_growth'].max()
raw_df['2_day_rainfall_growth'] = (raw_df['2_day_rainfall_growth'] - avg_p_min) / (avg_p_max - avg_p_min)

raw_df['2_day_q_growth'] = raw_df['discharge'].diff(periods=4).apply(lambda x: x if x > 0 else 0)
avg_q_min = raw_df['2_day_q_growth'].min()
avg_q_max = raw_df['2_day_q_growth'].max()
raw_df['2_day_q_growth'] = (raw_df['2_day_q_growth'] - avg_q_min) / (avg_q_max - avg_q_min)

raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df['year'] = raw_df['date'].dt.year
raw_df['precipitation'] = raw_df['precipitation'].fillna(method='bfill')
raw_df['cumulative_p'] = raw_df.groupby('year')['precipitation'].cumsum()
raw_df['total_annual_p'] = raw_df.groupby('year')['precipitation'].transform('sum')
raw_df['GEI'] = raw_df['cumulative_p'] / raw_df['total_annual_p']
# --- 特征工程部分 结束 ---


# 筛选数据：首先去除 NaN，然后去除 TP <= 0.005 的行
filtered_df = raw_df.dropna(subset=['discharge', 'TP']).copy()
filtered_df = filtered_df[filtered_df['TP'] > 0.005].copy()
print(f"Total data points used for calibration (TP > 0.005): {len(filtered_df)}")


# Prepare data for optimization (these arrays now only contain TP > 0.005 data)
discharge = filtered_df['discharge'].values
tp = filtered_df['TP'].values
avg_temp = filtered_df['8_day_avg_temp'].values
pcp_grow = filtered_df['2_day_q_growth'].values
GEI = filtered_df['GEI'].values

# Define reference directions for NSGA-III
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# Create an instance of the optimization problem
problem = TPOptimizationProblem(discharge, avg_temp, pcp_grow, GEI, tp)

# 配置 NSGA-III 算法 (增加种群规模)
algorithm = NSGA3(pop_size=400, ref_dirs=ref_dirs, eliminate_duplicates=True)

# 3. Run the optimization (增加迭代次数，关闭 verbose 以避免 Assertion Error)
res = minimize(problem, algorithm, ('n_gen', 1000), verbose=False)

# Extract optimized parameter values (使用找到的第一个最优解)
a1_opt, a2_opt, a3_opt, b1_opt, b2_opt, b3_opt, b4_opt, b5_opt = res.X[0]
print("\n--- Optimization Results ---")
print(f"Optimized parameters: a1={a1_opt:.4f}, a2={a2_opt:.4f}, a3={a3_opt:.4f}, b1={b1_opt:.4f}, b2={b2_opt:.4f}, b3={b3_opt:.4f}, b4={b4_opt:.4f}, b5={b5_opt:.4f}")

# Calculate R2 and NSE for the optimized model
y_pred_opt = SATP_model(discharge, avg_temp, pcp_grow, GEI, a1_opt, a2_opt, a3_opt, b1_opt, b2_opt, b3_opt, b4_opt, b5_opt)
y_pred_opt[y_pred_opt < 0.005] = 0.005 

r2_opt = calculate_r2(tp, y_pred_opt)
nse_opt = calculate_nse(tp, y_pred_opt)
print(f"R2 (using data where TP > 0.005): {r2_opt:.4f}")
print(f"NSE (using data where TP > 0.005): {nse_opt:.4f}")

# 提取所有Pareto最优解的 NSE 和 R2 值
r2_values = -res.F[:, 0]
nse_values = -res.F[:, 1]

# 调用绘图函数绘制Pareto前沿
plot_pareto_front(nse_values, r2_values, title="Pareto Front (NSE vs R2) - Filtered Data", xlabel="NSE", ylabel="R2")

# Plot the original and predicted TP over time (使用筛选后的数据)
plt.figure(figsize=(12, 6))
plt.scatter(filtered_df['date'], tp, label='Original TP (TP > 0.005)', color='blue', marker='o', s=10)
plt.plot(filtered_df['date'], y_pred_opt, label='Predicted TP', color='red', linestyle='-', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Total Phosphorus Concentration (TP)')
plt.title('Comparison of Original and Predicted TP over Time (Filtered Data)')
plt.legend()

# 优化 x 轴刻度显示
x_ticks = filtered_df['date'][::60]
plt.gca().set_xticks(x_ticks)
plt.gca().set_xticklabels(x_ticks.dt.strftime('%Y-%b'), rotation=45, ha='right')
plt.tight_layout()
plt.show()