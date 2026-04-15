"""
锂电池SOH预测复现代码 v2
基于实际数据格式重写（.txt文件版本）

数据目录结构：
  ./Capacity data/
      Data_Capacity_25C03.txt
      Data_Capacity_25C05.txt
      ...
  ./EIS data/
      EIS_state_V_25C03.txt
      EIS_state_V_25C05.txt
      ...

运行前安装依赖：
  pip install numpy pandas scikit-learn xgboost shap optuna matplotlib
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置（根据你的实际路径修改）
# ============================================================
CAPACITY_DIR = './Capacity data'
EIS_DIR      = './EIS data'

# 论文使用的6块电池
BATTERY_NAMES = ['25C03', '25C05', '35C01', '35C02', '45C01', '45C02']

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# ============================================================
# 1. 读取Capacity文件，提取每个cycle的最大容量
# ============================================================
def load_capacity(battery_name):
    path = os.path.join(CAPACITY_DIR, f'Data_Capacity_{battery_name}.txt')
    
    # 先读原始数据，不指定列名
    raw = pd.read_csv(path, sep='\t', header=None)
    n_cols = raw.shape[1]
    
    # 根据列数分配列名
    if n_cols >= 7:
        cols = ['time', 'cycle', 'ox_red', 'Ewe', 'I', 'capacity', 'extra']
    elif n_cols == 5:
        cols = ['time', 'cycle', 'ox_red', 'capacity', 'extra']
    else:  # 4列
        cols = ['time', 'cycle', 'ox_red', 'capacity']
    
    raw.columns = cols[:n_cols]
    
    # 第0行是表头文字，跳过
    df = raw.iloc[1:].copy()
    df['cycle']    = pd.to_numeric(df['cycle'],    errors='coerce')
    df['ox_red']   = pd.to_numeric(df['ox_red'],   errors='coerce')
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    df = df.dropna(subset=['cycle', 'capacity'])
    df['cycle'] = df['cycle'].astype(int)
    
    # 每个cycle取容量最大值（累积放电量的终值）
    cap_per_cycle = df.groupby('cycle')['capacity'].max()
    cap_per_cycle = cap_per_cycle[cap_per_cycle > 0]
    
    return cap_per_cycle


# ============================================================
# 2. 读取EIS文件（State V），提取每个cycle的阻抗数据
# ============================================================
def load_eis_stateV(battery_name):
    path = os.path.join(EIS_DIR, f'EIS_state_V_{battery_name}.txt')
    
    # 첫 줄로 헤더 유무 판단
    first = pd.read_csv(path, sep='\t', header=None, nrows=1)
    first_val = str(first.iloc[0, 0]).strip()
    
    COL_NAMES = ['time/s', 'cycle number', 'freq/Hz',
                 'Re(Z)/Ohm', '-Im(Z)/Ohm', '|Z|/Ohm', 'Phase(Z)/deg']
    
    if _is_number(first_val):
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = COL_NAMES[:df.shape[1]]
    else:
        df = pd.read_csv(path, sep='\t', header=0)
        df.columns = [c.strip() for c in df.columns]
        df.columns = COL_NAMES[:df.shape[1]]
    
    df['cycle number'] = pd.to_numeric(df['cycle number'],
                                        errors='coerce').astype('Int64')
    df['freq/Hz']      = pd.to_numeric(df['freq/Hz'],      errors='coerce')
    df['Re(Z)/Ohm']    = pd.to_numeric(df['Re(Z)/Ohm'],    errors='coerce')
    df['-Im(Z)/Ohm']   = pd.to_numeric(df['-Im(Z)/Ohm'],   errors='coerce')
    df = df.dropna(subset=['cycle number', 'freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm'])
    
    eis_dict = {}
    for cycle_num, group in df.groupby('cycle number'):
        group = group.sort_values('freq/Hz', ascending=False)
        eis_dict[int(cycle_num)] = {
            'freq': group['freq/Hz'].values.astype(float),
            'ReZ':  group['Re(Z)/Ohm'].values.astype(float),
            'ImZ':  -group['-Im(Z)/Ohm'].values.astype(float),
        }
    return eis_dict


# ============================================================
# 3. 对齐EIS和Capacity（按cycle number直接匹配）
# ============================================================
def align_eis_capacity(eis_dict, cap_series, nominal_capacity):
    records = []
    cap_cycles = np.array(cap_series.index.values, dtype=int)  
    cap_values = cap_series.values.astype(float)                

    for cycle_num, eis in eis_dict.items():
        idx = np.argmin(np.abs(cap_cycles - cycle_num))
        nearest_cap_cycle = cap_cycles[idx]

        if abs(nearest_cap_cycle - cycle_num) <= 2:
            capacity = cap_values[idx]                       
            soh = capacity / nominal_capacity * 100
            records.append({
                'cycle': cycle_num,
                'SOH': soh,
                'capacity': capacity,
                **{k: v for k, v in eis.items()}
            })

    records.sort(key=lambda x: x['cycle'])
    return records


# ============================================================
# 4. EIS特征提取（论文Section 2.2.2，20个统计特征）
# ============================================================
def split_semicircle_tail(re_z, im_z):
    """
    用原始-Im(Z)的最大值点作为切分点
    即Im(Z)最负的点，对应半圆底部
    """
    # im_z是还原后的Im(Z)，半圆底部是最负的点，即最小值
    # 但只在前半段找，避免低频区干扰
    search_range = max(3, len(im_z) // 2)
    split = np.argmin(im_z[:search_range])
    split = max(3, min(split, len(re_z) - 3))
    
    return re_z[:split], im_z[:split], re_z[split:], im_z[split:]


def compute_region_features(re_arr, im_arr, prefix):
    """
    对一个区域提取10个统计特征
    """
    feats = {}
    for arr, tag in [(re_arr, 'Re_Z'), (im_arr, 'Im_Z')]:
        if len(arr) == 0:
            arr = np.array([0.0])
        feats[f'{prefix}_{tag}_Mean']     = np.mean(arr)
        feats[f'{prefix}_{tag}_Max']      = np.max(arr)
        feats[f'{prefix}_{tag}_Min']      = np.min(arr)
        feats[f'{prefix}_{tag}_Variance'] = np.var(arr)

    min_len = min(len(re_arr), len(im_arr))
    if min_len > 1:
        cov_mat = np.cov(re_arr[:min_len], im_arr[:min_len])
        feats[f'{prefix}_Re_Im_Covariance'] = cov_mat[0, 1]
        feats[f'{prefix}_Re_Im_Ratio'] = (
            np.mean(re_arr[:min_len]) /
            (np.mean(np.abs(im_arr[:min_len])) + 1e-10)
        )
    else:
        feats[f'{prefix}_Re_Im_Covariance'] = 0.0
        feats[f'{prefix}_Re_Im_Ratio']      = 0.0

    return feats


def extract_eis_features(re_z, im_z):
    """
    提取全部20个EIS统计特征
    """
    semi_re, semi_im, tail_re, tail_im = split_semicircle_tail(re_z, im_z)
    feats = {}
    feats.update(compute_region_features(semi_re, semi_im, 'Semicircle'))
    feats.update(compute_region_features(tail_re, tail_im, 'Tail'))
    return feats


# ============================================================
# 5. 滑动窗口时序特征（window=8）
# ============================================================
def add_sliding_window(df, window=8, col='SOH'):
    df = df.copy()
    df['SlidingWindow'] = df[col].rolling(window=window, min_periods=1).mean()
    return df


# ============================================================
# 6. 构建完整数据集
# ============================================================
def build_dataset():
    all_dfs = []

    for name in BATTERY_NAMES:
        print(f'  处理 {name}...')
        try:
            cap_series = load_capacity(name)
            eis_dict   = load_eis_stateV(name)

            # 额定容量用第1个cycle的容量
            nominal = cap_series.iloc[0]

            records = align_eis_capacity(eis_dict, cap_series, nominal)
            if len(records) == 0:
                print(f'    {name}: 对齐后无数据，跳过')
                continue

            rows = []
            for rec in records:
                feats = extract_eis_features(rec['ReZ'], rec['ImZ'])
                feats['SOH']     = rec['SOH']
                feats['cycle']   = rec['cycle']
                feats['battery'] = name
                rows.append(feats)

            df = pd.DataFrame(rows)
            df = add_sliding_window(df)
            all_dfs.append(df)
            print(f'    {name}: {len(df)} 个样本，SOH范围 '
                  f'{df["SOH"].min():.1f}%~{df["SOH"].max():.1f}%')

        except FileNotFoundError as e:
            print(f'    {name}: 文件不存在 → {e}')
        except Exception as e:
            print(f'    {name}: 错误 → {e}')

    if not all_dfs:
        raise RuntimeError('所有电池数据加载失败，请检查路径')

    return pd.concat(all_dfs, ignore_index=True)


# ============================================================
# 7. LOOCV评估
# ============================================================
def evaluate_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}


def loocv(df, model_fn, feature_cols, use_sliding=True):
    """
    按电池留一交叉验证
    model_fn: 每次调用返回一个新模型实例
    """
    batteries = df['battery'].unique()
    all_true, all_pred = [], []

    cols = feature_cols + (['SlidingWindow'] if use_sliding else [])

    for test_bat in batteries:
        train = df[df['battery'] != test_bat]
        test  = df[df['battery'] == test_bat]

        X_train = train[cols].fillna(0).values
        y_train = train['SOH'].values
        X_test  = test[cols].fillna(0).values
        y_test  = test['SOH'].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train, y_train)

        all_pred.extend(model.predict(X_test))
        all_true.extend(y_test)

    return evaluate_metrics(np.array(all_true), np.array(all_pred))


# ============================================================
# 8. Optuna 超参数优化 + 模型定义
# ============================================================

def optimize_xgboost(X_train, y_train, n_trials=50):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 10),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'verbosity': 0, 'random_state': 42
        }
        model = XGBRegressor(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train,
                                 cv=3, scoring='neg_mean_squared_error')
        return -scores.mean()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    return study.best_params


MODELS = {
    'XGBoost': lambda: XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0),
    'Lasso':   lambda: Lasso(alpha=0.001, max_iter=10000),
    'RF':      lambda: RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42),
    'SVR':     lambda: SVR(kernel='rbf', C=100, epsilon=0.05, gamma='scale'),
    'MLP':     lambda: MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        max_iter=1000, random_state=42, early_stopping=True),
}


# ============================================================
# 9. 打印结果表格
# ============================================================
def print_table(results, title):
    print(f'\n{"="*62}')
    print(f'  {title}')
    print(f'{"="*62}')
    print(f'{"Model":<12} {"MAE":>8} {"MSE":>8} {"RMSE":>8} {"MAPE":>10}')
    print(f'{"-"*62}')
    for name, m in results.items():
        print(f'{name:<12} {m["MAE"]:>8.3f} {m["MSE"]:>8.3f} '
              f'{m["RMSE"]:>8.3f} {m["MAPE"]:>9.2f}%')
    print(f'{"="*62}')


# ============================================================
# 10. 可视化：SOH预测曲线
# ============================================================
def plot_predictions(df, model, scaler, feature_cols,
                     use_sliding=True, save_path=None):
    cols = feature_cols + (['SlidingWindow'] if use_sliding else [])
    batteries = df['battery'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, bat in enumerate(batteries):
        bat_df = df[df['battery'] == bat].sort_values('cycle')
        X = scaler.transform(bat_df[cols].fillna(0).values)
        y_pred = model.predict(X)
        y_true = bat_df['SOH'].values
        cycles = bat_df['cycle'].values

        axes[i].plot(cycles, y_true, 'k-',  label='True',      lw=1.5)
        axes[i].plot(cycles, y_pred, 'r--', label='Predicted', lw=1.5)
        axes[i].set_title(bat)
        axes[i].set_xlabel('Cycle')
        axes[i].set_ylabel('SOH (%)')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('SOH Prediction Results (XGBoost + EIS + Sliding Window)',
                 fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  图片已保存: {save_path}')
    plt.show()


# ============================================================
# 11. SHAP可解释性分析
# ============================================================
def run_shap(model, X_scaled, feature_names, temp_label, save_path=None):
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_scaled)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_scaled,
                      feature_names=feature_names,
                      show=False)
    plt.title(f'SHAP Summary — XGBoost at {temp_label}°C')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  SHAP图已保存: {save_path}')
    plt.show()


# ============================================================
# 12. 主流程
# ============================================================
def main():
    print('=' * 62)
    print('  锂电池SOH预测复现 — ETMS框架')
    print('=' * 62)

    # Step 1: 构建数据集
    print('\n[1/4] 加载数据...')
    df = build_dataset()
    print(f'  总样本数: {len(df)}，特征数: {len(df.columns)-3}')

    exclude = {'SOH', 'cycle', 'battery', 'SlidingWindow'}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Step 2: 仅EIS特征
    print('\n[2/4] LOOCV评估（仅EIS特征）...')
    results_eis = {}
    for name, fn in MODELS.items():
        print(f'  {name}...', end='', flush=True)
        results_eis[name] = loocv(df, fn, feature_cols, use_sliding=False)
        print(f' MAE={results_eis[name]["MAE"]:.3f}')
    print_table(results_eis, 'Table 1: 仅EIS特征  (论文目标: MLP MAE≈0.17)')

    # Step 3: EIS + 滑动窗口（XGBoost用Optuna调参）
    print('\n[3/4] LOOCV评估（EIS + 滑动窗口）...')
    results_comb = {}
    cols_with_sw = feature_cols + ['SlidingWindow']

    for name, fn in MODELS.items():
        print(f'  {name}...', end='', flush=True)

        if name == 'XGBoost':
            # XGBoost单独做：先用全量训练集调参，再LOOCV
            batteries = df['battery'].unique()
            all_true, all_pred = [], []

            for test_bat in batteries:
                train = df[df['battery'] != test_bat]
                test  = df[df['battery'] == test_bat]

                X_train = train[cols_with_sw].fillna(0).values
                y_train = train['SOH'].values
                X_test  = test[cols_with_sw].fillna(0).values
                y_test  = test['SOH'].values

                scaler  = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                # 用Optuna在当前训练集上找最优参数
                best_params = optimize_xgboost(X_train, y_train, n_trials=50)
                best_params.update({'verbosity': 0, 'random_state': 42})
                model = XGBRegressor(**best_params)
                model.fit(X_train, y_train)

                all_pred.extend(model.predict(X_test))
                all_true.extend(y_test)

            results_comb['XGBoost'] = evaluate_metrics(
                np.array(all_true), np.array(all_pred))
            print(f' MAE={results_comb["XGBoost"]["MAE"]:.3f}')
        else:
            results_comb[name] = loocv(df, fn, feature_cols, use_sliding=True)
            print(f' MAE={results_comb[name]["MAE"]:.3f}')

    print_table(results_comb,
                'Table 2: EIS + 滑动窗口  (论文目标: XGBoost MAE≈0.05)')

    # Step 4: 用全量数据训练XGBoost，做可视化和SHAP
    print('\n[4/4] 可视化和SHAP分析...')
    cols_full = feature_cols + ['SlidingWindow']
    X_all = df[cols_full].fillna(0).values
    y_all = df['SOH'].values

    scaler_all = StandardScaler()
    X_scaled   = scaler_all.fit_transform(X_all)

    xgb = MODELS['XGBoost']()
    xgb.fit(X_scaled, y_all)

    # 预测曲线图
    plot_predictions(df, xgb, scaler_all, feature_cols,
                     save_path='soh_prediction_curves.png')

    # SHAP分温度分析
    for temp in ['25', '35', '45']:
        mask   = df['battery'].str.startswith(temp)
        temp_df = df[mask]
        if len(temp_df) == 0:
            continue
        X_temp = scaler_all.transform(temp_df[cols_full].fillna(0).values)
        run_shap(xgb, X_temp, cols_full,
                 temp_label=temp,
                 save_path=f'shap_{temp}C.png')

    print('\n✅ 复现完成！')
    print('\n论文对照指标:')
    print('  Table 1 MLP  (EIS only):   MAE≈0.17, MAPE≈0.97%')
    print('  Table 2 XGBoost (combined): MAE≈0.05, MAPE≈0.41%')


if __name__ == '__main__':
    main()
