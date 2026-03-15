import os, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

# Tools
# R^2
def remove_high_correlation(X, threshold=0.95):
    corr    = X.corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return X.drop(columns=to_drop), to_drop

# VIF
def remove_high_vif(X, threshold=10, max_iter=15):
    scaler  = StandardScaler()
    Xs      = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    removed = []
    with tqdm(total=max_iter, desc='VIF', ncols=80) as pbar:
        for i in range(max_iter):
            if Xs.shape[1] < 3:
                pbar.update(max_iter - i); break
            vifs    = np.array([variance_inflation_factor(Xs.values, j)
                                for j in range(Xs.shape[1])])
            max_vif = vifs.max()
            pbar.set_postfix({'MaxVIF': f'{max_vif:.1f}'})
            if max_vif <= threshold:
                pbar.update(max_iter - i); break
            worst = np.argmax(vifs)
            removed.append(Xs.columns[worst])
            Xs = Xs.drop(columns=[Xs.columns[worst]])
            pbar.update(1)
    return X[Xs.columns.tolist()], removed

# Precision
def best_precision_at_recall(y_true, y_prob, recall_min=0.30):
    """Recall >= recall_min 條件下的最高 Precision"""
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
    mask = rec_arr[:-1] >= recall_min
    if mask.sum() == 0:
        return 0.0, 0.0, None
    idx = np.argmax(prec_arr[:-1][mask])
    return (float(prec_arr[:-1][mask][idx]),
            float(rec_arr[:-1][mask][idx]),
            float(thresh_arr[mask][idx]))

# n_splits gap
class ProportionalSeamlessCV:
    def __init__(self, date_column='date', n_splits=5, gap_days=7, val_ratio=0.1):
        self.date_column = date_column
        self.n_splits    = n_splits
        self.gap_days    = gap_days
        self.val_ratio   = val_ratio

    def split(self, df):
        unique_dates = sorted(df[self.date_column].unique())
        n_dates      = len(unique_dates)
        val_size     = int(n_dates * self.val_ratio)
        remained     = n_dates - val_size - self.gap_days
        train_step   = remained // self.n_splits
        if train_step <= 0:
            raise ValueError("數據量不足以支撐目前設定")
        for i in range(self.n_splits):
            train_end = train_step * (i + 1)
            val_start = train_end + self.gap_days
            val_end   = val_start + val_size
            yield unique_dates[:train_end], unique_dates[val_start:val_end]

# Optuna
def plot_trial_history(study, trial_details, save_path):
    passed = [(t.number, t.value) for t in study.trials if t.value is not None]
    pruned = [t.number for t in study.trials if t.value is None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    if passed:
        nums, vals = zip(*passed)
        axes[0].scatter(nums, vals, s=18, alpha=0.6,
                        color='steelblue', label='Trial Precision', zorder=3)
        axes[0].plot(nums, np.maximum.accumulate(vals),
                     color='crimson', lw=1.8, label='Best so far', zorder=4)
    if pruned:
        axes[0].scatter(pruned, [0]*len(pruned), s=15,
                        color='lightgray', label='Pruned', zorder=2)
    axes[0].set(xlabel='Trial', ylabel='CV Precision (Recall≥0.3)',
                title='Optuna — Precision History')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    passed_aucs = [trial_details[t.number]['cv_auc_mean']
                   for t in study.trials
                   if t.value is not None and t.number in trial_details]
    passed_nums = [t.number for t in study.trials
                   if t.value is not None and t.number in trial_details]
    if passed_aucs:
        axes[1].scatter(passed_nums, passed_aucs,
                        s=18, alpha=0.6, color='darkorange')
        axes[1].axhline(0.50, color='red', lw=1, ls='--', label='Prune AUC=0.5')
        axes[1].set(xlabel='Trial', ylabel='CV AUC mean',
                    title='Passed Trials — CV AUC')
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[儲存] {save_path}")

# ROC PR
def plot_roc_precision(y_true, y_prob, best_thr, label, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _= precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color='steelblue', lw=2,
                 label=f'AUC={roc_auc_score(y_true, y_prob):.4f}')
    axes[0].plot([0,1],[0,1],'k--',lw=1)
    axes[0].set(xlabel='FPR', ylabel='TPR', title=f'ROC — {label}')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color='darkorange', lw=2,
                 label=f'AP={average_precision_score(y_true, y_prob):.4f}')
    axes[1].axvline(0.30, color='gray', lw=1, ls='--', label='Recall≥0.3')
    if best_thr is not None:
        bp, br, _ = best_precision_at_recall(y_true, y_prob, 0.30)
        axes[1].scatter([br], [bp], color='red', s=80, zorder=5,
                        label=f'Best P={bp:.3f} R={br:.3f}')
    axes[1].set(xlabel='Recall', ylabel='Precision',
                title=f'Precision-Recall — {label}')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(label, fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[儲存] {save_path}")


# importing csv data
CSV_PATH = r'C:\Users\xqwer\Downloads\projet\final_xgboost_ready.csv'

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
df['date']  = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values('date').reset_index(drop=True)

feature_cols = [
    "open", "high", "low", "close", "volume",
    "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
    "rsi_14", "rsi_7", "rsi_change", "stddev",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "stoch_k", "stoch_d", "roc", "mom", "willr",
    "price_to_sma20", "price_change", "price_change_pct",
    "high_low_pct", "body_pct", "upper_shadow", "lower_shadow",
    "obv", "volume_sma20", "volume_ratio",
    "sentiment_mean", "sentiment_std",
]
existing_cols = [c for c in feature_cols if c in df.columns]
print(f"原始特徵數：{len(existing_cols)}")


# sieve x

X_all = df[existing_cols].copy()

X_all, removed_corr = remove_high_correlation(X_all, threshold=0.95)

X_all, removed_vif = remove_high_vif(X_all, threshold=10)

selected_cols = X_all.columns.tolist()


# cut in two
unique_dates = sorted(df['date'].unique())
cut          = int(len(unique_dates) * 0.80)

df_train = df[df['date'].isin(unique_dates[:cut])]
df_test  = df[df['date'].isin(unique_dates[cut:])]

X_train = df_train[selected_cols].copy()
y_train = df_train['y']
X_test  = df_test[selected_cols].copy()
y_test  = df_test['y']

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

spw = (y_train == 0).sum() / ((y_train == 1).sum() + 1e-9)

df_cv = df_train[['date'] + selected_cols + ['y']].copy()
cv    = ProportionalSeamlessCV(date_column='date', n_splits=5, gap_days=30)


# seed
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

# optuna gate
PRUNE_AUC_MIN    = 0.50
PRUNE_RECALL_MIN = 0.30

trial_details = {}

# model 
def build_model(trial):
    model_type = trial.suggest_categorical('model_type', ['XGB', 'RF', 'LGBM'])

    if model_type == 'XGB':
        p = dict(
            n_estimators     = trial.suggest_int  ('n_estimators',      100, 1000),
            max_depth        = trial.suggest_int  ('max_depth',            3,    9),
            learning_rate    = trial.suggest_float('learning_rate',    0.01,  0.3, log=True),
            subsample        = trial.suggest_float('subsample',         0.6,  1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree',  0.5,  1.0),
            min_child_weight = trial.suggest_int  ('min_child_weight',    1,   10),
            reg_alpha        = trial.suggest_float('reg_alpha',         1e-4,  10, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda',        1e-4,  10, log=True),
            scale_pos_weight = spw,
        )
        model = XGBClassifier(**p, tree_method='hist', device='cuda',
                              eval_metric='logloss', random_state=42,
                              n_jobs=-1, verbosity=0)
    elif model_type == 'RF':
        p = dict(
            n_estimators      = trial.suggest_int  ('n_estimators',    100, 500),
            max_depth         = trial.suggest_int  ('max_depth',          5,  20),
            min_samples_split = trial.suggest_int  ('min_samples_split',  2,  10),
            max_features      = trial.suggest_float('max_features',     0.3, 1.0),
            class_weight      = 'balanced',
        )
        model = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
    else:  # LGBM
        p = dict(
            n_estimators      = trial.suggest_int  ('n_estimators',       100, 1000),
            max_depth         = trial.suggest_int  ('max_depth',            3,   10),
            learning_rate     = trial.suggest_float('learning_rate',     0.01,  0.3, log=True),
            subsample         = trial.suggest_float('subsample',          0.6,  1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree',   0.5,  1.0),
            reg_alpha         = trial.suggest_float('reg_alpha',          1e-4,  10, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda',         1e-4,  10, log=True),
            scale_pos_weight  = spw,
            min_child_samples = trial.suggest_int  ('min_child_samples',    5,   50),
        )
        model = LGBMClassifier(**p, random_state=42, n_jobs=-1, verbose=-1)

    return model, model_type, p

# train
def objective(trial):
    model, model_type, p = build_model(trial)

    fold_aucs, fold_precs, fold_recs = [], [], []

    for train_dates, val_dates in cv.split(df_cv):
        tr = df_cv[df_cv['date'].isin(train_dates)]
        vl = df_cv[df_cv['date'].isin(val_dates)]

        X_tr, y_tr = tr[selected_cols], tr['y']
        X_v,  y_v  = vl[selected_cols], vl['y']

        sc = StandardScaler()
        model.fit(sc.fit_transform(X_tr), y_tr)
        probs = model.predict_proba(sc.transform(X_v))[:, 1]

        fold_aucs.append(roc_auc_score(y_v, probs))
        bp, br, _ = best_precision_at_recall(y_v, probs, PRUNE_RECALL_MIN)
        fold_precs.append(bp)
        fold_recs.append(br)

    cv_auc_mean  = float(np.mean(fold_aucs))
    cv_prec_mean = float(np.mean(fold_precs))
    cv_rec_mean  = float(np.mean(fold_recs))

    prune_reason = None
    if cv_auc_mean < PRUNE_AUC_MIN:
        prune_reason = f'CV AUC={cv_auc_mean:.4f} < {PRUNE_AUC_MIN}'
    elif cv_rec_mean < PRUNE_RECALL_MIN:
        prune_reason = f'CV Recall={cv_rec_mean:.4f} < {PRUNE_RECALL_MIN}'

    trial_details[trial.number] = {
        'model_type'     : model_type,
        'params'         : p,
        'pruned'         : prune_reason is not None,
        'prune_reason'   : prune_reason,
        'fold_aucs'      : fold_aucs,
        'fold_precs'     : fold_precs,
        'fold_recs'      : fold_recs,
        'cv_auc_mean'    : cv_auc_mean,
        'cv_auc_std'     : float(np.std(fold_aucs)),
        'cv_prec_mean'   : cv_prec_mean,
        'cv_rec_mean'    : cv_rec_mean,
    }

    if prune_reason:
        raise optuna.exceptions.TrialPruned()

    return cv_prec_mean


sampler = optuna.samplers.TPESampler(seed=42)
study   = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, show_progress_bar=True)

passed_trials = [t for t in study.trials if t.value is not None]
pruned_trials = [t for t in study.trials if t.value is None]

print(f"pass：{len(passed_trials)}  fail：{len(pruned_trials)}")

if not passed_trials:
    print("all defeat")
    exit()



#  best hyperparameters

best_trial  = study.best_trial
best_detail = trial_details[best_trial.number]
best_mtype  = best_detail['model_type']
best_p      = best_detail['params']

# holdout
def rebuild_model(model_type, params):
    if model_type == 'XGB':
        return XGBClassifier(**params, tree_method='hist', device='cuda',
                             eval_metric='logloss', random_state=42,
                             n_jobs=-1, verbosity=0)
    elif model_type == 'RF':
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    else:
        return LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)


final_model = rebuild_model(best_mtype, best_p)
final_model.fit(X_train_sc, y_train)

train_auc     = roc_auc_score(y_train, final_model.predict_proba(X_train_sc)[:, 1])
holdout_preds = final_model.predict_proba(X_test_sc)[:, 1]
holdout_auc   = roc_auc_score(y_test, holdout_preds)
holdout_ap    = average_precision_score(y_test, holdout_preds)

best_prec, best_rec, best_thr = best_precision_at_recall(
    y_test, holdout_preds, PRUNE_RECALL_MIN)
best_f1 = (2 * best_prec * best_rec / (best_prec + best_rec + 1e-9)
           if best_thr else 0.0)

print(f"\nTrain AUC      : {train_auc:.4f}")
print(f"CV AUC mean    : {best_detail['cv_auc_mean']:.4f}  std={best_detail['cv_auc_std']:.4f}")
print(f"Holdout AUC    : {holdout_auc:.4f}")
print(f"Holdout AP     : {holdout_ap:.4f}")
print(f"\nHoldout Highest Precision（Recall≥{PRUNE_RECALL_MIN}）：")
if best_thr:
    print(f"  Threshold       : {best_thr:.4f}")
    print(f"  Precision  : {best_prec:.4f}")
    print(f"  Recall     : {best_rec:.4f}")
    print(f"  F1         : {best_f1:.4f}")
    print(f"（threshold={best_thr:.4f}）：")
    print(classification_report(y_test, (holdout_preds >= best_thr).astype(int)))
else:
    print("Threshold too high")


# output
OUTPUT_DIR = r'C:\Users\xqwer\Downloads\projet\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plot_trial_history(study, trial_details,
    os.path.join(OUTPUT_DIR, 'optuna_trial_history.png'))
plot_roc_precision(y_test, holdout_preds, best_thr,
    label     = f'{best_mtype} Holdout',
    save_path = os.path.join(OUTPUT_DIR, 'holdout_roc_precision.png'))


# final check and save
cv_auc_mean = best_detail['cv_auc_mean']
cv_auc_std  = best_detail['cv_auc_std']

gate_results = {
    'Holdout AUC > 0.50'       : holdout_auc  > 0.50,
    'Holdout Recall > 0.30'    : best_rec     > 0.30,
    'CV-Holdout gap < 0.05'    : (cv_auc_mean - holdout_auc) < 0.05,
    'Train-CV gap < 0.05'      : (train_auc   - cv_auc_mean) < 0.05,
}

print(f"\n{'='*50}")
print("Gate result")

all_passed = True
for check, passed in gate_results.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {check}")
    if not passed: all_passed = False

if all_passed:
    print("Save to output")

    joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'final_model.pkl'))
    joblib.dump(scaler,      os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    report = {
        'seed'             : 42,
        'model_type'       : best_mtype,
        'best_params'      : best_p,
        'selected_cols'    : selected_cols,
        'removed_corr'     : removed_corr,
        'removed_vif'      : removed_vif,
        'corr_threshold'   : 0.95,
        'vif_threshold'    : 10,
        'train_date_range' : [str(unique_dates[0]),   str(unique_dates[cut-1])],
        'test_date_range'  : [str(unique_dates[cut]),  str(unique_dates[-1])],
        'train_auc'        : float(train_auc),
        'cv_auc_mean'      : float(cv_auc_mean),
        'cv_auc_std'       : float(cv_auc_std),
        'cv_prec_mean'     : float(best_detail['cv_prec_mean']),
        'cv_rec_mean'      : float(best_detail['cv_rec_mean']),
        'fold_aucs'        : best_detail['fold_aucs'],
        'fold_precs'       : best_detail['fold_precs'],
        'fold_recs'        : best_detail['fold_recs'],
        'holdout_auc'      : float(holdout_auc),
        'holdout_ap'       : float(holdout_ap),
        'best_threshold'   : float(best_thr) if best_thr else None,
        'best_precision'   : float(best_prec),
        'best_recall'      : float(best_rec),
        'best_f1'          : float(best_f1),
        'prune_criteria'   : {'auc_min': PRUNE_AUC_MIN, 'recall_min': PRUNE_RECALL_MIN},
        'gate_results'     : {k: bool(v) for k, v in gate_results.items()},
        'optuna_n_trials'  : 50,
        'optuna_passed'    : len(passed_trials),
        'optuna_pruned'    : len(pruned_trials),
    }

    with open(os.path.join(OUTPUT_DIR, 'model_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

else:
    print("Fail again try next time")

