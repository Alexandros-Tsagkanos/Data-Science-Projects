# Wine Quality Regression Analysis - Full Pipeline
# Dataset: UCI Red Wine Quality
# Σημ: random_state=42 παντού για αναπαραγωγιμότητα

import subprocess
import sys
import os
import warnings
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

# αυτόματη εγκατάσταση dependencies αν λείπουν (χρειάζεται σε fresh Colab)
for _pkg in ['tabulate', 'shap', 'nbformat']:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _pkg])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf
import shap

from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                      cross_validate, train_test_split)
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# βρίσκουμε δυναμικά το base dir - this took me forever to figure out
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / 'data'
FIG_DIR = BASE_DIR / 'figures'
RES_DIR = BASE_DIR / 'results'
SCRIPT_DIR = BASE_DIR / 'scripts'
REPORT_DIR = BASE_DIR / 'report'
NOTEBOOK_PATH = BASE_DIR / 'wine_quality_regression.ipynb'

for d in [DATA_DIR, FIG_DIR, RES_DIR, SCRIPT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")

UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
DATA_PATH = DATA_DIR / 'winequality-red.csv'
urlretrieve(UCI_URL, DATA_PATH)


def savefig(name, dpi=200, bbox_inches='tight'):
    fpath = FIG_DIR / name
    plt.savefig(fpath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()
    print(f"Saved: {fpath}")


def regression_metrics(y_true, y_pred):
    """Υπολογισμός βασικών μετρικών παλινδρόμησης"""
    mse_val = mean_squared_error(y_true, y_pred)
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mse_val,
        'RMSE': np.sqrt(mse_val),
        'R2': r2_score(y_true, y_pred)
    }


def get_models():
    # we need at least 4 models so here we go (βάζω 6 για σιγουριά)
    return {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('model', RandomForestRegressor(random_state=SEED))
        ]),
        'GradientBoosting': Pipeline([
            ('model', GradientBoostingRegressor(random_state=SEED))
        ]),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR())
        ]),
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(random_state=SEED, max_iter=10000))
        ]),
        'KNN': Pipeline([('scaler', StandardScaler()),
                         ('model', KNeighborsRegressor())])
    }


def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te):
    collected = []
    fitted = {}
    for name, model in models_dict.items():
        est = clone(model)
        est.fit(X_tr, y_tr)
        preds = est.predict(X_te)
        m = regression_metrics(y_te, preds)
        collected.append({'model': name, **m})
        fitted[name] = est
    return pd.DataFrame(collected).sort_values('RMSE').reset_index(drop=True), fitted


# ==========================
# PHASE 1 - EDA (Exploratory Data Analysis)
# =======================================
print('Running Phase 1 - EDA...')
df = pd.read_csv(DATA_PATH, sep=';')
features = [c for c in df.columns if c != 'quality']
X = df[features].copy()
y = df['quality'].copy()

# NOTE: keeping duplicates as discussed in class
# df_no_outliers = df[~any_outlier_mask]  # tried removing outliers but results were worse

# περιγραφικά στατιστικά
summary = df.describe(include='all').T
summary['median'] = df.median(numeric_only=True)
summary['skew'] = df.skew(numeric_only=True)
summary['kurtosis'] = df.kurtosis(numeric_only=True)
summary['missing_count'] = df.isna().sum()
summary['missing_pct'] = (df.isna().mean() * 100)
summary.to_csv(RES_DIR / 'descriptive_statistics.csv')
print(f"Saved: {RES_DIR / 'descriptive_statistics.csv'}")

quality_dist = df['quality'].value_counts().sort_index().rename_axis('quality').reset_index(name='count')
quality_dist.to_csv(RES_DIR / 'target_distribution.csv', index=False)
print("Saved: {}".format(RES_DIR / 'target_distribution.csv'))

corr = df.corr(numeric_only=True)
corr.to_csv(RES_DIR / 'correlation_matrix.csv')
print(f"Saved: {RES_DIR / 'correlation_matrix.csv'}")

# heatmap συσχετίσεων
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap - Red Wine Quality Dataset')
savefig('phase1_correlation_heatmap.png')

# boxplots για κάθε feature
n_cols = 3
n_rows = int(np.ceil(len(features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(y=df[col], ax=axes[i], color='#8ecae6')
    axes[i].set_title(f'Boxplot - {col}')
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.suptitle('Feature Boxplots', y=1.02, fontsize=16)
plt.tight_layout()
savefig('phase1_boxplots_features.png')

plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, color='#219ebc')
plt.title('Target Distribution - Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
savefig('phase1_target_distribution.png')

# pairplot - αυτό παίρνει λίγο ώρα
pairplot_data = df.copy()
sns.pairplot(pairplot_data, corner=True, diag_kind='hist',
             plot_kws={'alpha': 0.4, 's': 12, 'edgecolor': 'none'})
plt.suptitle('Pairplot - Red Wine Dataset', y=1.02)
plt.gcf().set_size_inches(24, 24)
plt.tight_layout()
plt.savefig(FIG_DIR / 'phase1_pairplot.png', dpi=150, bbox_inches='tight')
plt.close('all')
print(f"Saved: {FIG_DIR / 'phase1_pairplot.png'}")

# IQR outlier detection
outlier_info = []
outlier_mask_df = pd.DataFrame(index=df.index)
for col in features + ['quality']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr_val = q3 - q1
    lo = q1 - 1.5 * iqr_val
    hi = q3 + 1.5 * iqr_val
    mask = (df[col] < lo) | (df[col] > hi)
    outlier_mask_df[col] = mask
    outlier_info.append({
        'feature': col, 'Q1': q1, 'Q3': q3, 'IQR': iqr_val,
        'lower_bound': lo, 'upper_bound': hi,
        'outlier_count': int(mask.sum()),
        'outlier_pct': float(mask.mean() * 100)
    })

outlier_summary = pd.DataFrame(outlier_info).sort_values('outlier_count', ascending=False)
outlier_summary.to_csv(RES_DIR / 'iqr_outlier_summary.csv', index=False)
print(f"Saved: {RES_DIR / 'iqr_outlier_summary.csv'}")
outlier_mask_df.to_csv(RES_DIR / 'iqr_outlier_flags.csv', index=False)
print(f"Saved: {RES_DIR / 'iqr_outlier_flags.csv'}")

plt.figure(figsize=(12, 6))
sns.barplot(data=outlier_summary, x='outlier_count', y='feature', palette='viridis')
plt.title('IQR Outlier Count by Variable')
plt.xlabel('Outlier Count')
plt.ylabel('Variable')
savefig('phase1_iqr_outlier_counts.png')

print('Done with EDA!')


# ======================
# PHASE 2 - PREPROCESSING
# ==================================
print('Running Phase 2 - Preprocessing...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# documentation για preprocessing
preprocess_doc = f"""# Preprocessing Documentation\n\n- Dataset source: UCI Machine Learning Repository ({UCI_URL})\n- Records retained: {len(df)} (duplicates intentionally preserved as valid observations)\n- Features: {len(features)}\n- Target: quality\n- Train/Test split: 80/20 with random_state=42\n- Training rows: {len(X_train)}\n- Test rows: {len(X_test)}\n- Scaling strategy: StandardScaler inside sklearn Pipeline for scale-sensitive models (Linear Regression, SVR, ElasticNet, KNN).\n- Tree models (Random Forest, Gradient Boosting) were left unscaled in a Pipeline for methodological consistency and leakage prevention.\n- Missing values: none detected\n- Outliers: detected and documented via IQR, but not removed because the task explicitly requires keeping all observations.\n"""
preprocess_path = REPORT_DIR / 'preprocessing_documentation.md'
preprocess_path.write_text(preprocess_doc, encoding='utf-8')
print(f"Saved: {preprocess_path}")

split_df = pd.DataFrame({
    'dataset': ['full', 'train', 'test'],
    'rows': [len(df), len(X_train), len(X_test)],
    'pct_of_full': [100, len(X_train) / len(df) * 100, len(X_test) / len(df) * 100]
})
split_df.to_csv(RES_DIR / 'split_summary.csv', index=False)
print(f"Saved: {RES_DIR / 'split_summary.csv'}")


# ==============================================
# PHASE 3 - BASELINE MODELS
# ============================================================
print('Running Phase 3 - Baseline models...')
models = get_models()
baseline_metrics, fitted_baseline = evaluate_models(models, X_train, X_test, y_train, y_test)
baseline_metrics.to_csv(RES_DIR / 'baseline_metrics.csv', index=False)
print(f"Saved: {RES_DIR / 'baseline_metrics.csv'}")


# ==========================================
# PHASE 4 - 10-FOLD CV
# ==========================================
print('Running Phase 4 - 10-fold cross-validation...')
scoring = {'MAE': 'neg_mean_absolute_error', 'MSE': 'neg_mean_squared_error',
           'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}
cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

cv_collected = []
for name, model in models.items():
    scores = cross_validate(clone(model), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_collected.append({
        'model': name,
        'cv_MAE_mean': -scores['test_MAE'].mean(),
        'cv_MAE_std': scores['test_MAE'].std(),
        'cv_MSE_mean': -scores['test_MSE'].mean(),
        'cv_MSE_std': scores['test_MSE'].std(),
        'cv_RMSE_mean': -scores['test_RMSE'].mean(),
        'cv_RMSE_std': scores['test_RMSE'].std(),
        'cv_R2_mean': scores['test_R2'].mean(),
        'cv_R2_std': scores['test_R2'].std()
    })
cv_results = pd.DataFrame(cv_collected).sort_values('cv_RMSE_mean').reset_index(drop=True)
cv_results.to_csv(RES_DIR / 'cv_results.csv', index=False)
print("Saved: {}".format(RES_DIR / 'cv_results.csv'))

# =============================================
# PHASE 5 - FEATURE SELECTION
# ==================================================
print('Running Phase 5 - Feature selection...')
fs_collected = []

def fs_eval(selected_feats, method_name, setting_name):
    """Αξιολόγηση ενός subset features με LinearRegression"""
    sel = list(selected_feats)
    if len(sel) == 0:
        return None
    pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    pipe.fit(X_train[sel], y_train)
    pred = pipe.predict(X_test[sel])
    m = regression_metrics(y_test, pred)
    fs_collected.append({
        'method': method_name, 'setting': setting_name,
        'n_features': len(sel),
        'selected_features': ', '.join(sel),
        **m
    })

# Pearson correlation ranking
pearson_abs = X_train.assign(quality=y_train).corr(numeric_only=True)['quality'].drop('quality').abs().sort_values(ascending=False)
pearson_abs.rename('abs_pearson_with_quality').to_csv(RES_DIR / 'pearson_feature_ranking.csv')
print(f"Saved: {RES_DIR / 'pearson_feature_ranking.csv'}")

for k in range(3, len(features) + 1):
    top_k = pearson_abs.index[:k].tolist()
    fs_eval(top_k, 'Pearson', f'top_{k}')

# Mutual information - αυτό είναι αρκετά χρήσιμο
mi_scores = mutual_info_regression(X_train, y_train, random_state=SEED)
mi_ranking = pd.Series(mi_scores, index=features).sort_values(ascending=False)
mi_ranking.rename('mutual_information').to_csv(RES_DIR / 'mutual_information_ranking.csv')
print(f"Saved: {RES_DIR / 'mutual_information_ranking.csv'}")
for k in range(3, len(features) + 1):
    top_k = mi_ranking.index[:k].tolist()
    fs_eval(top_k, 'MutualInformation', 'top_{}'.format(k))

# RFE with Linear Regression
for k in range(3, len(features) + 1):
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', rfe),
        ('model', LinearRegression())
    ])
    rfe_pipe.fit(X_train, y_train)
    support_mask = rfe_pipe.named_steps['selector'].support_
    sel = X_train.columns[support_mask].tolist()
    pred = rfe_pipe.predict(X_test)
    m = regression_metrics(y_test, pred)
    fs_collected.append({
        'method': 'RFE', 'setting': f'top_{k}',
        'n_features': len(sel),
        'selected_features': ', '.join(sel), **m
    })

# Lasso embedded - TODO: maybe try more alphas later
alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
for alpha in alpha_values:
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    selector = SelectFromModel(Lasso(alpha=alpha, random_state=SEED, max_iter=10000))
    selector.fit(X_train_sc, y_train)
    support_mask = selector.get_support()
    sel = X_train.columns[support_mask].tolist()
    if len(sel) == 0:
        continue
    X_tr_sel = selector.transform(X_train_sc)
    X_te_sel = selector.transform(X_test_sc)
    lr = LinearRegression()
    lr.fit(X_tr_sel, y_train)
    pred = lr.predict(X_te_sel)
    m = regression_metrics(y_test, pred)
    fs_collected.append({
        'method': 'LassoEmbedded',
        'setting': f'alpha_{alpha}',
        'n_features': len(sel),
        'selected_features': ', '.join(sel),
        **m
    })

feature_selection_results = pd.DataFrame(fs_collected).sort_values(['RMSE', 'MAE']).reset_index(drop=True)
feature_selection_results.to_csv(RES_DIR / 'feature_selection_results.csv', index=False)
print(f"Saved: {RES_DIR / 'feature_selection_results.csv'}")
feature_selection_results.groupby('method').head(1).to_csv(RES_DIR / 'feature_selection_best_by_method.csv', index=False)
print(f"Saved: {RES_DIR / 'feature_selection_best_by_method.csv'}")


# ==================================================
# PHASE 6 - HYPERPARAMETER TUNING
# ============================================================
print('Running Phase 6 - Hyperparameter tuning...')

# παίρνουμε τα 2 καλύτερα μοντέλα από CV
top2_models = cv_results.nsmallest(2, 'cv_RMSE_mean')['model'].tolist()

tuning_candidates = {}

if 'RandomForest' in top2_models:
    tuning_candidates['RandomForest'] = (
        Pipeline([('model', RandomForestRegressor(random_state=SEED))]),
        {'model__n_estimators': [100, 200, 300, 500],
         'model__max_depth': [None, 5, 10, 20, 30],
         'model__min_samples_split': [2, 5, 10],
         'model__min_samples_leaf': [1, 2, 4],
         'model__max_features': ['sqrt', 'log2', None]},
        'random'
    )

if 'GradientBoosting' in top2_models:
    tuning_candidates['GradientBoosting'] = (
        Pipeline([('model', GradientBoostingRegressor(random_state=SEED))]),
        {'model__n_estimators': [100, 150, 200, 300],
         'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
         'model__max_depth': [2, 3, 4, 5],
         'model__subsample': [0.7, 0.85, 1.0],
         'model__min_samples_split': [2, 5, 10],
         'model__min_samples_leaf': [1, 2, 4]},
        'random'
    )

if 'SVR' in top2_models:
    tuning_candidates['SVR'] = (
        Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
        {'model__kernel': ['rbf', 'linear'],
         'model__C': [0.1, 1, 10, 50, 100],
         'model__gamma': ['scale', 'auto', 0.01, 0.1],
         'model__epsilon': [0.01, 0.05, 0.1, 0.2]},
        'grid'
    )

if 'KNN' in top2_models:
    tuning_candidates['KNN'] = (
        Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())]),
        {'model__n_neighbors': list(range(3, 31, 2)),
         'model__weights': ['uniform', 'distance'],
         'model__p': [1, 2]},
        'grid'
    )

if 'ElasticNet' in top2_models:
    tuning_candidates['ElasticNet'] = (
        Pipeline([('scaler', StandardScaler()),
                  ('model', ElasticNet(random_state=SEED, max_iter=10000))]),
        {'model__alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
         'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'grid'
    )

if 'LinearRegression' in top2_models:
    tuning_candidates['LinearRegression'] = (
        Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        {'model__fit_intercept': [True, False]},
        'grid'
    )

tuned_collected = []
tuned_estimators = {}
tuning_searches = {}

for name in top2_models:
    mdl, params, mode = tuning_candidates[name]
    if mode == 'random':
        search = RandomizedSearchCV(
            estimator=mdl, param_distributions=params,
            n_iter=15, scoring='neg_root_mean_squared_error',
            n_jobs=-1, cv=cv, random_state=SEED, refit=True
        )
    else:
        search = GridSearchCV(
            estimator=mdl, param_grid=params,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1, cv=cv, refit=True
        )

    search.fit(X_train, y_train)
    best_est = search.best_estimator_
    pred = best_est.predict(X_test)
    m = regression_metrics(y_test, pred)

    tuned_collected.append({
        'model': name,
        'search_type': mode,
        'best_cv_RMSE': -search.best_score_,
        'best_params': json.dumps(search.best_params_),
        'test_MAE': m['MAE'],
        'test_MSE': m['MSE'],
        'test_RMSE': m['RMSE'],
        'test_R2': m['R2']
    })
    tuned_estimators[name] = best_est
    tuning_searches[name] = search

tuned_metrics = pd.DataFrame(tuned_collected).sort_values('best_cv_RMSE').reset_index(drop=True)
tuned_metrics.to_csv(RES_DIR / 'tuned_metrics.csv', index=False)
print(f"Saved: {RES_DIR / 'tuned_metrics.csv'}")

# Επιλογή τελικού μοντέλου - strictly by training CV
if len(tuned_metrics) > 0:
    final_model_name = tuned_metrics.iloc[0]['model']
    final_model = tuned_estimators[final_model_name]
else:
    final_model_name = cv_results.iloc[0]['model']
    final_model = fitted_baseline[final_model_name]

final_pred = final_model.predict(X_test)
final_metrics = regression_metrics(y_test, final_pred)
pd.DataFrame([{'model': final_model_name, **final_metrics}]).to_csv(
    RES_DIR / 'final_model_metrics.csv', index=False)
print(f"Saved: {RES_DIR / 'final_model_metrics.csv'}")


# ===============================================
# PHASE 7 - RESIDUAL ANALYSIS
# ============================================================
print('Running Phase 7 - Residual analysis...')
residuals = y_test - final_pred

residual_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': final_pred,
    'residual': residuals
})
residual_df.to_csv(RES_DIR / 'residuals.csv', index=False)
print(f"Saved: {RES_DIR / 'residuals.csv'}")

# residuals vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=final_pred, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title(f'Residuals vs Predicted - {final_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
savefig('phase7_residuals_vs_predicted.png')

# actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=final_pred, alpha=0.7)
lims = [min(y_test.min(), final_pred.min()), max(y_test.max(), final_pred.max())]
plt.plot(lims, lims, 'r--')
plt.title("Actual vs Predicted - {}".format(final_model_name))
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
savefig('phase7_actual_vs_predicted.png')

# not sure if this is the best way but it works
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='#90be6d')
plt.title(f'Residual Distribution - {final_model_name}')
plt.xlabel('Residual')
savefig('phase7_residual_distribution.png')

# qq plot
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist='norm', plot=plt)
plt.title(f'QQ Plot of Residuals - {final_model_name}')
savefig('phase7_qq_plot.png')


# ==============================================
# PHASE 8 - SHAP INTERPRETABILITY
# ============================================================
print('Running Phase 8 - SHAP...')

# Προτιμάμε tree-based μοντέλο για SHAP (πιο γρήγορο + TreeExplainer)
shap_model_name = final_model_name
shap_model = final_model

if final_model_name not in ['RandomForest', 'GradientBoosting']:
    tree_pool = {**fitted_baseline, **tuned_estimators}
    for cand in ['RandomForest', 'GradientBoosting']:
        if cand in tree_pool:
            shap_model_name = cand
            shap_model = tree_pool[cand]
            break

if 'scaler' in shap_model.named_steps:
    X_train_transformed = shap_model.named_steps['scaler'].transform(X_train)
    X_test_transformed = shap_model.named_steps['scaler'].transform(X_test)
    transformed_feature_names = features
    model_for_shap = shap_model.named_steps['model']
    shap_explainer = shap.Explainer(model_for_shap, X_train_transformed)
    shap_values = shap_explainer(X_test_transformed)
    shap_X = pd.DataFrame(X_test_transformed, columns=transformed_feature_names)
else:
    model_for_shap = shap_model.named_steps['model']
    shap_explainer = shap.TreeExplainer(model_for_shap)
    shap_values = shap_explainer.shap_values(X_test)
    shap_X = X_test.copy()

# beeswarm summary
plt.figure()
shap.summary_plot(shap_values, shap_X, show=False)
plt.title(f'SHAP Summary Plot - {shap_model_name}')
plt.tight_layout()
savefig('phase8_shap_summary.png')

plt.figure()
shap.summary_plot(shap_values, shap_X, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance (Bar) - {shap_model_name}')
plt.tight_layout()
savefig('phase8_shap_bar.png')

# υπολογισμός mean |SHAP|
if isinstance(shap_values, np.ndarray):
    shap_arr = shap_values
else:
    shap_arr = shap_values.values

mean_abs = np.abs(shap_arr).mean(axis=0)
shap_importance = pd.Series(mean_abs, index=shap_X.columns).sort_values(ascending=False)
shap_importance.rename('mean_abs_shap').to_csv(RES_DIR / 'shap_importance.csv')
print(f"Saved: {RES_DIR / 'shap_importance.csv'}")

# dependence plots για τα top 3 features
top3_shap = shap_importance.head(3).index.tolist()
for feat in top3_shap:
    plt.figure()
    shap.dependence_plot(feat, shap_arr, shap_X, show=False)
    plt.title(f'SHAP Dependence Plot - {feat}')
    plt.tight_layout()
    safe_name = feat.replace(' ', '_')
    savefig(f'phase8_shap_dependence_{safe_name}.png')


# ======================================
# PHASE 9 - MODEL COMPARISON & SUMMARY
# ==========================================
print('Running Phase 9 - Model comparison...')

big_picture = baseline_metrics.merge(cv_results, on='model', how='left')
if len(tuned_metrics) > 0:
    big_picture = big_picture.merge(
        tuned_metrics[['model', 'best_cv_RMSE', 'search_type', 'best_params',
                        'test_MAE', 'test_MSE', 'test_RMSE', 'test_R2']].rename(columns={
            'test_MAE': 'tuned_test_MAE', 'test_MSE': 'tuned_test_MSE',
            'test_RMSE': 'tuned_test_RMSE', 'test_R2': 'tuned_test_R2'
        }),
        on='model', how='left'
    )
big_picture.to_csv(RES_DIR / 'final_comparison.csv', index=False)
print(f"Saved: {RES_DIR / 'final_comparison.csv'}")

# radar chart - κανονικοποιημένο
radar_df = baseline_metrics.copy().set_index('model')
radar_df['inv_MAE'] = 1 / radar_df['MAE']
radar_df['inv_MSE'] = 1 / radar_df['MSE']
radar_df['inv_RMSE'] = 1 / radar_df['RMSE']
radar_plot_df = radar_df[['inv_MAE', 'inv_MSE', 'inv_RMSE', 'R2']]
radar_plot_df = (radar_plot_df - radar_plot_df.min()) / (radar_plot_df.max() - radar_plot_df.min())
radar_plot_df = radar_plot_df.fillna(0)

labels = radar_plot_df.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)
for mdl_name, row in radar_plot_df.iterrows():
    vals = row.tolist()
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, label=mdl_name)
    ax.fill(angles, vals, alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title('Radar Chart - Baseline Model Comparison (Normalized)')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
savefig('phase9_model_radar_chart.png')

# quick bar comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=baseline_metrics.sort_values('RMSE'), x='RMSE', y='model', palette='magma')
plt.title('Baseline Model RMSE Comparison')
savefig('phase9_baseline_rmse_comparison.png')

# Permutation importance τελικού μοντέλου
print('Computing permutation importance...')
perm = permutation_importance(final_model, X_test, y_test,
                               n_repeats=20, random_state=SEED,
                               scoring='neg_root_mean_squared_error')

perm_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False)

perm_importance_df.to_csv(RES_DIR / 'permutation_importance_final_model.csv', index=False)
print(f"Saved: {RES_DIR / 'permutation_importance_final_model.csv'}")

plt.figure(figsize=(10, 6))
sns.barplot(data=perm_importance_df, x='importance_mean', y='feature', palette='crest')
plt.title("Permutation Importance - Final Model ({})".format(final_model_name))
savefig('phase9_permutation_importance_final_model.png')

# τελικό summary σε markdown
summary_lines = [
    '# Wine Quality Regression Pipeline Summary',
    '',
    f'- Final selected model: **{final_model_name}**',
    f"- Final test MAE: **{final_metrics['MAE']:.4f}**",
    f"- Final test RMSE: **{final_metrics['RMSE']:.4f}**",
    f"- Final test R²: **{final_metrics['R2']:.4f}**",
    '',
    '## Top baseline models by RMSE',
    baseline_metrics.to_markdown(index=False),
    '',
    '## Top 10-fold CV results',
    cv_results.to_markdown(index=False),
    '',
    '## Best feature-selection result per method',
    feature_selection_results.groupby('method').head(1).to_markdown(index=False),
    '',
    '## Tuned models (ranked by training CV RMSE; test metrics shown only for final diagnostic reporting)',
    tuned_metrics.to_markdown(index=False) if len(tuned_metrics) > 0 else 'No tuned models available.'
]
summary_path = REPORT_DIR / 'pipeline_summary.md'
summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
print(f"Saved: {summary_path}")


# ============================================================
# PHASE 10 - NOTEBOOK GENERATION
# ============================================================
print('Running Phase 10 - Creating notebook...')
nb = nbf.v4.new_notebook()

cells = []
cells.append(nbf.v4.new_markdown_cell("""### Ανάλυση Παλινδρόμησης για το Red Wine Quality Dataset\n\nΑυτό το notebook υλοποιεί ολόκληρο το 10-φασικό pipeline μηχανικής μάθησης για το σύνολο δεδομένων **Wine Quality Red** από το UCI.\n\n- Διατηρούμε **και τις 1599 εγγραφές**\n- **Δεν αφαιρούμε διπλότυπα**, επειδή θεωρούνται επιστημονικά έγκυρες επαναλαμβανόμενες παρατηρήσεις\n- Χρησιμοποιούμε **random_state=42** όπου είναι δυνατό\n- Ο στόχος `quality` αντιμετωπίζεται ως συνεχής μεταβλητή για πρόβλημα παλινδρόμησης\n"""))

imports_code = """# Εισαγωγή βιβλιοθηκών
import os
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy import stats
from sklearn.base import clone
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Auto-detect project root (works on both Colab and local)
BASE_DIR = Path(os.getcwd())
if (BASE_DIR / 'data').exists() or (BASE_DIR / 'scripts').exists():
    pass  # already in project root
elif (BASE_DIR / 'wine_quality_analysis').exists():
    BASE_DIR = BASE_DIR / 'wine_quality_analysis'
DATA_DIR = BASE_DIR / 'data'
FIG_DIR = BASE_DIR / 'figures'
RES_DIR = BASE_DIR / 'results'
for d in [DATA_DIR, FIG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
DATA_PATH = DATA_DIR / 'winequality-red.csv'
print(f"Base directory: {BASE_DIR}")
"""
cells.append(nbf.v4.new_code_cell(imports_code))

cells.append(nbf.v4.new_markdown_cell("### Φάση 1 — EDA\nΕξετάζουμε τη δομή των δεδομένων, τα βασικά στατιστικά, τις συσχετίσεις και τα outliers."))

eda_code = """# Φόρτωση δεδομένων από το UCI
urlretrieve(UCI_URL, DATA_PATH)
df = pd.read_csv(DATA_PATH, sep=';')
features = [c for c in df.columns if c != 'quality']
X = df[features].copy()
y = df['quality'].copy()

print(df.shape)
print(df.head())
print('Διπλότυπα:', df.duplicated().sum())
print(df['quality'].value_counts().sort_index())

# Περιγραφικά στατιστικά
summary = df.describe().T
summary['median'] = df.median(numeric_only=True)
summary['skew'] = df.skew(numeric_only=True)
summary['kurtosis'] = df.kurtosis(numeric_only=True)
summary
"""
cells.append(nbf.v4.new_code_cell(eda_code))

cells.append(nbf.v4.new_markdown_cell("### Φάση 2 — Προεπεξεργασία\nΚάνουμε split 80/20 και χρησιμοποιούμε StandardScaler μέσα σε Pipeline ώστε να αποφύγουμε leakage."))
pre_code = """# Διαχωρισμός train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Ορισμός pipelines για όλα τα baseline μοντέλα
def get_models():
    return {
        'LinearRegression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'RandomForest': Pipeline([('model', RandomForestRegressor(random_state=SEED))]),
        'GradientBoosting': Pipeline([('model', GradientBoostingRegressor(random_state=SEED))]),
        'SVR': Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
        'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(random_state=SEED, max_iter=10000))]),
        'KNN': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())]),
    }

models = get_models()
X_train.shape, X_test.shape
"""
cells.append(nbf.v4.new_code_cell(pre_code))

cells.append(nbf.v4.new_markdown_cell("### Φάσεις 3 έως 9\nΕκπαίδευση baseline μοντέλων, 10-fold CV, feature selection, tuning, residual analysis, SHAP και τελική σύγκριση."))

run_cell_code = """import os
from pathlib import Path
_base = Path(os.getcwd())
if (_base / 'scripts').exists():
    _script = _base / 'scripts' / 'full_pipeline.py'
elif (_base / 'wine_quality_analysis' / 'scripts').exists():
    _script = _base / 'wine_quality_analysis' / 'scripts' / 'full_pipeline.py'
else:
    _script = 'scripts/full_pipeline.py'
# Για το πλήρες reproducible execution, εκτελούμε το αποθηκευμένο script
%run {str(_script)}
"""
cells.append(nbf.v4.new_code_cell(run_cell_code))

cells.append(nbf.v4.new_markdown_cell("### Φάση 10 — Παραδοτέα\nΤα αποτελέσματα έχουν αποθηκευτεί στους φακέλους `figures/`, `results/` και `report/`."))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'}
}
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Saved: {NOTEBOOK_PATH}")

print('Pipeline completed successfully.')
print(f'Final model: {final_model_name}')
print(final_metrics)
