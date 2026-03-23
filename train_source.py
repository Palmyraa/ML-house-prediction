# ------------- Cell 0 -------------
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Create results directory using relative paths
PROJECT_DIR = Path('../')
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Setup complete!")

# ------------- Cell 1 -------------
import kagglehub

# Download latest version
path = kagglehub.dataset_download("arunjangir245/boston-housing-dataset")

print("Path to dataset files:", path)

# ------------- Cell 2 -------------
# Load the dataset using the path from kagglehub
df = pd.read_csv(path + '/BostonHousing.csv')

# Handle missing values - fill all numeric columns with median
df = df.fillna(df.median())

# Verify no missing values
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# Define features and target
X = df.drop('medv', axis=1)
y = df['medv']

# Train/test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Save split data
X_train.to_csv(RESULTS_DIR / 'X_train.csv', index=False)
X_test.to_csv(RESULTS_DIR / 'X_test.csv', index=False)
y_train.to_csv(RESULTS_DIR / 'y_train.csv', index=False)
y_test.to_csv(RESULTS_DIR / 'y_test.csv', index=False)
print("Data saved to results/")

# ------------- Cell 3 -------------
from typing import Any, Dict, Tuple

def evaluate_model(
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    model_name: str
) -> Tuple[Dict[str, Any], Any]:
    """Evaluate model and return metrics"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics: Dict[str, Any] = {
        'model_name': model_name,
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    metrics['cv_r2_mean'] = cv_scores.mean()
    metrics['cv_r2_std'] = cv_scores.std()
    
    return metrics, y_test_pred

def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print model metrics"""
    print(f"\n{'='*50}")
    print(f"Model: {metrics['model_name']}")
    print(f"{'='*50}")
    print(f"Train MSE: {metrics['train_mse']:.4f} | Test MSE: {metrics['test_mse']:.4f}")
    print(f"Train RMSE: {metrics['train_rmse']:.4f} | Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.4f} | Test MAE: {metrics['test_mae']:.4f}")
    print(f"Train R²: {metrics['train_r2']:.4f} | Test R²: {metrics['test_r2']:.4f}")
    print(f"CV R² (mean±std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    # Overfitting/Underfitting analysis
    diff = metrics['train_r2'] - metrics['test_r2']
    if diff > 0.1:
        print(f"⚠️  Overfitting detected (train-test R² gap: {diff:.4f})")
    elif metrics['train_r2'] < 0.5 and metrics['test_r2'] < 0.5:
        print(f"⚠️  Underfitting detected (low R² on both sets)")
    else:
        print(f"✅ Good fit")

print("Evaluation functions defined!")

# ------------- Cell 4 -------------
# 1. UNIVARIATE LINEAR REGRESSION
# Using only 'rm' (rooms) - strongest correlation with target
print("="*60)
print("1. UNIVARIATE LINEAR REGRESSION (rm → medv)")
print("="*60)

X_train_uni = X_train[['rm']]
X_test_uni = X_test[['rm']]

lr_uni = LinearRegression()
lr_uni.fit(X_train_uni, y_train)

metrics_uni, pred_uni = evaluate_model(lr_uni, X_train_uni, X_test_uni, y_train, y_test, "Linear Regression (Univariate)")
print_metrics(metrics_uni)

# Save model and predictions
joblib.dump(lr_uni, str(RESULTS_DIR) + '/linear_univariate.joblib')
np.save(str(RESULTS_DIR) + '/pred_linear_univariate.npy', pred_uni)

# Save metrics
with open(str(RESULTS_DIR) + '/metrics_linear_univariate.json', 'w') as f:
    json.dump(metrics_uni, f, indent=2)

print("\n✅ Model and predictions saved!")

# ------------- Cell 5 -------------
# 2. MULTIVARIATE LINEAR REGRESSION
# Using all features
print("="*60)
print("2. MULTIVARIATE LINEAR REGRESSION (all features)")
print("="*60)

lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)

metrics_multi, pred_multi = evaluate_model(lr_multi, X_train, X_test, y_train, y_test, "Linear Regression (Multivariate)")
print_metrics(metrics_multi)

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_multi.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Save model and predictions
joblib.dump(lr_multi, str(RESULTS_DIR) + '/linear_multivariate.joblib')
np.save(str(RESULTS_DIR) + '/pred_linear_multivariate.npy', pred_multi)

# Save metrics
with open(str(RESULTS_DIR) + '/metrics_linear_multivariate.json', 'w') as f:
    json.dump(metrics_multi, f, indent=2)

print("\n✅ Model and predictions saved!")

# ------------- Cell 6 -------------
# 3. FEATURE SELECTION - Using top correlated features
print("="*60)
print("3. FEATURE SELECTION - Top Correlated Features")
print("="*60)

# Select top features based on correlation with target
correlations = df.corr()['medv'].drop('medv').abs().sort_values(ascending=False)
top_features = correlations.head(6).index.tolist()
print(f"Selected features: {top_features}")

X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]

lr_fs = LinearRegression()
lr_fs.fit(X_train_fs, y_train)

metrics_fs, pred_fs = evaluate_model(lr_fs, X_train_fs, X_test_fs, y_train, y_test, "Linear Regression (Feature Selection)")
print_metrics(metrics_fs)

# Save model and predictions
joblib.dump(lr_fs, str(RESULTS_DIR) + '/linear_feature_selection.joblib')
np.save(str(RESULTS_DIR) + '/pred_linear_feature_selection.npy', pred_fs)

with open(str(RESULTS_DIR) + '/metrics_linear_feature_selection.json', 'w') as f:
    json.dump(metrics_fs, f, indent=2)

print("\n✅ Model and predictions saved!")

# ------------- Cell 7 -------------
# 4. POLYNOMIAL REGRESSION
print("="*60)
print("4. POLYNOMIAL REGRESSION")
print("="*60)

results_poly = {}

for degree in [2, 3]:
    print(f"\n--- Degree {degree} ---")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_uni)
    X_test_poly = poly.transform(X_test_uni)
    
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    
    metrics_poly, pred_poly = evaluate_model(
        lr_poly, X_train_poly, X_test_poly, y_train, y_test, 
        f"Polynomial Regression (degree={degree})"
    )
    print_metrics(metrics_poly)
    
    results_poly[degree] = {
        'model': lr_poly,
        'metrics': metrics_poly,
        'predictions': pred_poly,
        'poly': poly
    }
    
    # Save model
    joblib.dump(lr_poly, f'{RESULTS_DIR}/polynomial_degree{degree}.joblib')
    joblib.dump(poly, f'{RESULTS_DIR}/polynomial_transformer_degree{degree}.joblib')
    np.save(f'{RESULTS_DIR}/pred_polynomial_degree{degree}.npy', pred_poly)
    
    with open(f'{RESULTS_DIR}/metrics_polynomial_degree{degree}.json', 'w') as f:
        json.dump(metrics_poly, f, indent=2)

# Compare degrees
print("\n" + "="*60)
print("POLYNOMIAL DEGREE COMPARISON")
print("="*60)
for degree, data in results_poly.items():
    m = data['metrics']
    print(f"Degree {degree}: Train R²={m['train_r2']:.4f}, Test R²={m['test_r2']:.4f}, CV R²={m['cv_r2_mean']:.4f}")

print("\n✅ Polynomial models saved!")

# ------------- Cell 8 -------------
# 5. GRADIENT DESCENT (SGDRegressor)
print("="*60)
print("5. GRADIENT DESCENT OPTIMIZATION (SGDRegressor)")
print("="*60)

# Scale features for gradient descent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SGDRegressor with different configurations
sgd_configs = [
    {'loss': 'squared_error', 'learning_rate': 'constant', 'eta0': 0.01, 'name': 'SGD (constant)'},
    {'loss': 'squared_error', 'learning_rate': 'adaptive', 'eta0': 0.01, 'name': 'SGD (adaptive)'},
]

results_sgd = {}

for config in sgd_configs:
    print(f"\n--- {config['name']} ---")
    
    sgd = SGDRegressor(
        loss=config['loss'],
        learning_rate=config['learning_rate'],
        eta0=config['eta0'],
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    sgd.fit(X_train_scaled, y_train)
    
    metrics_sgd, pred_sgd = evaluate_model(
        sgd, X_train_scaled, X_test_scaled, y_train, y_test,
        config['name']
    )
    print_metrics(metrics_sgd)
    
    results_sgd[config['name']] = {
        'model': sgd,
        'metrics': metrics_sgd,
        'predictions': pred_sgd
    }
    
    # Save model and scaler
    safe_name = config['name'].replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(sgd, f'{RESULTS_DIR}/{safe_name}.joblib')
    np.save(f'{RESULTS_DIR}/pred_{safe_name}.npy', pred_sgd)
    
    with open(f'{RESULTS_DIR}/metrics_{safe_name}.json', 'w') as f:
        json.dump(metrics_sgd, f, indent=2)

# Save scaler
joblib.dump(scaler, f'{RESULTS_DIR}/scaler.joblib')

print("\n✅ Gradient descent models saved!")

# ------------- Cell 9 -------------
# 5b. DECISION TREE REGRESSION
print("="*60)
print("5b. DECISION TREE REGRESSION")
print("="*60)
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train, y_train)
metrics_dt, pred_dt = evaluate_model(dt_reg, X_train, X_test, y_train, y_test, "Decision Tree Regression")
print_metrics(metrics_dt)
joblib.dump(dt_reg, str(RESULTS_DIR) + '/decision_tree.joblib')
np.save(str(RESULTS_DIR) + '/pred_decision_tree.npy', pred_dt)
with open(str(RESULTS_DIR) + '/metrics_decision_tree.json', 'w') as f:
    json.dump(metrics_dt, f, indent=2)
print("\n✅ Decision Tree model saved!")


# ------------- Cell 10 -------------
# 5c. NEURAL NETWORK REGRESSION (MLP)
print("="*60)
print("5c. NEURAL NETWORK REGRESSION (MLP)")
print("="*60)
nn_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
nn_reg.fit(X_train_scaled, y_train)
metrics_nn, pred_nn = evaluate_model(nn_reg, X_train_scaled, X_test_scaled, y_train, y_test, "Neural Network Regression")
print_metrics(metrics_nn)
joblib.dump(nn_reg, str(RESULTS_DIR) + '/neural_network.joblib')
np.save(str(RESULTS_DIR) + '/pred_neural_network.npy', pred_nn)
with open(str(RESULTS_DIR) + '/metrics_neural_network.json', 'w') as f:
    json.dump(metrics_nn, f, indent=2)
print("\n✅ Neural Network model saved!")


# ------------- Cell 11 -------------
# 6. CROSS-VALIDATION ANALYSIS
print("="*60)
print("6. CROSS-VALIDATION ANALYSIS (5-Fold)")
print("="*60)

models_to_cv = {
    'Linear (Uni)': (lr_uni, X_train_uni),
    'Linear (Multi)': (lr_multi, X_train),
    'Linear (FS)': (lr_fs, X_train_fs),
}

cv_results = {}

for name, (model, X_data) in models_to_cv.items():
    # R² cross-validation
    cv_r2 = cross_val_score(model, X_data, y_train, cv=5, scoring='r2')
    
    # Negative MSE cross-validation
    cv_mse = cross_val_score(model, X_data, y_train, cv=5, scoring='neg_mean_squared_error')
    
    cv_results[name] = {
        'r2_mean': cv_r2.mean(),
        'r2_std': cv_r2.std(),
        'mse_mean': -cv_mse.mean(),
        'mse_std': cv_mse.std()
    }
    
    print(f"\n{name}:")
    print(f"  R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  MSE: {-cv_mse.mean():.4f} ± {cv_mse.std():.4f}")

# Save CV results
with open(str(RESULTS_DIR) + '/cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

print("\n✅ Cross-validation results saved!")

# ------------- Cell 12 -------------
# 7. FINAL MODEL COMPARISON
print("="*60)
print("7. FINAL MODEL COMPARISON")
print("="*60)

all_metrics = [
    metrics_uni,
    metrics_multi,
    metrics_fs,
    results_poly[2]['metrics'],
    results_poly[3]['metrics'],
    results_sgd['SGD (constant)']['metrics'],
    results_sgd['SGD (adaptive)']['metrics'],
    metrics_dt,
    metrics_nn
]

comparison_df = pd.DataFrame(all_metrics)
comparison_df = comparison_df[['model_name', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse', 'cv_r2_mean', 'cv_r2_std']]
comparison_df = comparison_df.sort_values('test_r2', ascending=False)

print("\nModel Performance Ranking (by Test R²):")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(str(RESULTS_DIR) + '/model_comparison.csv', index=False)

best_model = comparison_df.iloc[0]['model_name']
best_r2 = comparison_df.iloc[0]['test_r2']

print(f"\n🏆 Best Model: {best_model}")
print(f"   Test R²: {best_r2:.4f}")

print("\n✅ Comparison saved!")

