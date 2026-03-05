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