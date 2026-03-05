# ------------- Cell 0 -------------
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Paths using pathlib with relative paths
PROJECT_DIR = Path('../')
RESULTS_DIR = PROJECT_DIR / 'results'

print("Setup complete!")

# ------------- Cell 1 -------------
# Load test data
X_test = pd.read_csv(RESULTS_DIR / 'X_test.csv')
y_test = pd.read_csv(RESULTS_DIR / 'y_test.csv')['medv']
X_train = pd.read_csv(RESULTS_DIR / 'X_train.csv')
y_train = pd.read_csv(RESULTS_DIR / 'y_train.csv')['medv']

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"Features: {X_test.shape[1]}")

# ------------- Cell 2 -------------
# Load all models and predictions
models = {}
predictions = {}
metrics = {}

# Linear models
models['Linear Univariate'] = joblib.load(RESULTS_DIR / 'linear_univariate.joblib')
predictions['Linear Univariate'] = np.load(RESULTS_DIR / 'pred_linear_univariate.npy')
with open(RESULTS_DIR / 'metrics_linear_univariate.json') as f:
    metrics['Linear Univariate'] = json.load(f)

models['Linear Multivariate'] = joblib.load(RESULTS_DIR / 'linear_multivariate.joblib')
predictions['Linear Multivariate'] = np.load(RESULTS_DIR / 'pred_linear_multivariate.npy')
with open(RESULTS_DIR / 'metrics_linear_multivariate.json') as f:
    metrics['Linear Multivariate'] = json.load(f)

models['Linear Feature Selection'] = joblib.load(RESULTS_DIR / 'linear_feature_selection.joblib')
predictions['Linear Feature Selection'] = np.load(RESULTS_DIR / 'pred_linear_feature_selection.npy')
with open(RESULTS_DIR / 'metrics_linear_feature_selection.json') as f:
    metrics['Linear Feature Selection'] = json.load(f)

# Polynomial models
models['Polynomial Degree 2'] = joblib.load(RESULTS_DIR / 'polynomial_degree2.joblib')
predictions['Polynomial Degree 2'] = np.load(RESULTS_DIR / 'pred_polynomial_degree2.npy')
with open(RESULTS_DIR / 'metrics_polynomial_degree2.json') as f:
    metrics['Polynomial Degree 2'] = json.load(f)

models['Polynomial Degree 3'] = joblib.load(RESULTS_DIR / 'polynomial_degree3.joblib')
predictions['Polynomial Degree 3'] = np.load(RESULTS_DIR / 'pred_polynomial_degree3.npy')
with open(RESULTS_DIR / 'metrics_polynomial_degree3.json') as f:
    metrics['Polynomial Degree 3'] = json.load(f)

# SGD models
models['SGD Constant'] = joblib.load(RESULTS_DIR / 'SGD_constant.joblib')
predictions['SGD Constant'] = np.load(RESULTS_DIR / 'pred_SGD_constant.npy')
with open(RESULTS_DIR / 'metrics_SGD_constant.json') as f:
    metrics['SGD Constant'] = json.load(f)

models['SGD Adaptive'] = joblib.load(RESULTS_DIR / 'SGD_adaptive.joblib')
predictions['SGD Adaptive'] = np.load(RESULTS_DIR / 'pred_SGD_adaptive.npy')
with open(RESULTS_DIR / 'metrics_SGD_adaptive.json') as f:
    metrics['SGD Adaptive'] = json.load(f)

# Decision Tree
models['Decision Tree'] = joblib.load(RESULTS_DIR / 'decision_tree.joblib')
predictions['Decision Tree'] = np.load(RESULTS_DIR / 'pred_decision_tree.npy')
with open(RESULTS_DIR / 'metrics_decision_tree.json') as f:
    metrics['Decision Tree'] = json.load(f)

# Neural Network
models['Neural Network'] = joblib.load(RESULTS_DIR / 'neural_network.joblib')
predictions['Neural Network'] = np.load(RESULTS_DIR / 'pred_neural_network.npy')
with open(RESULTS_DIR / 'metrics_neural_network.json') as f:
    metrics['Neural Network'] = json.load(f)

print(f"Loaded {len(models)} models")

# ------------- Cell 3 -------------
# 1. ACTUAL VS PREDICTED PLOTS (Interactive Plotly)
n_models = len(predictions)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"{name} (R² = {metrics[name]['test_r2']:.3f})" for name in predictions.keys()],
    vertical_spacing=0.12/n_rows
)

model_names = list(predictions.keys())

for idx, (name, pred) in enumerate(predictions.items()):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=pred,
            mode='markers',
            marker=dict(size=8, opacity=0.6, line=dict(width=1, color='black')),
            name=name,
            text=[f'Actual: ${x:.1f}k<br>Predicted: ${y:.1f}k' for x, y in zip(y_test, pred)],
            hoverinfo='text'
        ),
        row=row, col=col
    )
    
    # Perfect prediction line
    x_line = [y_test.min(), y_test.max()]
    y_line = [y_test.min(), y_test.max()]
    fig.add_trace(
        go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Actual Price ($1000s)", row=row, col=col)
    fig.update_yaxes(title_text="Predicted Price ($1000s)", row=row, col=col)

fig.update_layout(
    height=400*n_rows,
    title_text="Actual vs Predicted Prices (Interactive - Hover for details)",
    showlegend=True,
    template='plotly_white'
)

fig.write_image(f"{RESULTS_DIR}/actual_vs_predicted.png", width=1600, height=400*n_rows, scale=2)
fig.show()
print("Saved: actual_vs_predicted.png")

# ------------- Cell 4 -------------
# 2. RESIDUAL PLOTS (Interactive Plotly)
n_models = len(predictions)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=list(predictions.keys()),
    vertical_spacing=0.12/n_rows
)

for idx, (name, pred) in enumerate(predictions.items()):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    residuals = y_test - pred
    
    fig.add_trace(
        go.Scatter(
            x=pred,
            y=residuals,
            mode='markers',
            marker=dict(size=8, opacity=0.6, line=dict(width=1, color='black')),
            name=name,
            text=[f'Predicted: ${x:.1f}k<br>Residual: ${y:.1f}k' for x, y in zip(pred, residuals)],
            hoverinfo='text'
        ),
        row=row, col=col
    )
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=row, col=col)
    
    fig.update_xaxes(title_text="Predicted Price ($1000s)", row=row, col=col)
    fig.update_yaxes(title_text="Residuals ($1000s)", row=row, col=col)

fig.update_layout(
    height=400*n_rows,
    title_text="Residual Plots (Interactive - Hover for details)",
    template='plotly_white'
)

fig.write_image(f"{RESULTS_DIR}/residual_plots.png", width=1600, height=400*n_rows, scale=2)
fig.show()
print("Saved: residual_plots.png")

# ------------- Cell 5 -------------
# 3. RESIDUAL DISTRIBUTION (Interactive Plotly)
n_models = len(predictions)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"{name}<br>Mean: {np.mean(y_test - pred):.2f}, Std: {np.std(y_test - pred):.2f}" 
                    for name, pred in predictions.items()],
    vertical_spacing=0.12/n_rows
)

for idx, (name, pred) in enumerate(predictions.items()):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    residuals = y_test - pred
    
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=20,
            marker_color='steelblue',
            marker_line_color='black',
            marker_line_width=1,
            name=name,
            hovertemplate='Residual: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ),
        row=row, col=col
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=row, col=col)
    
    fig.update_xaxes(title_text="Residuals ($1000s)", row=row, col=col)
    fig.update_yaxes(title_text="Frequency", row=row, col=col)

fig.update_layout(
    height=400*n_rows,
    title_text="Residual Distribution (Interactive - Hover for details)",
    template='plotly_white',
    barmode='overlay'
)

fig.write_image(f"{RESULTS_DIR}/residual_distribution.png", width=1600, height=400*n_rows, scale=2)
fig.show()
print("Saved: residual_distribution.png")

# ------------- Cell 6 -------------
# 4. MODEL COMPARISON - R² Scores (Interactive Plotly)
model_names = list(metrics.keys())
train_r2 = [metrics[m]['train_r2'] for m in model_names]
test_r2 = [metrics[m]['test_r2'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig = go.Figure()

fig.add_trace(go.Bar(
    x=model_names,
    y=train_r2,
    name='Train R²',
    marker_color='steelblue',
    text=[f'{v:.3f}' for v in train_r2],
    textposition='auto',
    hovertemplate='Model: %{x}<br>Train R²: %{y:.3f}<extra></extra>'
))

fig.add_trace(go.Bar(
    x=model_names,
    y=test_r2,
    name='Test R²',
    marker_color='coral',
    text=[f'{v:.3f}' for v in test_r2],
    textposition='auto',
    hovertemplate='Model: %{x}<br>Test R²: %{y:.3f}<extra></extra>'
))

fig.update_layout(
    xaxis_title='Model',
    yaxis_title='R² Score',
    title='Model Comparison: Train vs Test R² (Interactive - Hover for values)',
    barmode='group',
    template='plotly_white',
    yaxis=dict(range=[0, 1])
)

fig.write_image(f'{RESULTS_DIR}/model_comparison_r2.png', width=1400, height=600, scale=2)
fig.show()
print("Saved: model_comparison_r2.png")

# ------------- Cell 7 -------------
# 5. MODEL COMPARISON - RMSE (Interactive Plotly)
train_rmse = [metrics[m]['train_rmse'] for m in model_names]
test_rmse = [metrics[m]['test_rmse'] for m in model_names]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=model_names,
    y=train_rmse,
    name='Train RMSE',
    marker_color='steelblue',
    text=[f'{v:.2f}' for v in train_rmse],
    textposition='auto',
    hovertemplate='Model: %{x}<br>Train RMSE: %{y:.2f}<extra></extra>'
))

fig.add_trace(go.Bar(
    x=model_names,
    y=test_rmse,
    name='Test RMSE',
    marker_color='coral',
    text=[f'{v:.2f}' for v in test_rmse],
    textposition='auto',
    hovertemplate='Model: %{x}<br>Test RMSE: %{y:.2f}<extra></extra>'
))

fig.update_layout(
    xaxis_title='Model',
    yaxis_title='RMSE ($1000s)',
    title='Model Comparison: Train vs Test RMSE (Interactive - Hover for values)',
    barmode='group',
    template='plotly_white'
)

fig.write_image(f'{RESULTS_DIR}/model_comparison_rmse.png', width=1400, height=600, scale=2)
fig.show()
print("Saved: model_comparison_rmse.png")

# ------------- Cell 8 -------------
# 6. FEATURE IMPORTANCE (Interactive Plotly)
lr_multi = models['Linear Multivariate']
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr_multi.coef_
}).sort_values('coefficient', key=abs, ascending=True)

colors = ['coral' if c < 0 else 'steelblue' for c in feature_importance['coefficient']]

fig = go.Figure(go.Bar(
    x=feature_importance['coefficient'],
    y=feature_importance['feature'],
    orientation='h',
    marker_color=colors,
    text=[f'{c:.3f}' for c in feature_importance['coefficient']],
    textposition='auto',
    hovertemplate='Feature: %{y}<br>Coefficient: %{x:.3f}<extra></extra>'
))

fig.update_layout(
    xaxis_title='Coefficient Value',
    yaxis_title='Feature',
    title='Feature Importance: Linear Regression Coefficients (Interactive - Hover for details)',
    template='plotly_white',
    height=500
)

fig.add_vline(x=0, line_color='black', line_width=1)

fig.write_image(f'{RESULTS_DIR}/feature_importance.png', width=1000, height=800, scale=2)
fig.show()
print("Saved: feature_importance.png")

# ------------- Cell 9 -------------
# 7. CORRELATION HEATMAP (Interactive Plotly)
# Load full dataset for correlation
import kagglehub
path = kagglehub.dataset_download("arunjangir245/boston-housing-dataset")
df = pd.read_csv(path + '/BostonHousing.csv')
df = df.fillna(df.median())

corr_matrix = df.corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdBu_r',
    zmid=0,
    text=np.round(corr_matrix.values, 2),
    texttemplate='%{text:.2f}',
    textfont={"size": 8},
    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
    colorbar_title='Correlation'
))

fig.update_layout(
    title='Feature Correlation Heatmap (Interactive - Hover for correlation values)',
    xaxis_title='Features',
    yaxis_title='Features',
    template='plotly_white',
    width=1200,
    height=1000
)

fig.write_image(f'{RESULTS_DIR}/correlation_heatmap.png', width=1400, height=1000, scale=2)
fig.show()
print("Saved: correlation_heatmap.png")

# ------------- Cell 10 -------------
# 8. TARGET VARIABLE DISTRIBUTION (Interactive Plotly)
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Training Set", "Test Set"]
)
# Training set
fig.add_trace(
    go.Histogram(
        x=y_train,
        nbinsx=30,
        marker_color='steelblue',
        marker_line_color='black',
        marker_line_width=1,
        name='Training Set',
        hovertemplate='Price: $%{x:.1f}k<br>Count: %{y}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_vline(x=y_train.mean(), line_dash="dash", line_color="red", row=1, col=1)
fig.add_vline(x=y_train.median(), line_dash="dot", line_color="green", row=1, col=1)

# Test set
fig.add_trace(
    go.Histogram(
        x=y_test,
        nbinsx=30,
        marker_color='coral',
        marker_line_color='black',
        marker_line_width=1,
        name='Test Set',
        hovertemplate='Price: $%{x:.1f}k<br>Count: %{y}<extra></extra>'
    ),
    row=1, col=2
)

fig.add_vline(x=y_test.mean(), line_dash="dash", line_color="red", row=1, col=2)
fig.add_vline(x=y_test.median(), line_dash="dot", line_color="green", row=1, col=2)

fig.update_xaxes(title_text="Price ($1000s)", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_xaxes(title_text="Price ($1000s)", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.update_layout(
    title_text="Target Distribution (Interactive - Hover for details)",
    template='plotly_white',
    height=500,
    showlegend=True
)

fig.add_annotation(x=y_train.mean(), y=1, xref='x', yref='paper', text=f"Mean: {y_train.mean():.2f}", showarrow=False, row=1, col=1)
fig.add_annotation(x=y_test.mean(), y=1, xref='x2', yref='paper', text=f"Mean: {y_test.mean():.2f}", showarrow=False, row=1, col=2)

fig.write_image(f'{RESULTS_DIR}/target_distribution.png', width=1400, height=500, scale=2)
fig.show()
print("Saved: target_distribution.png")

# ------------- Cell 11 -------------
# 9. CROSS-VALIDATION RESULTS (Interactive Plotly)
cv_results = json.load(open(f'{RESULTS_DIR}/cv_results.json'))

model_cv = list(cv_results.keys())
cv_means = [cv_results[m]['r2_mean'] for m in model_cv]
cv_stds = [cv_results[m]['r2_std'] for m in model_cv]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=model_cv,
    y=cv_means,
    error_y=dict(type='data', array=cv_stds, visible=True),
    marker_color='steelblue',
    text=[f'{v:.3f}' for v in cv_means],
    textposition='auto',
    hovertemplate='Model: %{x}<br>Mean CV R²: %{y:.3f}<br>Std: %{error_y.array:.3f}<extra></extra>'
))

fig.update_layout(
    xaxis_title='Model',
    yaxis_title='Mean CV R² Score',
    title='Cross-Validation Results (5-Fold) (Interactive - Hover for details)',
    template='plotly_white',
    yaxis=dict(range=[0, 1])
)

fig.write_image(f'{RESULTS_DIR}/cross_validation_results.png', width=1000, height=600, scale=2)
fig.show()
print("Saved: cross_validation_results.png")

# ------------- Cell 12 -------------
# 10. OVERFITTING VS UNDERFITTING ANALYSIS (Interactive Plotly)
train_scores = [metrics[m]['train_r2'] for m in model_names]
test_scores = [metrics[m]['test_r2'] for m in model_names]
cv_scores = [metrics[m]['cv_r2_mean'] for m in model_names]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=model_names,
    y=train_scores,
    name='Train R²',
    marker_color='steelblue',
    hovertemplate='Model: %{x}<br>Train R²: %{y:.3f}<extra></extra>'
))

fig.add_trace(go.Bar(
    x=model_names,
    y=test_scores,
    name='Test R²',
    marker_color='coral',
    hovertemplate='Model: %{x}<br>Test R²: %{y:.3f}<extra></extra>'
))

fig.add_trace(go.Bar(
    x=model_names,
    y=cv_scores,
    name='CV R²',
    marker_color='green',
    hovertemplate='Model: %{x}<br>CV R²: %{y:.3f}<extra></extra>'
))

fig.update_layout(
    xaxis_title='Model',
    yaxis_title='R² Score',
    title='Overfitting vs Underfitting Analysis (Interactive - Hover for values)',
    barmode='group',
    template='plotly_white',
    yaxis=dict(range=[0, 1])
)

fig.write_image(f'{RESULTS_DIR}/overfitting_analysis.png', width=1200, height=600, scale=2)
fig.show()
print("Saved: overfitting_analysis.png")

# ------------- Cell 13 -------------
# 11. BEST MODEL DETAILED ANALYSIS (Interactive Plotly)
best_model_name = 'Linear Multivariate'
best_pred = predictions[best_model_name]
best_metrics = metrics[best_model_name]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Actual vs Predicted', 'Residual Plot', 'Residual Distribution', 'Q-Q Plot'],
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "histogram"}, {"type": "scatter"}]]
)

residuals = y_test - best_pred

# Actual vs Predicted
fig.add_trace(
    go.Scatter(
        x=y_test,
        y=best_pred,
        mode='markers',
        marker=dict(size=8, opacity=0.6, line=dict(width=1, color='black')),
        name='Predictions',
        text=[f'Actual: ${x:.1f}k<br>Predicted: ${y:.1f}k' for x, y in zip(y_test, best_pred)],
        hoverinfo='text'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=1, col=1
)

# Residuals
fig.add_trace(
    go.Scatter(
        x=best_pred,
        y=residuals,
        mode='markers',
        marker=dict(size=8, opacity=0.6, line=dict(width=1, color='black')),
        name='Residuals',
        text=[f'Predicted: ${x:.1f}k<br>Residual: ${y:.1f}k' for x, y in zip(best_pred, residuals)],
        hoverinfo='text'
    ),
    row=1, col=2
)

fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

# Residual Distribution
fig.add_trace(
    go.Histogram(
        x=residuals,
        nbinsx=25,
        marker_color='steelblue',
        marker_line_color='black',
        name='Residuals',
        hovertemplate='Residual: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ),
    row=2, col=1
)

fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)

# Q-Q Plot
from scipy import stats
sorted_residuals = np.sort(residuals)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))

fig.add_trace(
    go.Scatter(
        x=theoretical_quantiles,
        y=sorted_residuals,
        mode='markers',
        marker=dict(size=8, opacity=0.6, line=dict(width=1, color='black')),
        name='Q-Q Points',
        text=[f'Theoretical: {x:.2f}<br>Sample: {y:.2f}' for x, y in zip(theoretical_quantiles, sorted_residuals)],
        hoverinfo='text'
    ),
    row=2, col=2
)

min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Normal Line',
        showlegend=False,
        hoverinfo='skip'
    ),
    row=2, col=2
)

fig.update_xaxes(title_text="Actual Price ($1000s)", row=1, col=1)
fig.update_yaxes(title_text="Predicted Price ($1000s)", row=1, col=1)
fig.update_xaxes(title_text="Predicted Price ($1000s)", row=1, col=2)
fig.update_yaxes(title_text="Residuals ($1000s)", row=1, col=2)
fig.update_xaxes(title_text="Residuals ($1000s)", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)
fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

fig.update_layout(
    title_text=f"Best Model Analysis: {best_model_name} (Interactive - Hover for details)",
    template='plotly_white',
    height=800,
    showlegend=True
)

fig.write_image(f'{RESULTS_DIR}/best_model_analysis.png', width=1400, height=1000, scale=2)
fig.show()
print("Saved: best_model_analysis.png")

