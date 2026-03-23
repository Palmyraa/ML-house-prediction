import pandas as pd
import json
from pathlib import Path

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found. Have you trained the models yet?")
        return

    metrics_data = []
    
    # Load all metrics files
    for filepath in results_dir.glob("metrics_*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Extract key metrics
            model_name = data.get('model_name', filepath.stem.replace('metrics_', ''))
            train_r2 = data.get('train_r2', float('nan'))
            test_r2 = data.get('test_r2', float('nan'))
            cv_r2_mean = data.get('cv_r2_mean', float('nan'))
            train_rmse = data.get('train_rmse', float('nan'))
            test_rmse = data.get('test_rmse', float('nan'))
            
            metrics_data.append({
                'Model Name': model_name,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'CV R² Mean': cv_r2_mean,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse
            })
            
    if not metrics_data:
        print("No metrics JSON files found in the results directory.")
        return

    df = pd.DataFrame(metrics_data)
    
    # Sort by Test R² descending (higher is better)
    df = df.sort_values(by='Test R²', ascending=False).reset_index(drop=True)
    
    # Print the comparison table
    print("="*95)
    print(" "*35 + "MODEL COMPARISON OVERVIEW")
    print("="*95)
    
    # Format the dataframe for pretty printing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    # Rename column layout for pretty print
    print(df.to_string())
    print("-" * 95)
    
    # Determine the best model
    best_model_row = df.iloc[0]
    
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print(f"\n🏆 THE BEST MODEL IS:")
    print(f"👉 Name:         {best_model_row['Model Name']}")
    print(f"👉 Test R²:      {best_model_row['Test R²']:.4f}")
    if pd.notna(best_model_row['CV R² Mean']):
        print(f"👉 CV R² Mean:   {best_model_row['CV R² Mean']:.4f}")
    print(f"👉 Test RMSE:    {best_model_row['Test RMSE']:.4f}")
    print("=" * 95)

if __name__ == "__main__":
    main()

#===============================================================================================
#              MODEL COMPARISON OVERVIEW
#===============================================================================================
#                              Model Name  Train R²  Test R²  CV R² Mean  Train RMSE  Test RMSE
#0               Decision Tree Regression    0.9277   0.8495      0.7239      2.5206     3.3487
#1              Neural Network Regression    0.8456   0.8058      0.7853      3.6842     3.8044
#2                         SGD (adaptive)    0.7421   0.7102      0.6901      4.7609     4.6466
#3       Linear Regression (Multivariate)    0.7432   0.7099      0.6880      4.7508     4.6496
#4                         SGD (constant)    0.7347   0.6940      0.6657      4.8291     4.7747
#5  Linear Regression (Feature Selection)    0.6873   0.6511      0.6512      5.2428     5.0990
#6       Polynomial Regression (degree=3)    0.5491   0.5825      0.4908      6.2955     5.5773
#7       Polynomial Regression (degree=2)    0.5362   0.5672      0.4829      6.3851     5.6791
#8         Linear Regression (Univariate)    0.4887   0.4580      0.4524      6.7039     6.3550
#---------------------------------------------------------------------------------------------

#THE BEST MODEL :
 #Name:         Decision Tree Regression
 #Test R²:      0.8495
#CV R² Mean:   0.7239
 #Test RMSE:    3.3487
#===============================================================================================