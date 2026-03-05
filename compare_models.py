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
