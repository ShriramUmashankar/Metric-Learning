import numpy as np 
import pandas as pd
from xgboost import XGBRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error

# Set seed for reproducibility
np.random.seed(42)

def main():
    # --- 1. Load Data ---
    print("--- 1. Loading Data ---")
    try:
        # Load augmented training features and scores
        X = np.load('full_features.npy')
        y = np.load('full_scores.npy').astype(float)
        X_test = np.load('test_features.npy')
    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        return

    # --- 2. Data Cleaning / Preparation ---
    # Convert 9.5 scores to 9.0 (as per the original logic)
    y[y == 9.5] = 9.0
    
    print(f"Training features shape: {X.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Display unique score counts (EDA)
    print("\n--- 2. Training Score Distribution ---")
    unique_vals, counts = np.unique(y, return_counts=True)
    print("="*50)
    for val, count in zip(unique_vals, counts):
        print(f"Number of samples with score {val:.1f}: {count}")
    print("="*50)

    # --- 3. Model Initialization and Training ---
    print("\n--- 3. Training Final XGBRegressor ---")
    
    # NOTE: The provided parameters (n_estimators=10) indicate a highly regularized/early stop model.
    # We use these exact parameters as requested.
    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=10, 
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )

    # Fit on the full augmented dataset
    # (sample_weight was commented out, so we assume weights are not used here)
    xgb.fit(
        X, y,
        verbose=True
    )

    # --- 4. Prediction and Post-Processing ---
    print("\n--- 4. Predicting Test Scores ---")
    y_pred_test = xgb.predict(X_test)
    
    # Clip and round predictions to the valid 0.0 - 10.0 range
    y_pred_test = np.clip(np.round(y_pred_test).astype(float), 0.0, 10.0)

    # --- 5. Final Output Statistics ---
    print("\n--- 5. Submission Statistics ---")
    
    # Check prediction distribution
    print("Prediction counts:")
    print(pd.Series(y_pred_test).value_counts().sort_index())
    
    # Calculate key statistics for the submission array
    print(f"Standard Deviation of predictions: {np.std(y_pred_test):.4f}")
    print(f"Mean of predictions: {np.mean(y_pred_test):.4f}")

    # --- 6. Create and Save Submission File ---
    submission = pd.DataFrame({
        "ID": np.arange(1, len(y_pred_test) + 1),
        "score": y_pred_test
    })

    # Saving to the file path requested
    filepath = "submission_file.csv"
    submission.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()