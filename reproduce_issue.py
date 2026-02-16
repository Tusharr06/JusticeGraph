import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append('/home/Sohan/codes/Practice/justice_graph')

from justice_graph.models.backlog_predictor import BacklogPredictor

def test_model():
    # Path to model
    model_path = '/home/Sohan/codes/Practice/justice_graph/models/backlog_model.pkl'
    
    print(f"Loading model from {model_path}...")
    try:
        predictor = BacklogPredictor(model_path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy input based on user's request
    attributes = {
        'judge_strength': [5],
        'pending_cases': [22000],
        'filing_rate': [700],
        'disposal_rate': [200],
        'budget_per_capita': [80],
        'courthall_shortfall': [40]
    }
    features = pd.DataFrame(attributes)
    
    print("\n--- Prediction Test (Original) ---")
    print("Input Features:", features.to_dict(orient='records')[0])
    
    try:
        prob = predictor.predict(features)[0]
        print(f"Prediction Probability: {prob}")
        
        explanation = predictor.explain(features)
        print("Explanation (SHAP values):", explanation)
        
    except Exception as e:
        print(f"Prediction failed: {e}")

    # Test shifting values
    print("\n--- Sensitivity Test ---")
    # Change courthall_shortfall significantly
    features_modified = features.copy()
    features_modified['courthall_shortfall'] = [0]
    
    print("Modified Input (shortfall=0):", features_modified.to_dict(orient='records')[0])
    try:
        prob_mod = predictor.predict(features_modified)[0]
        print(f"Prediction Probability (Modified): {prob_mod}")
        explanation_mod = predictor.explain(features_modified)
        print("Explanation (Modified):", explanation_mod)
    except Exception as e:
        print(f"modified prediction failed: {e}")

if __name__ == "__main__":
    test_model()
