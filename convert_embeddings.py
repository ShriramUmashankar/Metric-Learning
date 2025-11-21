import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from typing import List, Dict, Any, Tuple

# --- Configuration ---
DATASET_PATH = 'data/' # This assumes 'data' is the directory path
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# --- 1. Setup and Loading ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize the model
try:
    model = SentenceTransformer(MODEL_NAME).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit if model fails to load
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device) 
    print("Using fallback model.")


# --- Data Loading ---
def load_data(path: str) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    """Loads training data, metric names, and pre-computed metric embeddings."""
    with open(f"{path}/train_data.json", "r") as f:
        train_data = json.load(f)
    with open(f"{path}/metric_names.json") as f:
        metric_names = json.load(f)
    metric_name_embeddings = np.load(f"{path}/metric_name_embeddings.npy")
    
    print(f"\nLoaded {len(train_data)} training samples.")
    print(f"Loaded {len(metric_names)} metric names.")
    print(f"Metric embedding shape: {metric_name_embeddings.shape}")
    
    # Create the mapping dictionary
    metric_embedding_map = {
        name: embedding
        for name, embedding in zip(metric_names, metric_name_embeddings)
    }
    return train_data, metric_embedding_map

# Load training data
train_data, metric_embedding_map = load_data(DATASET_PATH)

# --- 2. Encoding Function ---
def encode_data(data: List[Dict[str, Any]], metric_map: Dict[str, np.ndarray], model: SentenceTransformer) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes the user_prompt and response into a single vector and concatenates it with the metric embedding."""
    X: List[np.ndarray] = []
    y: List[float] = []
    
    # Check if the data has 'score' key (for training data)
    is_training = 'score' in data[0]

    for record in tqdm(data, desc="Encoding samples"):
        metric_name = record["metric_name"]
        
        # --- 1. Text Preparation ---
        prompt = record.get("user_prompt") or ""
        response = record.get("response") or ""
        # The original code ignores system_prompt, maintaining that logic:
        # system_prompt = record.get("system_prompt") 

        # Combine text for encoding
        combined_text = prompt + " [SEP] " + response

        # --- 2. Embedding Generation (S-BERT) ---
        # Get a single embedding vector for the prompt-response pair
        pair_emb = model.encode(combined_text)

        # --- 3. Concatenation ---
        # Get the pre-computed metric embedding
        metric_emb = metric_map[metric_name]

        # Combine the metric embedding and the pair embedding
        combined_emb = np.concatenate([metric_emb, pair_emb])

        X.append(combined_emb)
        
        if is_training:
            y.append(float(record["score"]))

    print("\nEncoding complete.")
    
    X_out = np.array(X)
    Y_out = np.array(y) if is_training else None

    print(f"Shape of X_out: {X_out.shape}")
    if is_training:
        print(f"Shape of y_out: {Y_out.shape}")

    return X_out, Y_out

# --- 3. Execution ---

# 3a. Encode Training Data
X_train, y_train = encode_data(train_data, metric_embedding_map, model)
np.save("train_features.npy", X_train)
np.save("train_scores.npy", y_train)
print("Saved train_features.npy and train_scores.npy")


# 3b. Encode Testing Data
with open(f"{DATASET_PATH}/test_data.json", "r") as f:
    test_data = json.load(f)
print(f"\nLoaded {len(test_data)} test samples.")

X_test, _ = encode_data(test_data, metric_embedding_map, model)
np.save("test_features.npy", X_test)
print("Saved test_features.npy")