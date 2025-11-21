
This project implements a regression pipeline to predict LLM response quality scores (0.0 - 10.0), focusing on overcoming severe class imbalance through **Negative Sampling-based data augmentation** 

### Prerequisites

You must have Python 3.8+ and the necessary libraries installed (`numpy`, `pandas`, `xgboost`, `scikit-learn`, `tqdm`, `sentence-transformers`, `torch`).

### 1\. Data Setup

Download the dataset and place all required files into a folder named `data/` at the root of the project directory.

The final folder structure should look like this:

```
.
├── data/
│   ├── train_data.json
│   ├── test_data.json
│   ├── sample_submission.csv
│   ├── metric_names.json
│   └── metric_name_embeddings.npy
├── convert_embeddings.py
├── generate_dataset.py
└── model_run.py
```

### 2\. Execution Pipeline

The solution runs in three sequential steps. Execute these files in order to generate the predictions.

#### Step 2a: Generate Initial Features

This step uses the `sentence-transformers` model to encode the `user_prompt` and `response` and concatenates them with the pre-computed `metric_name` embeddings.

```bash
python convert_embeddings.py
```

**Output Files Generated:**

  * `train_features.npy`
  * `train_scores.npy`
  * `test_features.npy`

#### Step 2b: Data Augmentation (Negative Sampling / MixUp)

This is the solution to the imbalance problem. This script generates synthetic training samples using negative sampling.

```bash
python generate_dataset.py
```

**Output Files Generated:**

  * `augmented_features.npy`
  * `augmented_scores.npy`
  * `full_features.npy` (Original + Augmented features)
  * `full_scores.npy` (Original + Augmented scores)

#### Step 2c: Train Model and Predict

This final step loads the full augmented dataset, trains the XGBoost Regressor, and generates the final prediction file for submission.

```bash
python model_run.py
```

**Final Output:**

  * `submissions/submission_file.csv`
