# Online Payment Fraud Detection using Machine Learning

A Python-based mini project that explores online transaction data and trains machine learning classifiers to identify potentially fraudulent transactions.

The project follows the general workflow described in the GeeksforGeeks tutorial, with additional data checks and evaluation metrics suited to an imbalanced fraud-detection dataset.

## Project Objectives

- Understand transaction data through descriptive statistics and exploratory data analysis (EDA).
- Examine transaction types, amounts, time steps, and the distribution of fraud labels.
- Prepare transaction features for supervised machine learning.
- Train and compare Logistic Regression, Random Forest, and XGBoost classifiers.
- Evaluate model performance using precision, recall, F1-score, ROC-AUC, average precision, and a confusion matrix.

## Dataset

The dataset used in the project is available through the source tutorial:

- **Tutorial:** [Online Payment Fraud Detection using Machine Learning in Python — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/online-payment-fraud-detection-using-machine-learning-in-python/)
- **Dataset download:** [Online Payment Data (`new_file.csv`)](https://drive.google.com/file/d/133E0TDrfIjnhwRoGTw9OEozwBXUL38D8/view?usp=sharing)

Download the CSV file and place it inside the local `dataset/` directory. The notebook expects exactly one CSV file in that directory. Keep the downloaded filename; the notebook discovers the file automatically.

The dataset includes these columns:

| Column | Description |
|---|---|
| `step` | Simulation time step of the transaction |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender account identifier |
| `oldbalanceOrg` | Sender balance before the transaction |
| `newbalanceOrig` | Sender balance after the transaction |
| `nameDest` | Receiver account identifier |
| `oldbalanceDest` | Receiver balance before the transaction |
| `newbalanceDest` | Receiver balance after the transaction |
| `isFraud` | Target label: `0` for legitimate and `1` for fraudulent transactions |

The full dataset can be large. The notebook uses the complete dataset for EDA, and by default selects a stratified sample of up to 500,000 records for model development to reduce training time. This setting can be changed in the notebook.

> **Dataset handling:** The CSV is excluded from Git by the project's `.gitignore` rules to avoid committing a large data file. Download it separately when setting up the project. Check the source's terms before redistributing the dataset.

## Technologies Used

- **Python 3.11**
- **Pandas and NumPy** — data loading, manipulation, and numerical operations
- **Matplotlib and Seaborn** — data visualization and EDA
- **Scikit-learn** — preprocessing, train/test splitting, classifiers, and evaluation metrics
- **XGBoost** — gradient-boosting classifier
- **Jupyter Notebook** — interactive analysis and model development
- **Conda** — environment management

## Project Structure

```text
online-payment-fraud-detection/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── dataset/
│   └── <downloaded-csv-file>
└── Online_Payment_Fraud_Detection.ipynb
```

## Methodology

1. **Load the data:** Find the CSV in `dataset/` and load it into a Pandas DataFrame.
2. **Understand the data:** Inspect dimensions, columns, data types, summary statistics, missing values, duplicates, transaction types, and class distribution.
3. **Perform EDA:** Visualize transaction counts, fraud rates by type, average amounts, transaction steps, amount distributions, and numerical-feature correlations.
4. **Validate and prepare features:** Check required columns and target labels. Exclude sender and receiver identifiers from model inputs and one-hot encode the `type` feature.
5. **Create a development sample if needed:** Use stratified sampling to preserve the approximate fraud proportion when limiting the number of rows used for model development.
6. **Split the data:** Use a stratified 70:30 training/testing split.
7. **Train classifiers:** Fit Logistic Regression, Random Forest, and XGBoost, with class-imbalance handling configured for each model.
8. **Evaluate and compare:** Calculate ROC-AUC, average precision, precision, recall, and F1-score; display a classification report and confusion matrix for the selected model.

## Installation and Setup

### Option A — Conda (recommended)

Install Anaconda or Miniconda, then open Anaconda Prompt or a terminal where `conda` is available.

1. Clone or download this repository.
2. Change to the project directory:

   ```powershell
   cd path\to\online-payment-fraud-detection
   ```

3. Create the environment from the YAML file:

   ```powershell
   conda env create -f environment.yml
   ```

4. Activate the environment:

   ```powershell
   conda activate online-payment-fraud-detection
   ```

5. Download the dataset using the link above and place the CSV in `dataset/`.

### Option B — pip with a virtual environment

If you are not using Conda, create a standard Python virtual environment and install the dependencies listed in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then download the CSV and put it in `dataset/`.

## Run the Project

### In VS Code

1. Open the project folder in VS Code.
2. Activate the Conda environment in the terminal:

   ```powershell
   conda activate online-payment-fraud-detection
   ```

3. Open `Online_Payment_Fraud_Detection.ipynb`.
4. Select the `online-payment-fraud-detection` Python kernel.
5. Run the notebook cells from top to bottom.

### Using Jupyter Notebook

With the Conda environment activated, run:

```powershell
jupyter notebook
```

Open `Online_Payment_Fraud_Detection.ipynb` in the browser and run the cells in order.

## Model Evaluation Results

Run the notebook to generate metrics for your dataset and model configuration. Add your actual values to this table after execution; scores are intentionally not pre-filled because results depend on the data, sample, software versions, and model settings.

| Model | ROC-AUC | Average Precision (PR-AUC) | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Pending | Pending | Pending | Pending | Pending |
| Random Forest | Pending | Pending | Pending | Pending | Pending |
| XGBoost | Pending | Pending | Pending | Pending | Pending |

The notebook also displays a confusion matrix for the selected model. In fraud detection, review **recall** (fraud cases detected) and **precision** (flagged cases that are truly fraud), not ROC-AUC alone. A high ROC-AUC score does not necessarily mean the model identifies enough fraud cases at the default probability threshold of `0.5`.

> **Evaluation note:** The current notebook compares models using metrics calculated on the test split and selects the leading model by test-set average precision. For a more rigorous final performance estimate, use validation data or cross-validation for model selection and threshold tuning, then evaluate the chosen model on an untouched test set.

## Limitations

- This is an educational mini project, not a production payment-monitoring system.
- Performance depends on the dataset and the chosen sampling and model settings.
- The default prediction threshold of `0.5` may not be appropriate for real-world fraud costs.
- The dataset's simulation time step and identifiers should not be interpreted as verified real-world customer information.
- A prediction indicates a pattern learned from this dataset; it does not prove that a transaction is fraudulent.

## Acknowledgement

This project is based on the learning workflow presented in the [GeeksforGeeks Online Payment Fraud Detection tutorial](https://www.geeksforgeeks.org/machine-learning/online-payment-fraud-detection-using-machine-learning-in-python/). The notebook has been organized for local execution and includes additional validation and evaluation steps.
