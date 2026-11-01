# Fake News Detection Using NLP

A course-project prototype that classifies news text as **FAKE** or **REAL** using TF-IDF features and three machine-learning classifiers.

## Important limitation

This is a text-pattern classifier trained on a particular labelled dataset. It does **not** fact-check claims, consult authoritative sources, or prove that an article is true or false. Dataset bias, source leakage, topic changes and writing-style differences can affect predictions. Present the output as a model classification, not a verified verdict.

## Project features

- Text cleaning and normalization
- TF-IDF unigram and bigram feature extraction
- Logistic Regression, Multinomial Naive Bayes and calibrated Linear SVM
- Stratified train/test split
- Accuracy, precision, recall and F1-score comparison
- Confusion matrix and model comparison chart
- Streamlit web interface for entering news text

## 1. Requirements

- Windows 10/11
- Python 3.10 or newer (Python 3.11 is a good choice)
- VS Code with the Python extension

Check Python:

```powershell
python --version
```

## 2. Download the dataset

Download the **Fake and Real News Dataset** from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Extract the download. Copy these two files into this project's `dataset` folder:

```text
dataset/
    Fake.csv
    True.csv
```

Do not rename the files. The dataset is not included in this project ZIP; download it from Kaggle and follow its licence/terms.

## 3. Create and activate a virtual environment

Open this project folder in VS Code. Select **Terminal > New Terminal**, then run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt in VS Code and run:

```cmd
venv\Scripts\activate.bat
```

## 4. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Train the models

Make sure both CSV files are inside `dataset/`, then run:

```powershell
python train_model.py
```

The script cleans and deduplicates the data, trains three models, evaluates them on a stratified held-out test set, chooses the model with the highest F1-score, and saves:

```text
models/fake_news_pipeline.joblib
results/metrics.json
results/confusion_matrix.png
results/model_comparison.png
```

Training may take a few minutes depending on your laptop.

## 6. Run the application

```powershell
streamlit run app.py
```

Streamlit will open the application in your browser. If it does not, open the local URL printed in the terminal (usually http://localhost:8501).

## Folder structure

```text
Fake-News-Detection/
├── app.py
├── train_model.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   └── (put Fake.csv and True.csv here)
├── models/
│   └── (trained model is generated here)
├── notebooks/
│   └── README.md
└── results/
    └── (metrics and charts are generated here)
```

## Methodology

1. Load fake and real article datasets.
2. Combine headline and body text when a title field exists.
3. Normalize text and remove duplicate article text.
4. Split data into 80% training and 20% testing using stratification.
5. Convert text to TF-IDF unigram and bigram features.
6. Train Logistic Regression, Multinomial Naive Bayes and calibrated Linear SVM.
7. Compare accuracy, precision, recall and F1-score.
8. Select the model with the highest test F1-score and save the full pipeline.

## Research and evaluation cautions

- Do not report an accuracy value until you have run the experiment and read `results/metrics.json`.
- Keep the test set separate from training and model selection.
- The commonly used dataset may contain publisher, topic, time-period or writing-style signals. Random splitting can therefore overestimate generalization to new sources or future news.
- For a stronger study, add source-aware or time-based evaluation if the dataset supports it, inspect false positives/negatives, and test on an independently collected dataset.
- Cite the dataset and all research papers you actually read in the IEEE paper. Do not invent references.

## Troubleshooting

**`python` is not recognized:** Install Python from https://www.python.org/downloads/ and enable "Add Python to PATH", then reopen VS Code.

**Dataset file not found:** Check that the filenames are exactly `Fake.csv` and `True.csv` and that they are in the `dataset` folder.

**Module not found:** Activate the virtual environment and run `pip install -r requirements.txt` again.

**PowerShell activation error:** Use the Command Prompt activation command shown above.

**Port already in use:** Run `streamlit run app.py --server.port 8502`.
