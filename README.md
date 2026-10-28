# Cyberbullying Detection System Using NLP

A beginner-friendly college mini project that classifies a text message into:
- `cyberbullying`
- `not_bullying`

It uses TF-IDF to convert text into numeric features and Logistic Regression to learn patterns from labeled examples. A Streamlit web page lets a user test messages.

## Project structure

```text
Cyberbullying_Detection_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── cyberbullying_dataset.csv
└── models/
    └── (trained model is created here)
```

## Requirements

- Python 3.9 or newer
- Internet connection only if pip needs to download packages
- VS Code (recommended)

## Run on Windows

Open the project folder in VS Code, then open Terminal and run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If `python` does not work, try `py` in place of `python`.

## Run on Ubuntu / Linux

Open a terminal in the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If Ubuntu says that `venv` is missing, install it with:

```bash
sudo apt update
sudo apt install python3-venv
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## How it works

1. The dataset contains text examples and their labels.
2. `train_model.py` reads the CSV file.
3. TF-IDF converts each message into word and two-word phrase features.
4. Logistic Regression learns patterns associated with each label.
5. The model is evaluated on a held-out test split.
6. The trained pipeline is saved in `models/cyberbullying_model.joblib`.
7. `app.py` loads the saved model and predicts the class of user-entered text.

## Test examples

Try:
- `Thank you for helping me with the assignment.`
- `You are useless and nobody wants you here.`
- `I disagree with your answer, but let's discuss it respectfully.`
- `Everyone should laugh at you and exclude you.`

The prediction can be wrong. This small sample dataset is for learning and demonstration only.

## Dataset note

The CSV is a small, hand-written educational sample. It is not a validated research dataset and should not be used for real moderation, disciplinary decisions, or safety decisions. A larger balanced dataset with expert-reviewed labels is needed for meaningful performance.

## Troubleshooting

- **Model not found:** run `python train_model.py` before starting Streamlit.
- **ModuleNotFoundError:** activate the virtual environment and run `pip install -r requirements.txt`.
- **Streamlit command not found:** use `python -m streamlit run app.py`.
- **Low or misleading accuracy:** this is expected with a tiny sample dataset. Add a larger, representative dataset and evaluate it carefully.
