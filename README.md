# Clothing-Color-Recommendation
Clothing Color Recommendation Data Mining Project

# Clothing Color Recommendation Data Mining Project

This project predicts suitable clothing color categories using physical features and style preference.

## Project Topic

**Clothing Color Recommendation Based on Physical Features Using Data Mining Techniques**

The previous rule-based idea was changed into a classification-based data mining project. The target variable is `Recommended Clothing Colors`.

## Dataset

The dataset is downloaded from Kaggle:

https://www.kaggle.com/datasets/suryaprabha19/fashion-and-color-recommendation-dataset

Each row contains several categorical fashion and physical-feature attributes. The project uses these input columns:

- `Hair Color`
- `Eye Color`
- `Skin Tone`
- `Under Tone`
- `Torso length`
- `Body Proportion`
- `Recommended Clothing Colors`

## Models

The project applies:

- Decision Tree Classifier
- Naive Bayes Classifier

## How to Run

Install the required Python libraries:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Download the Kaggle dataset:

```bash
.venv/bin/kaggle datasets download -d suryaprabha19/fashion-and-color-recommendation-dataset -p kaggle_data --unzip
```

Run the project:

```bash
.venv/bin/python color_recommendation_project.py
```

Generate the final report PDF:

```bash
.venv/bin/python generate_report_pdf.py
```

Generate the presentation:

```bash
.venv/bin/python create_presentation.py
```

The script creates:

- `recommended_color_distribution.png`
- `skin_tone_recommendation.png`
- `decision_tree.png`
- `model_results.csv`
