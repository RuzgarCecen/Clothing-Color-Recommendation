import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


DATASET_PATH = "kaggle_data/recommendations.csv"
FEATURE_COLUMNS = [
    "Hair Color",
    "Eye Color",
    "Skin Tone",
    "Under Tone",
    "Torso length",
    "Body Proportion",
]
TARGET_COLUMN = "Recommended Clothing Colors"


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def explore_data(df):
    print("First 5 rows:")
    print(df.head())
    print("\nDataset shape:")
    print(df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())


def save_visualizations(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y=TARGET_COLUMN, order=df[TARGET_COLUMN].value_counts().index)
    plt.title("Recommended Color Distribution")
    plt.tight_layout()
    plt.savefig("recommended_color_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Skin Tone", hue=TARGET_COLUMN)
    plt.title("Skin Tone and Recommended Color")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("skin_tone_recommendation.png", dpi=200)
    plt.close()


def build_models():
    decision_tree = Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS)
                    ]
                ),
            ),
            ("model", DecisionTreeClassifier(random_state=42, max_depth=4)),
        ]
    )

    naive_bayes = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("model", CategoricalNB()),
        ]
    )

    return {
        "Decision Tree": decision_tree,
        "Naive Bayes": naive_bayes,
    }


def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results = []
    models = build_models()

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(f"Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions, labels=sorted(y.unique())))

        results.append({"model": model_name, "accuracy": accuracy})

    pd.DataFrame(results).to_csv("model_results.csv", index=False)
    return models


def save_decision_tree_visualization(models, df):
    tree_pipeline = models["Decision Tree"]
    encoder = tree_pipeline.named_steps["encoder"]
    model = tree_pipeline.named_steps["model"]
    feature_names = encoder.get_feature_names_out()

    plt.figure(figsize=(18, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=sorted(df[TARGET_COLUMN].unique()),
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=200)
    plt.close()


def recommend_color(models, user_input):
    prediction = models["Decision Tree"].predict(pd.DataFrame([user_input]))[0]
    return prediction


def main():
    df = load_dataset()
    explore_data(df)
    save_visualizations(df)
    models = train_and_evaluate(df)
    save_decision_tree_visualization(models, df)

    example_user = {
        "Hair Color": "Brown",
        "Eye Color": "Green",
        "Skin Tone": "Medium",
        "Under Tone": "Warm",
        "Torso length": "Balanced",
        "Body Proportion": "Hourglass",
    }
    recommendation = recommend_color(models, example_user)

    print("\nExample Recommendation")
    print("----------------------")
    print(example_user)
    print(f"Recommended clothing color category: {recommendation}")
    print("\nGenerated files:")
    print("- recommended_color_distribution.png")
    print("- skin_tone_recommendation.png")
    print("- decision_tree.png")
    print("- model_results.csv")


if __name__ == "__main__":
    main()
