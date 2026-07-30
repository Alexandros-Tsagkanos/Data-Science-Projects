import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# datasets I want to compare on
# picked these mostly because they’re common and not too huge
datasets = ['breast-cancer', 'credit-g', 'adult']


for ds_name in datasets:
    print("\n==============================")
    print(f"Running experiments on: {ds_name}")
    print("==============================")

    # grab data from openml
    # parser='auto' avoids some annoying warnings in newer sklearn
    X, y = fetch_openml(
        ds_name,
        version=1,
        return_X_y=True,
        as_frame=True,
        parser='auto'
    )

    print("Original shape:", X.shape)

    # --- preprocessing ---
    # NOTE: this is intentionally very basic
    # goal here is quick comparisons, not a production pipeline

    # one-hot encode everything categorical
    # yeah, this can blow up dimensionality but it’s fine for now
    X = pd.get_dummies(X)

    # missing values:
    # TODO: try mean/median imputation and see if it matters
    X = X.fillna(0)

    # scale features (needed for SVM / MLP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # encode labels if needed
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # train / test split
    # no random_state on purpose — I want to see some variance
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2
    )

    # models to try
    # hyperparams are pretty arbitrary right now
    models = [
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('DT', DecisionTreeClassifier(max_depth=10)),
        ('RF', RandomForestClassifier(n_estimators=50)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(50,), max_iter=300)),
        ('SVM', SVC())
    ]

    results = []

    for model_name, model in models:
        print(f"\nTraining {model_name}...")

        # quick CV just to get a rough accuracy estimate
        # cv=3 to keep runtime reasonable
        cv_acc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring='accuracy'
        ).mean()

        # fit on full training set
        model.fit(X_train, y_train)

        # evaluate on test set
        preds = model.predict(X_test)

        # weighted metrics since some datasets are imbalanced
        prec = precision_score(
            y_test, preds, average='weighted', zero_division=0
        )
        rec = recall_score(
            y_test, preds, average='weighted', zero_division=0
        )
        f1 = f1_score(
            y_test, preds, average='weighted', zero_division=0
        )

        # just eyeballing confusion matrices for now
        cm = confusion_matrix(y_test, preds)
        # print(cm)  # uncomment if needed for debugging

        results.append({
            'Model': model_name,
            'CV_Accuracy': cv_acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })

    # summarize results for this dataset
    print("\nSummary:")
    results_df = pd.DataFrame(results)
    print(results_df.round(3))