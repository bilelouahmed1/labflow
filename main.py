import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import labflow  # Assurez-vous que mlflow_alt.py est dans votre PYTHONPATH

# Chargement d'un jeu de données (ici iris pour un exemple de classification)
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Définir le projet
labflow.set_experiment("Pegasus")

# Lancer un run en fournissant les données d'évaluation (pour le calcul automatique des métriques)
with labflow.start_run(run_name="RegLog_2024", eval_data=(X_test, y_test)) as run:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Sauvegarder le modèle
    run.log_model(clf)
    run.log_parameter("max_iter", 1000)

    # Vous pouvez également loguer des artifacts additionnels (ex: graphiques, rapports)
    # run.log_artifact("chemin/vers/rapport.txt")

from labflow import compute_shap_values, plot_shap_summary
import shap
import pandas as pd

# Supposons que vous avez un modèle entraîné, X_train et X_test
shap_values, explainer = compute_shap_values(
    clf, X_train, X_test, feature_names=list(X_train.columns)
)
plot_shap_summary(shap_values, X_test, feature_names=list(X_train.columns))

labflow.generate_static_site()
