import os
import uuid
import json
import shutil
import pickle
from datetime import datetime

from contextlib import contextmanager
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Répertoire de base pour stocker les expériences
BASE_DIR = "experiments"
os.makedirs(BASE_DIR, exist_ok=True)

# Variable globale pour stocker le projet courant
CURRENT_EXPERIMENT = None


def set_experiment(experiment_name):
    """
    Définit le projet courant (similaire à mlflow.set_experiment).
    Crée un dossier dédié si inexistant.
    """
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT = experiment_name
    exp_dir = os.path.join(BASE_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Projet défini : {experiment_name}")


def detect_model_type(model):
    """
    Tente de détecter le type du modèle utilisé.
    Retourne une chaîne indiquant le type (ex. "xgboost", "random_forest",
    "logistic_regression", "neural_network_keras", "neural_network_pytorch") ou "unknown".
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor

        if isinstance(model, (XGBClassifier, XGBRegressor)):
            return "xgboost"
    except ImportError:
        pass
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            return "random_forest"
    except ImportError:
        pass
    try:
        from sklearn.linear_model import LogisticRegression

        if isinstance(model, LogisticRegression):
            return "logistic_regression"
    except ImportError:
        pass
    try:
        from tensorflow.keras.models import Model as KerasModel

        if isinstance(model, KerasModel):
            return "neural_network_keras"
    except ImportError:
        pass
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return "neural_network_pytorch"
    except ImportError:
        pass
    return "unknown"


@contextmanager
def start_run(run_name=None, eval_data=None):
    """
    Contexte de run, similaire à mlflow.start_run.

    Paramètres :
      - run_name (str) : Nom du run.
      - eval_data (tuple) : Optionnel, sous la forme (X_test, y_test) pour évaluation automatique
        (uniquement pour tâches de classification).

    Exemple d'utilisation :
        with start_run(run_name="Test_Run", eval_data=(X_test, y_test)) as run:
            run.log_parameter("lr", 0.01)
            for epoch in range(1, N+1):
                ...  # entraînement pour l'époque epoch
                run.log_train_metric("loss", current_loss, epoch)
                run.log_val_metric("accuracy", current_val_accuracy, epoch)
            run.log_final_metric("loss", final_loss)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            run.log_model(clf)
    """
    if CURRENT_EXPERIMENT is None:
        raise ValueError(
            "Aucun projet n'est défini. Utilisez set_experiment() d'abord."
        )

    # Création d'un identifiant unique et d'un dossier pour le run
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = run_name if run_name else f"run_{timestamp}"
    run_folder = os.path.join(
        BASE_DIR, CURRENT_EXPERIMENT, f"{run_folder_name}_{run_id}"
    )
    os.makedirs(run_folder, exist_ok=True)

    # Dictionnaire de log initial, avec un champ "model_type"
    run_log = {
        "run_id": run_id,
        "run_name": run_folder_name,
        "start_time": datetime.now().isoformat(),
        "parameters": {},
        "final_metrics": {},  # métriques finales d'entraînement
        "test_metrics": {},  # métriques de test (calculées en fin de run si eval_data est fourni)
        "train_metrics_evolution": {},  # métriques d'entraînement (par époque)
        "val_metrics_evolution": {},  # métriques de validation (par époque)
        "artifacts": {},
        "model_type": "unknown",  # Champ qui sera mis à jour lors du logging du modèle
    }

    # Sauvegarde initiale dans un fichier JSON
    run_log_path = os.path.join(run_folder, "run_log.json")
    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    # Fournir une instance de RunLogger pour loguer durant le run
    logger = RunLogger(run_folder, run_log)

    try:
        yield logger
    finally:
        # Fin du run : enregistrer l'heure de fin
        run_log["end_time"] = datetime.now().isoformat()
        # Évaluation sur le jeu de test (si applicable)
        if eval_data and "model" in run_log["artifacts"]:
            model_path = run_log["artifacts"]["model"]
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            X_test, y_test = eval_data
            y_pred = model.predict(X_test)
            run_log["test_metrics"]["accuracy"] = accuracy_score(y_test, y_pred)
            run_log["test_metrics"]["confusion_matrix"] = confusion_matrix(
                y_test, y_pred
            ).tolist()
            run_log["test_metrics"]["classification_report"] = classification_report(
                y_test, y_pred, output_dict=True
            )
        # Sauvegarde finale du log
        with open(run_log_path, "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=4)
        print(f"Run sauvegardé dans : {run_folder}")


class RunLogger:
    """
    Classe permettant de loguer les paramètres, métriques, artifacts et modèles.
    """

    def __init__(self, run_folder, run_log):
        self.run_folder = run_folder
        self.run_log = run_log

    def log_parameter(self, key, value):
        self.run_log["parameters"][key] = value
        self._update_log()

    def log_train_metric(self, key, value, epoch):
        """
        Logue une métrique d'entraînement pour une époque donnée.
        """
        if key not in self.run_log["train_metrics_evolution"]:
            self.run_log["train_metrics_evolution"][key] = []
        self.run_log["train_metrics_evolution"][key].append(
            {"epoch": epoch, "value": value}
        )
        self._update_log()

    def log_val_metric(self, key, value, epoch):
        """
        Logue une métrique de validation pour une époque donnée.
        """
        if key not in self.run_log["val_metrics_evolution"]:
            self.run_log["val_metrics_evolution"][key] = []
        self.run_log["val_metrics_evolution"][key].append(
            {"epoch": epoch, "value": value}
        )
        self._update_log()

    def log_final_metric(self, key, value):
        """
        Enregistre une métrique finale d'entraînement.
        """
        self.run_log["final_metrics"][key] = value
        self._update_log()

    def log_artifact(self, file_path, artifact_name=None):
        if artifact_name is None:
            artifact_name = os.path.basename(file_path)
        dest_path = os.path.join(self.run_folder, artifact_name)
        shutil.copy(file_path, dest_path)
        self.run_log["artifacts"][artifact_name] = dest_path
        self._update_log()

    def log_model(self, model, model_name="model.pkl"):
        """
        Sauvegarde le modèle en utilisant pickle, puis le logue comme artifact.
        Détecte également le type de modèle utilisé et met à jour le champ "model_type" dans le run_log.json.
        """
        model_type = detect_model_type(model)
        model_path = os.path.join(self.run_folder, model_name)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        self.run_log["artifacts"]["model"] = model_path
        self.run_log["model_type"] = model_type
        self._update_log()

    def _update_log(self):
        run_log_path = os.path.join(self.run_folder, "run_log.json")
        with open(run_log_path, "w", encoding="utf-8") as f:
            json.dump(self.run_log, f, indent=4)

    def log_shap(self, shap_summary_path, shap_dependence_path=None):
        """
        Enregistre les chemins des graphiques SHAP dans le log du run.
        """
        self.run_log["artifacts"]["shap_summary"] = shap_summary_path
        if shap_dependence_path:
            self.run_log["artifacts"]["shap_dependence"] = shap_dependence_path
        self._update_log()
