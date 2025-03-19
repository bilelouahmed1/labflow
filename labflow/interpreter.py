import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X_train, X_test, feature_names=None):
    """
    Calcule les valeurs SHAP pour un modèle donné.

    Paramètres :
      - model : modèle entraîné (compatible avec SHAP, ex. scikit-learn, XGBoost, Keras, etc.)
      - X_train : données d'entraînement utilisées pour calibrer l'explainer
      - X_test : données sur lesquelles calculer les valeurs SHAP
      - feature_names : liste optionnelle de noms de colonnes

    Retourne :
      - shap_values : les valeurs SHAP calculées pour X_test
      - explainer : l'explainer SHAP utilisé
    """
    try:
        explainer = shap.Explainer(model, X_train)
    except Exception as e:
        raise ValueError(
            "Impossible de créer l'explainer SHAP pour ce modèle: " + str(e)
        )

    shap_values = explainer(X_test)
    return shap_values, explainer


def plot_shap_summary(shap_values, X_test, feature_names=None, show=True):
    """
    Affiche un summary plot des valeurs SHAP.

    Paramètres :
      - shap_values : valeurs SHAP calculées (résultat de compute_shap_values)
      - X_test : données correspondantes
      - feature_names : liste optionnelle des noms des caractéristiques
      - show : booléen, si True affiche le plot immédiatement (par défaut True)
    """
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=show)


def plot_shap_dependence(shap_values, X_test, feature, feature_names=None, show=True):
    """
    Affiche un dependence plot pour une feature donnée.

    Paramètres :
      - shap_values : valeurs SHAP calculées
      - X_test : données correspondantes
      - feature : nom (ou index) de la feature à analyser
      - feature_names : liste optionnelle des noms de colonnes
      - show : booléen, si True affiche le plot immédiatement (par défaut True)
    """
    shap.dependence_plot(
        feature, shap_values.values, X_test, feature_names=feature_names, show=show
    )
