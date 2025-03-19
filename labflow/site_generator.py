import os
import json
from datetime import datetime

import plotly.graph_objs as go
import plotly.offline as offline


def generate_static_site(output_file="index.html"):
    BASE_DIR = "experiments"
    projects = {}

    # Parcours de tous les projets (chaque dossier dans BASE_DIR est un projet)
    for project in os.listdir(BASE_DIR):
        project_path = os.path.join(BASE_DIR, project)
        if os.path.isdir(project_path):
            runs = []
            # Chaque sous-dossier correspond à un run
            for run in os.listdir(project_path):
                run_path = os.path.join(project_path, run)
                log_file = os.path.join(run_path, "run_log.json")
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        run_log = json.load(f)
                    runs.append(run_log)
            if runs:
                projects[project] = runs

    sidebar_links = ""
    project_sections = ""
    run_details_sections = ""

    # Pour chaque projet, on construit le tableau et le graphique global
    for proj_name, runs in projects.items():
        # Tri des runs par date décroissante (la plus récente en premier)
        runs = sorted(runs, key=lambda r: r.get("start_time", ""), reverse=True)

        # Récupérer toutes les clés des hyperparamètres
        hp_keys = set()
        # Pour les métriques de test, on souhaite afficher "accuracy" et les sous-clés du rapport "macro avg"
        test_metric_keys = set()
        for run in runs:
            params = run.get("parameters", {})
            hp_keys.update(params.keys())
            test_metrics = run.get("test_metrics", {})
            if "accuracy" in test_metrics:
                test_metric_keys.add("accuracy")
            if "classification_report" in test_metrics:
                macro_avg = test_metrics["classification_report"].get("macro avg", {})
                test_metric_keys.update(macro_avg.keys())
        hp_keys = sorted(list(hp_keys))
        test_metric_keys = sorted(list(test_metric_keys))

        # Création des cases à cocher pour les colonnes du tableau
        hp_checkboxes = ""
        for key in hp_keys:
            hp_checkboxes += f'<label style="margin-right:10px;"><input type="checkbox" onchange="toggleColumn(\'col_hp_{key}\', this)" checked> {key}</label>'
        test_checkboxes = ""
        for key in test_metric_keys:
            test_checkboxes += f'<label style="margin-right:10px;"><input type="checkbox" onchange="toggleColumn(\'col_test_{key}\', this)" checked> {key}</label>'
        toggle_div = f"""
        <div style="margin-bottom:15px;">
            <div><strong>Hyperparamètres :</strong> {hp_checkboxes}</div>
            <div><strong>Métriques de test :</strong> {test_checkboxes}</div>
        </div>
        """

        # Préparation de la section "courbe globale"
        x_run_names = []
        y_values_dict = {key: [] for key in test_metric_keys}
        for run in runs:
            run_name = run.get("run_name", "unknown")
            x_run_names.append(run_name)
            test_metrics = run.get("test_metrics", {})
            for key in test_metric_keys:
                if key == "accuracy":
                    val = test_metrics.get("accuracy", None)
                else:
                    macro_avg = {}
                    if "classification_report" in test_metrics:
                        macro_avg = test_metrics["classification_report"].get(
                            "macro avg", {}
                        )
                    val = macro_avg.get(key, None)
                try:
                    y = float(val) if val is not None and val != "" else None
                except Exception:
                    y = None
                y_values_dict[key].append(y)

        overall_traces = []
        for key in test_metric_keys:
            trace = go.Scatter(
                x=x_run_names, y=y_values_dict[key], mode="lines+markers", name=key
            )
            overall_traces.append(trace)

        if overall_traces:
            fig_overall = go.Figure(data=overall_traces)
            fig_overall.update_layout(
                title=f"Métriques de test pour le projet '{proj_name}'",
                template="plotly_white",
            )
            # On encapsule la div générée dans un conteneur avec l'id souhaité
            overall_graph_div_content = offline.plot(
                fig_overall, include_plotlyjs=False, output_type="div"
            )
            overall_graph_div = (
                f'<div id="plotly-graph">{overall_graph_div_content}</div>'
            )
        else:
            overall_graph_div = "<p>Aucune donnée de test disponible.</p>"

        # Construction de l'en-tête du tableau :
        # Colonnes : Nom de l'expérience, Date, Type de modèle, hyperparamètres, test métriques, Modèle
        table_header = (
            "<tr><th>Nom de l'expérience</th><th>Date</th><th>Type de modèle</th>"
        )
        for key in hp_keys:
            table_header += f'<th class="hp-col col_hp_{key}">{key}</th>'
        for key in test_metric_keys:
            table_header += f'<th class="test-col col_test_{key}">{key}</th>'
        table_header += "<th>Modèle</th></tr>"

        table_rows = ""
        x_run_names = []
        y_accuracies = []
        for run in runs:
            run_name = run.get("run_name", "unknown")
            run_id = run.get("run_id", "unknown")
            # Récupération et reformatage de la date depuis "start_time"
            start_time = run.get("start_time", "N/A")
            try:
                dt = datetime.fromisoformat(start_time)
                start_time_formatted = dt.strftime("%d-%m-%Y %H:%M")
            except Exception:
                start_time_formatted = start_time

            clickable_name = f"<a href=\"#\" onclick=\"showRunDetail('{proj_name}', '{run_id}')\">{run_name}</a>"
            model_type = run.get("model_type", "unknown")
            row = f"<tr><td>{clickable_name}</td><td>{start_time_formatted}</td><td>{model_type}</td>"
            params = run.get("parameters", {})
            for key in hp_keys:
                value = params.get(key, "")
                row += f'<td class="hp-col col_hp_{key}">{value}</td>'
            test_metrics = run.get("test_metrics", {})
            acc_val = test_metrics.get("accuracy", "")
            row += f'<td class="test-col col_test_accuracy">{acc_val}</td>'
            macro_avg = {}
            if "classification_report" in test_metrics:
                macro_avg = test_metrics["classification_report"].get("macro avg", {})
            for key in test_metric_keys:
                if key == "accuracy":
                    continue
                val = macro_avg.get(key, "")
                row += f'<td class="test-col col_test_{key}">{val}</td>'
            model_path = run.get("artifacts", {}).get("model", "N/A")
            row += f"<td>{model_path}</td></tr>"
            table_rows += row

            # Pour le graphique global du projet, on collecte l'accuracy si disponible
            try:
                y_acc = float(acc_val) if acc_val != "" else None
            except:
                y_acc = None
            x_run_names.append(run_name)
            y_accuracies.append(y_acc)

            # Préparation de la vue détaillée pour ce run : courbe d'évolution des métriques d'entraînement
            train_evolution = run.get("train_metrics_evolution", {})
            training_curve_div = ""
            if (
                train_evolution
                and isinstance(train_evolution, dict)
                and len(train_evolution) > 0
            ):
                traces = []
                for metric, values in train_evolution.items():
                    epochs = [v.get("epoch") for v in values if "epoch" in v]
                    vals = []
                    for v in values:
                        try:
                            vals.append(float(v.get("value")))
                        except:
                            vals.append(v.get("value"))
                    trace = go.Scatter(
                        x=epochs, y=vals, mode="lines+markers", name=metric
                    )
                    traces.append(trace)
                if traces:
                    fig = go.Figure(data=traces)
                    fig.update_layout(
                        title="Courbe d'entraînement",
                        xaxis_title="Époch",
                        yaxis_title="Valeur",
                        template="plotly_white",
                    )
                    training_curve_div = offline.plot(
                        fig, include_plotlyjs=False, output_type="div"
                    )
            else:
                training_curve_div = (
                    "<p>Aucune métrique d'entraînement enregistrée.</p>"
                )

            run_detail_div = f"""
            <div class="run-detail" id="run_detail_{proj_name}_{run_id}" style="display:none; margin-bottom:30px;">
                <h3>Détails du run : {run_name}</h3>
                {training_curve_div}
                <button class="btn btn-secondary" onclick="backToProject('{proj_name}')">Retour</button>
            </div>
            """
            run_details_sections += run_detail_div

        table_html = f"""
        <table class="table table-bordered table-striped">
            <thead>{table_header}</thead>
            <tbody>{table_rows}</tbody>
        </table>
        """

        project_section = f"""
        <div class="project-section" id="project_{proj_name}" style="display:none; margin-bottom:30px;">
            <h2>{proj_name}</h2>
            <div id="project_main_{proj_name}">
                <div class="plotly-graph">
                    {overall_graph_div}
                </div>
                <div class="chart-separator"></div>
                <h3>Résumé des expériences</h3>
                {toggle_div}
                {table_html}
            </div>
        </div>
        """
        project_sections += project_section
        sidebar_links += f'<a href="#" class="list-group-item list-group-item-action" onclick="showProject(\'{proj_name}\')">{proj_name}</a>\n'

    html_template = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>LabFlow</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet" href="static/style.css">
    </head>
    <body>
        <div id="sidebar">
            <h4>Projets ouverts</h4>
            <div class="list-group">
                {sidebar_links}
            </div>
        </div>
        <div id="content">
            {project_sections}
            {run_details_sections}
        </div>
        <script>
            function showProject(projName) {{
                var projects = document.getElementsByClassName('project-section');
                for (var i = 0; i < projects.length; i++) {{
                    projects[i].style.display = 'none';
                }}
                var runDetails = document.getElementsByClassName('run-detail');
                for (var i = 0; i < runDetails.length; i++) {{
                    runDetails[i].style.display = 'none';
                }}
                var proj = document.getElementById('project_' + projName);
                if (proj) {{
                    proj.style.display = 'block';
                    var main = document.getElementById('project_main_' + projName);
                    if (main) {{
                        main.style.display = 'block';
                    }}
                }}
            }}
            function showRunDetail(projName, runId) {{
                var main = document.getElementById('project_main_' + projName);
                if (main) {{
                    main.style.display = 'none';
                }}
                var runDetails = document.getElementsByClassName('run-detail');
                for (var i = 0; i < runDetails.length; i++) {{
                    runDetails[i].style.display = 'none';
                }}
                var detail = document.getElementById('run_detail_' + projName + '_' + runId);
                if (detail) {{
                    detail.style.display = 'block';
                }}
            }}
            function backToProject(projName) {{
                var runDetails = document.getElementsByClassName('run-detail');
                for (var i = 0; i < runDetails.length; i++) {{
                    runDetails[i].style.display = 'none';
                }}
                var main = document.getElementById('project_main_' + projName);
                if (main) {{
                    main.style.display = 'block';
                }}
            }}
            function toggleColumn(colClass, checkbox) {{
                var elements = document.getElementsByClassName(colClass);
                var display = checkbox.checked ? '' : 'none';
                for (var i = 0; i < elements.length; i++) {{
                    elements[i].style.display = display;
                }}
            }}
            // Fonction pour basculer la visibilité d'une trace dans le graphique global
            function toggleGlobalTrace(projName, checkbox) {{
                var traceIndex = parseInt(checkbox.getAttribute('data-trace-index'));
                var chartId = 'global_chart_' + projName;
                var newVisibility = checkbox.checked ? true : 'legendonly';
                Plotly.restyle(chartId, {{'visible': newVisibility}}, [traceIndex]);
            }}
            window.onload = function() {{
                var links = document.getElementsByClassName('list-group-item');
                if (links.length > 0) {{
                    links[0].click();
                }}
            }};
        </script>
    </body>
    </html>
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Site statique généré dans {output_file}")


if __name__ == "__main__":
    generate_static_site()
