import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = LOGS_DIR / "plots"
RESULTS_FILE = LOGS_DIR / "validation_results.json"
MODEL_STATUS_FILE = LOGS_DIR / "model_status.json"
DASHBOARD_FILE = BASE_DIR / "dashboard.html"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_metrics_table(report):
    rows = []

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue

        rows.append(f"""
        <tr>
            <td>{label}</td>
            <td>{metrics['precision']:.2f}</td>
            <td>{metrics['recall']:.2f}</td>
            <td>{metrics['f1-score']:.2f}</td>
            <td>{int(metrics['support'])}</td>
        </tr>
        """)

    return "\n".join(rows)


def build_errors_table(errors):
    if not errors:
        return """
        <tr>
            <td colspan="3">No se detectaron errores en este conjunto de prueba.</td>
        </tr>
        """

    rows = []
    for item in errors:
        rows.append(f"""
        <tr>
            <td>{item['text']}</td>
            <td>{item['real']}</td>
            <td>{item['predicted']}</td>
        </tr>
        """)

    return "\n".join(rows)


def generate_dashboard():
    results = load_json(RESULTS_FILE)
    model_status = load_json(MODEL_STATUS_FILE)

    accuracy = results["accuracy"]
    labels = results["labels"]
    total_examples = len(results["examples"])
    total_errors = len(results["errors"])
    total_correct = total_examples - total_errors

    metrics_table = build_metrics_table(results["classification_report"])
    errors_table = build_errors_table(results["errors"])

    status = model_status.get("status", "unknown")
    last_update = model_status.get("last_update", "No disponible")
    model_path = model_status.get("model_path", "No disponible")
    checkpoint_available = model_status.get("checkpoint_available", False)

    checkpoint_text = "Sí" if checkpoint_available else "No"

    status_color = "#16a34a" if status == "trained" else "#dc2626"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard del Modelo - Chatbot PLN</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}

        body {{
            background: #f4f6f9;
            color: #222;
            padding: 24px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: linear-gradient(135deg, #1e3a8a, #2563eb);
            color: white;
            padding: 28px;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.12);
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 8px;
        }}

        .header p {{
            opacity: 0.95;
            line-height: 1.5;
        }}

        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
            margin-bottom: 24px;
        }}

        .card {{
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        }}

        .card h3 {{
            font-size: 0.95rem;
            color: #666;
            margin-bottom: 10px;
        }}

        .card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #111827;
        }}

        .section {{
            background: white;
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        }}

        .section h2 {{
            margin-bottom: 16px;
            font-size: 1.35rem;
            color: #1f2937;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}

        .chart-box {{
            background: #fafafa;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
        }}

        .chart-box h3 {{
            margin-bottom: 12px;
            font-size: 1rem;
            color: #374151;
        }}

        .chart-box img {{
            width: 100%;
            height: auto;
            border-radius: 10px;
            display: block;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}

        th, td {{
            border: 1px solid #e5e7eb;
            padding: 12px;
            text-align: left;
            font-size: 0.95rem;
            vertical-align: top;
        }}

        th {{
            background: #eff6ff;
            color: #1e3a8a;
        }}

        .labels {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}

        .label-badge {{
            background: #dbeafe;
            color: #1d4ed8;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: bold;
        }}

        .footer-note {{
            margin-top: 12px;
            color: #6b7280;
            font-size: 0.92rem;
            line-height: 1.5;
        }}

        .status-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
        }}

        .status-item {{
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
        }}

        .status-item h3 {{
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 8px;
        }}

        .status-value {{
            font-size: 1rem;
            font-weight: bold;
            color: #111827;
            word-break: break-word;
        }}

        .badge-status {{
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            color: white;
            font-weight: bold;
            background: {status_color};
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 14px;
            }}

            .header h1 {{
                font-size: 1.5rem;
            }}

            .card .value {{
                font-size: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dashboard del Modelo de Chatbot PLN</h1>
            <p>
                Panel de resultados del modelo de clasificación de intenciones para una escuela de programación.
                Aquí se visualizan métricas clave, estado del modelo, gráficas de desempeño y errores detectados durante la evaluación.
            </p>
        </div>

        <div class="cards">
            <div class="card">
                <h3>Accuracy</h3>
                <div class="value">{accuracy:.2%}</div>
            </div>
            <div class="card">
                <h3>Total de ejemplos evaluados</h3>
                <div class="value">{total_examples}</div>
            </div>
            <div class="card">
                <h3>Predicciones correctas</h3>
                <div class="value">{total_correct}</div>
            </div>
            <div class="card">
                <h3>Errores detectados</h3>
                <div class="value">{total_errors}</div>
            </div>
        </div>

        <div class="section">
            <h2>Estado del modelo</h2>
            <div class="status-box">
                <div class="status-item">
                    <h3>Estado actual</h3>
                    <div class="status-value">
                        <span class="badge-status">{status}</span>
                    </div>
                </div>
                <div class="status-item">
                    <h3>Última actualización</h3>
                    <div class="status-value">{last_update}</div>
                </div>
                <div class="status-item">
                    <h3>Ruta del modelo</h3>
                    <div class="status-value">{model_path}</div>
                </div>
                <div class="status-item">
                    <h3>Checkpoint disponible</h3>
                    <div class="status-value">{checkpoint_text}</div>
                </div>
            </div>
            <p class="footer-note">
                Esta información proviene del archivo <strong>logs/model_status.json</strong> y permite verificar si el modelo fue entrenado correctamente y si existen puntos de control disponibles.
            </p>
        </div>

        <div class="section">
            <h2>Etiquetas del modelo</h2>
            <div class="labels">
                {''.join([f'<span class="label-badge">{label}</span>' for label in labels])}
            </div>
            <p class="footer-note">
                Estas son las intenciones que el modelo puede reconocer a partir del texto ingresado por el usuario.
            </p>
        </div>

        <div class="section">
            <h2>Gráficas de resultados</h2>
            <div class="grid-2">
                <div class="chart-box">
                    <h3>Accuracy del modelo</h3>
                    <img src="logs/plots/accuracy.png" alt="Accuracy del modelo">
                </div>
                <div class="chart-box">
                    <h3>Matriz de confusión</h3>
                    <img src="logs/plots/confusion_matrix.png" alt="Matriz de confusión">
                </div>
                <div class="chart-box">
                    <h3>Métricas por clase</h3>
                    <img src="logs/plots/metrics_by_class.png" alt="Métricas por clase">
                </div>
                <div class="chart-box">
                    <h3>Resumen de predicciones</h3>
                    <img src="logs/plots/prediction_summary.png" alt="Resumen de predicciones">
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Métricas por clase</h2>
            <table>
                <thead>
                    <tr>
                        <th>Clase</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics_table}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Errores del modelo</h2>
            <table>
                <thead>
                    <tr>
                        <th>Texto</th>
                        <th>Etiqueta real</th>
                        <th>Etiqueta predicha</th>
                    </tr>
                </thead>
                <tbody>
                    {errors_table}
                </tbody>
            </table>
            <p class="footer-note">
                Esta sección ayuda a responder preguntas como: qué errores comete el modelo, entre qué clases se confunde y cómo podría mejorarse.
            </p>
        </div>
    </div>
</body>
</html>
"""

    with open(DASHBOARD_FILE, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"Dashboard generado correctamente en: {DASHBOARD_FILE}")


if __name__ == "__main__":
    generate_dashboard()