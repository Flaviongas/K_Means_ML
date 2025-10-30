import pandas as pd
from datetime import datetime
import io


class BaseModel:
    def __init__(
        self, file_route, generate_file=True,
        head=True,dshape=True, columns=True, info=True, describe=True,
        nulls=True, duplicated_sum=True, distribution=True, distribution_column=None, model_name="Dataset_Placeholder"
    ):
        self.df = pd.read_csv(file_route)
        self.generate_file = generate_file
        self.head = head
        self.dshape = dshape
        self.columns = columns
        self.info = info
        self.describe = describe
        self.nulls = nulls
        self.duplicated_sum = duplicated_sum
        self.distribution = distribution
        self.distribution_column = distribution_column
        self.model_name = model_name

    def _get_info_html(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue().replace('\n', '<br>')
        return f"<pre style='background:#f9fafb;padding:12px;border-radius:10px;border:1px solid #ddd;'>{info_str}</pre>"

    def explore(self):
        html = []
        html.append("""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {
                    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                    background-color: #fcfcfc;
                    margin: 40px auto;
                    max-width: 900px;
                    color: #222;
                    line-height: 1.6;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    font-size: 28px;
                    margin-bottom: 10px;
                }
                h2 {
                    color: #34495e;
                    border-left: 5px solid #2980b9;
                    padding-left: 10px;
                    margin-top: 40px;
                    margin-bottom: 10px;
                }
                p, ul {
                    font-size: 15px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-top: 10px;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f6fc;
                    font-weight: 600;
                }
                tr:nth-child(even) { background-color: #fafafa; }
                footer {
                    margin-top: 60px;
                    text-align: center;
                    font-size: 13px;
                    color: #777;
                }
            </style>
        </head>
        <body>
        """)
        html.append(f"<h1>Informe Exploratorio de  {self.model_name} </h1>")
        html.append(f"<p style='text-align:center'><em>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</em></p>")

        if self.head:
            html.append("<h2>Primeras filas del DataFrame</h2>")
            html.append(self.df.head().to_html(classes='table table-striped', border=0))

        if self.dshape:
            html.append("<h2>Dimensiones del DataFrame</h2>")
            html.append(f"<p>El DataFrame tiene <strong>{self.df.shape[0]}</strong> filas y <strong>{self.df.shape[1]}</strong> columnas.</p>")

        if self.columns:
            html.append("<h2>Columnas del DataFrame</h2>")
            html.append("<ul>" + "".join(f"<li>{c}</li>" for c in self.df.columns) + "</ul>")

        if self.info:
            html.append("<h2>Información del DataFrame</h2>")
            html.append(self._get_info_html())

        if self.describe:
            html.append("<h2>Estadísticas descriptivas</h2>")
            html.append(self.df.describe(include='all').to_html(classes='table', border=0))

        if self.nulls:
            html.append("<h2>Valores nulos por columna</h2>")
            nulls_df = self.df.isnull().sum().to_frame("Cantidad de Nulos")
            html.append(nulls_df.to_html(classes='table', border=0))

        if self.duplicated_sum:
            html.append("<h2>Cantidad de filas duplicadas</h2>")
            html.append(f"<p>{self.df.duplicated().sum()}</p>")

        if self.distribution:
            html.append("<h2>Distribución de valores</h2>")
            try:
                if self.distribution_column:
                    html.append(self.df[self.distribution_column].value_counts().to_frame("Frecuencia").to_html(classes='table', border=0))
                else:
                    html.append(self.df.value_counts().to_frame("Frecuencia").to_html(classes='table', border=0))
            except Exception:
                html.append("<p><em>No fue posible calcular la distribución (el DataFrame no es hasheable).</em></p>")

        report = "\n".join(html)

        if self.generate_file:
            file_name = f"html_reports/informe_datos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(report)
            print(f" Informe generado: {file_name}")

        return report

    def train(self):
        pass
