from .BaseModel import BaseModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


class ClassificationModel(BaseModel):

    def __init__(self, file_route, generate_file, head, dshape, columns,
                 info, describe, nulls, duplicated_sum, distribution, distribution_column, dataset_name):

        super().__init__(file_route, generate_file, head, dshape,
                         columns, info, describe, nulls, duplicated_sum, distribution, distribution_column, dataset_name)

    def cleanup(self):
        print("Eliminando columnas irrelevantes...")
        self.df = self.df.drop(columns=['obj_ID', 'spec_obj_ID', 'field_ID'])

    def box_plots(self):
        new_df = self.df.drop(columns='class')
        for col in new_df.columns:
            new_df.boxplot(column=col)
            plt.title(col)
            plt.savefig(f'artifacts/classification/boxplot_{col}.png')

    def manejo_outliers(self):
        new_df = self.df.copy()

        for col in new_df.columns:

            if not pd.api.types.is_numeric_dtype(new_df[col]):
                continue

            limite_inferior = new_df[col].quantile(0.0005)   # 0.05%
            limite_superior = new_df[col].quantile(0.9995)   # 99.95%

            new_df = new_df[
                (new_df[col] >= limite_inferior) &
                (new_df[col] <= limite_superior)
            ]

        self.df = new_df

    def heatmap(self):
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.savefig('artifacts/classification/heatmap_correlation.png')

    def split_data(self, test_size=0.25, random_state=21):

        X = self.df.drop(columns='class')
        y = self.df['class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

    def scale_data(self):
        self.X_scaled = StandardScaler().fit_transform(self.X_train)

    def PCA(self):
        pca = PCA(n_components=0.95)
        pca.fit(self.X_scaled)
        self.explained_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.explained_variance)+1), self.explained_variance, marker='o')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Método del Codo (Elbow Method) - PCA')
        plt.grid(True)
        plt.savefig('artifacts/classification/pca_elbow_method.png')

    def pipeline_log_exec(self):
        self.pipeline_log = Pipeline([
            ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=0.95)),
            ('rlg', LogisticRegression(max_iter=160))
        ])
        self.pipeline_log.fit(self.X_train, self.y_train)
        self.log_y_pred = self.pipeline_log.predict(self.X_test)
        print(f" Acc={accuracy_score(self.y_test, self.log_y_pred):.3f}  F1={f1_score(self.y_test, self.log_y_pred, average='weighted'):.3f}")

    def PCA_plot(self):
        Z_train = self.pipeline_log.named_steps['pca'].transform(
               self.pipeline_log.named_steps['scaler'].transform(self.X_train)
        )

        plt.figure(figsize=(6, 6))
        class_colors = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
        y_encoded = self.y_train.map(class_colors)

        plt.figure(figsize=(12, 8))
        plt.scatter(Z_train[:, 0], Z_train[:, 1],
                    c=y_encoded, cmap='viridis',
                    s=20, alpha=0.6, edgecolor='k', linewidth=0.5)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.xlabel(f'PC1 ({self.pipeline_log.named_steps["pca"].explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({self.pipeline_log.named_steps["pca"].explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('PCA - Clasificación de Objetos Astronómicos')
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=plt.cm.viridis(class_colors[c]/2),
                   markersize=10, label=c)
                   for c in class_colors.keys()]
        plt.legend(handles=handles, title='Clase')
        plt.grid(True)
        plt.savefig('artifacts/classification/pca_scatter_plot.png')

    def pipeline_rf_exec(self):
        self.pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=0.95)),
            ('rf', RandomForestClassifier())
        ])
        self.pipeline_rf.fit(self.X_train, self.y_train)
        self.self.rf_y_pred = self.pipeline_rf.predict(self.X_test)
        print(f" Acc={accuracy_score(self.y_test, self.rf_y_pred):.3f}  F1={f1_score(self.y_test, self.rf_y_pred, average='weighted'):.3f}")

    def train(self):
        print("Limpiando datos...")
        self.cleanup()
        print("Generando gráficos de caja..")
        self.box_plots()
        print("Eliminando outliers...")
        self.manejo_outliers()
        print("Generando mapa de calor de correlaciones...")
        self.heatmap()
        print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        self.split_data()
        print("Estandarizando datos...")
        self.scale_data()
        print("Realizando análisis de componentes principales (PCA)...")
        print("Generando gráfico del método del codo...")
        self.PCA()
        print("Entrenando modelo de Regresión Logística...")
        self.pipeline_log_exec()
        print("Generando gráfico de dispersión PCA...")
        self.PCA_plot()
        print("Entrenando modelo de Random Forest...")
        self.pipeline_rf_exec()

    def tune(self):
        param_distributions = {
            'n_estimators': np.arange(),
            # 'max_depth': [None, , , ],
            # 'min_samples_split': [, , ],
            # 'min_samples_leaf': [, , ],
            'bootstrap': [True, False]
        }
        random_search = RandomizedSearchCV(
            estimator=self.pipeline_rf,
            param_distributions=param_distributions,
            n_iter=50,
            scoring='accuracy',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        print("NO TERMINADO")
        raise NotImplementedError("Tuning no está implementado aún.")
