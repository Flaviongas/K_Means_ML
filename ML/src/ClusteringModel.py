import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .KmeansPipelineEvaluator import KMeansPipelineEvaluator
from .KMeansAnalyzer import KMeansAnalyzer
from .BaseModel import BaseModel


class ClusteringModel(BaseModel):
    def __init__(self, file_route, generate_file=True,
                 head=True, dshape=True, columns=True, info=True, describe=True,
                 nulls=True, duplicated_sum=True, distribution=True,
                 distribution_column=None, dataset_name="Dataset_Placeholder",
                 random_state=42, k_value=5):

        super().__init__(file_route, generate_file, head, dshape,
                         columns, info, describe, nulls, duplicated_sum,
                         distribution, distribution_column, dataset_name)

        self.random_state = random_state
        self.k_value = k_value
        self.output_dir = os.path.join("artifacts/clustering", "")
        if self.generate_file:
            os.makedirs(self.output_dir, exist_ok=True)

    def preprocess(self):
        print("Seleccionando columnas estadísticas del Pokémon...")
        columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        Pokedex_stats = self.df[columns]
        print(f"Shape final de Pokedex_stats: {Pokedex_stats.shape}")
        return Pokedex_stats

    def _save_fig(self, name):
        if self.generate_file:
            path = os.path.join(self.output_dir, name)
            plt.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Guardado: {path}")

    def _plot_preliminaries(self, df):
        df_melted = df.melt()
        plt.figure(figsize=(12, 5))
        sns.boxplot(x='variable', y='value', data=df_melted, palette='pastel', hue='variable')
        plt.title('Boxplot of Pokemon Stats')
        plt.xlabel('Stats')
        plt.ylabel('Value')
        plt.legend([], [], frameon=False)
        plt.tight_layout()
        self._save_fig("pre_boxplot.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Pokemon Stats')
        plt.tight_layout()
        self._save_fig("correlation_heatmap.png")
        plt.close()

    def train(self):
        print("Entrenando ClusteringModel siguiendo notebook original...")
        X = self.preprocess()
        self._plot_preliminaries(X)

        preprocessing_steps = [('scaler', StandardScaler())]
        evaluator = KMeansPipelineEvaluator(
            preprocessing_steps=preprocessing_steps,
            k_range=range(2, 16),
            kmeans_kwargs={'init': 'k-means++', 'random_state': self.random_state}
        )

        evaluator.fit(X)
        print("Guardando métricas de K-Means...")
        evaluator.plot_metrics()
        self._save_fig("metrics.png")
        plt.close('all')

        labels = evaluator.kmeans_models_[self.k_value].labels_
        analyzer = KMeansAnalyzer(X, labels)

        print("Generando boxplots por cluster...")
        analyzer.plot_boxplots()
        self._save_fig("cluster_boxplots.png")
        plt.close('all')

        median_profiles = analyzer.get_profiles(agg_func='median')
        print(median_profiles)
        median_profiles.to_csv(os.path.join(self.output_dir, "profiles.csv"))

        print("Generando radar charts...")
        analyzer.plot_radar_charts(agg_func='mean')
        self._save_fig("radar_charts.png")
        plt.close('all')

        print("Proceso de clustering completado.")
        return analyzer, median_profiles

