import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class KMeansAnalyzer:
    
    def __init__(self, X_data, labels):
        if not isinstance(X_data, pd.DataFrame):
            raise TypeError("X_data debe ser un DataFrame de pandas.")
            
        self.X_data = X_data
        self.labels = labels
        
        self.df_clusters = self.X_data.copy()
        self.df_clusters['Cluster'] = self.labels 
        
        self.features_ = self.X_data.columns.to_list()

    def get_profiles(self, agg_func='median'):
        print(f"Generando perfiles de cluster usando: {agg_func}")
        
        if isinstance(agg_func, dict):
            return self.df_clusters.groupby('Cluster').agg(agg_func)
        
        return self.df_clusters.groupby('Cluster')[self.features_].agg(agg_func)

    def plot_boxplots(self, features=None, figsize=(15, 7)):
        if features is None:
            features_to_plot = self.features_
        else:
            features_to_plot = features
            
        df_melted = self.df_clusters.melt(
            id_vars=['Cluster'], 
            value_vars=features_to_plot, 
            var_name='Variable',
            value_name='Value' 
        )
        
        plt.figure(figsize=figsize)
        sns.boxplot(
            x="Variable", 
            y="Value", 
            hue="Cluster", 
            data=df_melted,
            palette="pastel"
        )
        plt.title('Distribución de Features por Cluster', fontsize=16)
        plt.xlabel('Features')
        plt.ylabel('Valor (preprocesado)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_radar_charts(self, agg_func='median', figsize=(5, 5), palette='Set2'):
        """
        Genera un radar chart por cada cluster, todos en la misma figura,
        con la misma escala para comparar perfiles visualmente.
        """
        profiles = self.get_profiles(agg_func=agg_func)
        features = profiles.columns.tolist()
        n_features = len(features)
        n_clusters = len(profiles)
        
        # Ángulos para el radar chart
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # cerrar círculo
        
        # Escala común
        min_val = profiles.min().min()
        max_val = profiles.max().max()

        # Crear figura con subplots horizontales
        fig, axes = plt.subplots(1, n_clusters, 
                                 figsize=(figsize[0]*n_clusters, figsize[1]), 
                                 subplot_kw=dict(polar=True))
        
        # Si hay solo un cluster, axes no será iterable
        if n_clusters == 1:
            axes = [axes]
        
        colors = sns.color_palette(palette, n_colors=n_clusters)

        # Dibujar un radar por cada cluster
        for ax, (cluster_id, row), color in zip(axes, profiles.iterrows(), colors):
            values = row.tolist()
            values += values[:1]

            ax.plot(angles, values, color=color, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features, fontsize=9)
            ax.set_ylim(min_val, max_val)
            ax.set_title(f'Cluster {cluster_id}', size=13, pad=15)
            ax.set_rlabel_position(0)

        plt.suptitle('Perfiles de Clusters (Radar Charts comparativos)', size=16, y=1.05)
        plt.tight_layout()
        plt.show()
