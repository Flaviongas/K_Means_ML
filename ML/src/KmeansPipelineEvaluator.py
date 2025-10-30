from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import pandas as pd



class KMeansPipelineEvaluator:
    """
    Calcula y grafica:
        1. Método del Codo (Inertia)
        2. Silhouette Score (SH)
        3. Calinski-Harabasz Index (CH)
        4. Davies-Bouldin Index (DB)
    """
    
    def __init__(self, preprocessing_steps, k_range, kmeans_kwargs=None):
        self.preprocessing_steps = preprocessing_steps
        self.k_range = k_range
        
        if kmeans_kwargs is None:
            self.kmeans_kwargs = {'init': "k-means++", 'random_state': 42}
        else:
            self.kmeans_kwargs = kmeans_kwargs
            self.kmeans_kwargs.setdefault('init', "k-means++")
            self.kmeans_kwargs.setdefault('random_state', 42)
            
        # Atributos que se llenarán después de llamar a .fit()
        self.preprocessor_pipeline_ = None
        self.results_ = None
        self.kmeans_models_ = {}
    
    def fit(self, X):
        if self.preprocessing_steps:
            self.preprocessor_pipeline_ = Pipeline(self.preprocessing_steps)
            X_processed = self.preprocessor_pipeline_.fit_transform(X)
        else:
            self.preprocessor_pipeline_ = None
            X_processed = X
        
        results_data = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
            labels = kmeans.fit_predict(X_processed)
            
            results_data['k'].append(k)
            results_data['inertia'].append(kmeans.inertia_)
            results_data['silhouette'].append(silhouette_score(X_processed, labels))
            results_data['calinski_harabasz'].append(calinski_harabasz_score(X_processed, labels))
            results_data['davies_bouldin'].append(davies_bouldin_score(X_processed, labels))
            
            self.kmeans_models_[k] = kmeans
        
        self.results_ = pd.DataFrame(results_data).set_index('k')
        return self
    
    def plot_metrics(self):
        """
        Genera una visualización de 2x2 con todas las métricas de evaluación.
        """
        if self.results_ is None:
            raise RuntimeError("Debes llamar al método .fit() primero.")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Evaluación de Clustering para K-Means', fontsize=16)
        
        # 1. Método del Codo (Inertia)
        axes[0, 0].plot(self.results_.index, self.results_['inertia'], 'bo-')
        axes[0, 0].set_title('Método del Codo (Inertia)')
        axes[0, 0].set_xlabel('Número de clusters (k)')
        axes[0, 0].set_ylabel('Inertia (WCSS)')
        
        # 2. Silhouette Score (SH)
        axes[0, 1].plot(self.results_.index, self.results_['silhouette'], 'go-')
        axes[0, 1].set_title('Silhouette Score (Mayor es mejor)')
        axes[0, 1].set_xlabel('Número de clusters (k)')
        axes[0, 1].set_ylabel('Score')
        
        # 3. Calinski-Harabasz (CH)
        axes[1, 0].plot(self.results_.index, self.results_['calinski_harabasz'], 'ro-')
        axes[1, 0].set_title('Calinski-Harabasz (Mayor es mejor)')
        axes[1, 0].set_xlabel('Número de clusters (k)')
        axes[1, 0].set_ylabel('Score')
        
        # 4. Davies-Bouldin (DB)
        axes[1, 1].plot(self.results_.index, self.results_['davies_bouldin'], 'yo-')
        axes[1, 1].set_title('Davies-Bouldin (Menor es mejor)')
        axes[1, 1].set_xlabel('Número de clusters (k)')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    
    def get_best_k(self, metric='silhouette_score', maximize=True):
        if self.results_ is None:
            raise RuntimeError("Debes llamar al método .fit() primero.")
        
        best_k = {
            "silhouette": int(self.results_['silhouette'].idxmax()),
            "calinski_harabasz": int(self.results_['calinski_harabasz'].idxmax()),
            "davies_bouldin": int(self.results_['davies_bouldin'].idxmin())
        }

        return best_k

