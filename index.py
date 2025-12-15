import random
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_regression,
    mutual_info_regression
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import multiprocessing as mp
from functools import partial
import os

# Import pour Gradient Boosting Survival
try:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("WARNING: scikit-survival not installed. Install with: pip install scikit-survival")
    print("Falling back to simulated fitness calculation.")

# Import pour GPU (CUDA avec CuPy)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"✓ GPU disponible: {cp.cuda.Device()}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy
    print("INFO: CuPy not installed. Running on CPU only.")
    print("For GPU acceleration, install with: pip install cupy-cuda11x or cupy-cuda12x")

# Import pour TPU (JAX)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    
    # Vérifier si TPU est disponible
    TPU_AVAILABLE = len(jax.devices('tpu')) > 0
    
    if TPU_AVAILABLE:
        print(f"✓ TPU disponible: {len(jax.devices('tpu'))} device(s)")
        print(f"  Devices: {jax.devices('tpu')}")
        # Configurer JAX pour utiliser le TPU
        jax.config.update('jax_platform_name', 'tpu')
    else:
        print("INFO: TPU not detected. JAX will use CPU/GPU.")
        
except Exception:
    TPU_AVAILABLE = False
    jnp = np  # Fallback to numpy
    print("INFO: JAX not installed. TPU acceleration unavailable.")
    print("For TPU support, install with: pip install jax[tpu]")

# Import pour parallélisation CPU
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class CSVDataLoader:
    """
    Classe pour charger et préparer les fichiers CSV multi-omics
    pour l'algorithme de Genetic Programming
    """
    
    def __init__(self):
        # Configuration des colonnes ID pour chaque fichier
        self.id_columns = {
            'everything.csv': 'Sample ID',
            'oncotype21.csv': 'Hugo_Symbol',
            'union_pam50_oncotype.csv': 'Hugo_Symbol',
            'pam50.csv': 'Hugo_Symbol',
            'intersection_pam50_oncotype.csv': 'Hugo_Symbol'
        }

        self.files = {
            'everything.csv',
            'oncotype21.csv',
            'union_pam50_oncotype.csv',
            'pam50.csv',
            'intersection_pam50_oncotype.csv'
        }
        
        # Mapper les fichiers aux types d'omics
        self.file_to_omics_mapping = {
            'oncotype21.csv': 'GeneExpression_Oncotype',
            'pam50.csv': 'GeneExpression_PAM50',
            'union_pam50_oncotype.csv': 'GeneExpression_Union',
            'intersection_pam50_oncotype.csv': 'GeneExpression_Intersection',
        }
        
        self.data_frames = {}
        self.omics_data = {}
        self.sample_ids = None
        self.survival_times = None
        self.survival_events = None
    
    def load_csv_files(self, file_paths: Dict[str, str], verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Charge tous les fichiers CSV avec leurs index appropriés
        
        Args:
            file_paths: Dictionnaire {nom_fichier: chemin_complet}
            verbose: Afficher les informations de chargement
        
        Returns:
            Dictionnaire avec les DataFrames chargés
        """
        if verbose:
            print("="*60)
            print("CHARGEMENT DES FICHIERS CSV")
            print("="*60)
        
        for file_name, file_path in file_paths.items():
            try:
                if not os.path.exists(file_path):
                    if verbose:
                        print(f"⚠ Fichier introuvable: {file_name}")
                    continue
                
                # Charger le CSV
                df = pd.read_csv(file_path)
                
                # Définir l'index selon le fichier
                id_col = "Hugo_Symbol"
                
                if id_col and id_col in df.columns:
                    df = df.set_index(id_col)
                    if verbose:
                        print(f"✓ {file_name:40s} {df.shape} (index: {id_col})")
                else:
                    if verbose:
                        print(f"⚠ {file_name:40s} {df.shape} (index non défini)")
                
                self.data_frames[file_name] = df
                
            except Exception as e:
                if verbose:
                    print(f"✗ Erreur lors du chargement de {file_name}: {str(e)}")
        
        return self.data_frames
    
    def extract_survival_data(self, everything_df: pd.DataFrame = None,
                            time_column: str = None,
                            event_column: str = None,
                            generate_synthetic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait les données de survie depuis everything.csv
        
        Args:
            everything_df: DataFrame everything.csv (ou None pour utiliser le chargé)
            time_column: Nom de la colonne temps de survie (None = auto-détection)
            event_column: Nom de la colonne événement (None = auto-détection)
            generate_synthetic: Générer des données synthétiques si colonnes absentes
        
        Returns:
            (survival_times, survival_events)
        """
        if everything_df is None:
            everything_df = self.data_frames.get('everything.csv')
        
        if everything_df is None:
            if generate_synthetic:
                print("⚠ everything.csv non disponible, génération de données synthétiques")
                n_samples = 200
                self.survival_times = np.random.exponential(scale=10, size=n_samples)
                self.survival_events = np.random.binomial(1, 0.7, size=n_samples)
                self.sample_ids = np.array([f"Sample_{i}" for i in range(n_samples)])
                return self.survival_times, self.survival_events
            else:
                raise ValueError("everything.csv n'est pas chargé")
        
        # Stocker les IDs des échantillons
        self.sample_ids = everything_df.index.values
        n_samples = len(self.sample_ids)
        
        # Auto-détection des colonnes de temps
        time_candidates = ['OS_MONTHS', 'OS.MONTHS', 'TIME', 'SURVIVAL_TIME', 'TIME_TO_EVENT']
        if time_column is None:
            for candidate in time_candidates:
                if candidate in everything_df.columns:
                    time_column = candidate
                    break
        
        # Auto-détection des colonnes d'événement
        event_candidates = ['OS_STATUS', 'OS.STATUS', 'STATUS', 'EVENT', 'VITAL_STATUS']
        if event_column is None:
            for candidate in event_candidates:
                if candidate in everything_df.columns:
                    event_column = candidate
                    break
        
        # Extraire les temps
        if time_column and time_column in everything_df.columns:
            self.survival_times = everything_df[time_column].values
            print(f"✓ Colonne temps trouvée: {time_column}")
        else:
            if generate_synthetic:
                print(f"⚠ Colonne temps non trouvée, génération synthétique")
                self.survival_times = np.random.exponential(scale=10, size=n_samples)
            else:
                available_cols = ', '.join(everything_df.columns[:10].tolist())
                raise ValueError(f"Colonne temps introuvable. Colonnes disponibles: {available_cols}...")
        
        # Extraire les événements
        if event_column and event_column in everything_df.columns:
            events = everything_df[event_column]
            # Convertir si format texte (LIVING/DECEASED, ALIVE/DEAD, etc.)
            if events.dtype == 'object':
                # Essayer différentes variantes
                deceased_values = ['DECEASED', 'DEAD', '1', 'TRUE', 'YES', 'EVENT']
                events_lower = events.astype(str).str.upper()
                self.survival_events = events_lower.isin(deceased_values).astype(int).values
            else:
                self.survival_events = events.astype(int).values
            print(f"✓ Colonne événement trouvée: {event_column}")
        else:
            if generate_synthetic:
                print(f"⚠ Colonne événement non trouvée, génération synthétique")
                self.survival_events = np.random.binomial(1, 0.7, size=n_samples)
            else:
                available_cols = ', '.join(everything_df.columns[:10].tolist())
                raise ValueError(f"Colonne événement introuvable. Colonnes disponibles: {available_cols}...")
        
        print(f"\n✓ Données de survie extraites:")
        print(f"  - {len(self.survival_times)} échantillons")
        print(f"  - Temps: min={self.survival_times.min():.2f}, max={self.survival_times.max():.2f}")
        print(f"  - Événements: {self.survival_events.sum()}/{len(self.survival_events)} " +
              f"({100*self.survival_events.mean():.1f}%)")
        
        return self.survival_times, self.survival_events
    
    def get_data_for_gp(self) -> Tuple[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Retourne les données prêtes pour GeneticProgramming
        
        Returns:
            (omics_data, (survival_times, survival_events))
        """
        if not self.omics_data:
            raise ValueError("Appelez d'abord prepare_omics_data()")
        
        if self.survival_times is None or self.survival_events is None:
            raise ValueError("Appelez d'abord extract_survival_data()")
        
        return self.omics_data, (self.survival_times, self.survival_events)
    
    @staticmethod
    def load_from_directory(directory: str, 
                           time_column: str = None,
                           event_column: str = None,
                           files_to_load: List[str] = None,
                           generate_synthetic_survival: bool = True) -> 'CSVDataLoader':
        """
        Méthode pratique pour charger tous les fichiers depuis un répertoire
        
        Args:
            directory: Chemin du répertoire contenant les CSV
            time_column: Colonne temps de survie (None = auto-détection)
            event_column: Colonne événement (None = auto-détection)
            files_to_load: Liste des fichiers à charger (None = tous)
            generate_synthetic_survival: Générer données synthétiques si absentes
        
        Returns:
            Instance de CSVDataLoader avec données chargées
        """
        loader = CSVDataLoader()
        
        # Définir les fichiers à charger
        if files_to_load is None:
            files_to_load = list(loader.id_columns.keys())
        
        # Créer les chemins complets
        file_paths = {
            file_name: os.path.join(directory, file_name)
            for file_name in files_to_load
        }
        
        # Charger les fichiers
        loader.load_csv_files(file_paths)
        
        return loader


@dataclass
class TreeNode:
    """Représente un noeud dans l'arbre chromosomique"""
    max_depth: int  # Profondeur maximale du sous-arbre
    num_children: int  # Nombre de fils
    feature_selection_algo: str  # Algorithme de sélection de features
    num_features: int  # Nombre de features à sélectionner
    children: List['TreeNode']
    depth: int  # Profondeur actuelle du noeud
    omics_type: Optional[str] = None  # Type d'omics pour les feuilles
    
    def is_leaf(self) -> bool:
        """Vérifie si c'est un noeud feuille"""
        return len(self.children) == 0
    
    def get_tree_depth(self) -> int:
        """Calcule la profondeur réelle de l'arbre"""
        if not self.children:
            return self.depth
        return max(child.get_tree_depth() for child in self.children)
    
    def get_all_nodes(self) -> List['TreeNode']:
        """Retourne tous les noeuds de l'arbre"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes
    
    def get_subtree_at_depth(self, target_depth: int) -> Optional['TreeNode']:
        """Trouve un sous-arbre à une profondeur spécifique"""
        if self.depth == target_depth:
            return self
        for child in self.children:
            result = child.get_subtree_at_depth(target_depth)
            if result:
                return result
        return None


class FeatureSelector:
    """
    Classe pour implémenter tous les algorithmes de sélection de features
    mentionnés dans le papier - avec support GPU/TPU optionnel
    """
    
    @staticmethod
    def select_features(X: np.ndarray, y: np.ndarray, method: str, 
                       n_features: int, feature_names: List[str] = None,
                       use_gpu: bool = False, use_tpu: bool = False) -> Tuple[np.ndarray, List[int]]:
        """
        Sélectionne les meilleures features selon la méthode spécifiée
        
        Args:
            X: Matrice de features (n_samples, n_features)
            y: Variable cible (n_samples,)
            method: Nom de la méthode de sélection
            n_features: Nombre de features à sélectionner
            feature_names: Noms optionnels des features
            use_gpu: Utiliser le GPU si disponible
            use_tpu: Utiliser le TPU si disponible (prioritaire sur GPU)
            
        Returns:
            X_selected: Matrice avec features sélectionnées
            selected_indices: Indices des features sélectionnées
        """
        if X.shape[1] <= n_features:
            return X, list(range(X.shape[1]))
        
        n_features = min(n_features, X.shape[1])
        
        if method == 'Variance':
            return FeatureSelector._variance_selection(X, n_features, use_gpu, use_tpu)
        elif method == 'Pearson':
            return FeatureSelector._pearson_selection(X, y, n_features, use_gpu, use_tpu)
        elif method == 'RandomForest':
            return FeatureSelector._random_forest_selection(X, y, n_features)
        elif method == 'GradientBoosting':
            return FeatureSelector._gradient_boosting_selection(X, y, n_features)
        elif method == 'AdaBoost':
            return FeatureSelector._adaboost_selection(X, y, n_features)
        elif method == 'ExtraTrees':
            return FeatureSelector._extra_trees_selection(X, y, n_features)
        elif method == 'MutualInfo':
            return FeatureSelector._mutual_info_selection(X, y, n_features)
        elif method == 'FRegression':
            return FeatureSelector._f_regression_selection(X, y, n_features)
        else:
            indices = np.random.choice(X.shape[1], n_features, replace=False)
            return X[:, indices], list(indices)
    
    @staticmethod
    def _variance_selection(X: np.ndarray, n_features: int, 
                           use_gpu: bool = False, use_tpu: bool = False) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur la variance - avec support GPU/TPU"""
        # TPU a priorité sur GPU
        if use_tpu and TPU_AVAILABLE:
            X_tpu = jnp.array(X)
            variances = jnp.var(X_tpu, axis=0)
            top_indices = jnp.argsort(variances)[-n_features:][::-1]
            top_indices = np.array(top_indices)
        elif use_gpu and GPU_AVAILABLE:
            X_gpu = cp.asarray(X)
            variances = cp.var(X_gpu, axis=0)
            top_indices = cp.argsort(variances)[-n_features:][::-1]
            top_indices = cp.asnumpy(top_indices)
        else:
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-n_features:][::-1]
        
        return X[:, top_indices], list(top_indices)
    
    @staticmethod
    @jit
    def _pearson_correlation_jax(X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Calcul optimisé de corrélation de Pearson sur TPU avec JAX"""
        # Centrer les données
        X_centered = X - jnp.mean(X, axis=0)
        y_centered = y - jnp.mean(y)
        
        # Calcul des corrélations
        numerator = jnp.abs(jnp.dot(X_centered.T, y_centered))
        denominator = jnp.sqrt(jnp.sum(X_centered**2, axis=0) * jnp.sum(y_centered**2))
        
        correlations = numerator / (denominator + 1e-10)
        return jnp.nan_to_num(correlations, 0)
    
    @staticmethod
    def _pearson_selection(X: np.ndarray, y: np.ndarray, n_features: int,
                          use_gpu: bool = False, use_tpu: bool = False) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur la corrélation de Pearson - avec support GPU/TPU"""
        # TPU a priorité sur GPU
        if use_tpu and TPU_AVAILABLE:
            X_tpu = jnp.array(X)
            y_tpu = jnp.array(y)
            
            # Utiliser la fonction JIT compilée
            correlations = FeatureSelector._pearson_correlation_jax(X_tpu, y_tpu)
            top_indices = jnp.argsort(correlations)[-n_features:][::-1]
            top_indices = np.array(top_indices)
            
        elif use_gpu and GPU_AVAILABLE:
            X_gpu = cp.asarray(X)
            y_gpu = cp.asarray(y)
            
            X_centered = X_gpu - cp.mean(X_gpu, axis=0)
            y_centered = y_gpu - cp.mean(y_gpu)
            
            numerator = cp.abs(cp.dot(X_centered.T, y_centered))
            denominator = cp.sqrt(cp.sum(X_centered**2, axis=0) * cp.sum(y_centered**2))
            
            correlations = numerator / (denominator + 1e-10)
            correlations = cp.nan_to_num(correlations, 0)
            top_indices = cp.argsort(correlations)[-n_features:][::-1]
            top_indices = cp.asnumpy(top_indices)
        else:
            correlations = np.abs([pearsonr(X[:, i], y)[0] if len(np.unique(X[:, i])) > 1 
                                   else 0 for i in range(X.shape[1])])
            correlations = np.nan_to_num(correlations, 0)
            top_indices = np.argsort(correlations)[-n_features:][::-1]
        
        return X[:, top_indices], list(top_indices)
    
    @staticmethod
    def _random_forest_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur l'importance des features via Random Forest"""
        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-n_features:][::-1]
            return X[:, top_indices], list(top_indices)
        except:
            return FeatureSelector._variance_selection(X, n_features)
    
    @staticmethod
    def _gradient_boosting_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur Gradient Boosting"""
        try:
            gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            gb.fit(X, y)
            importances = gb.feature_importances_
            top_indices = np.argsort(importances)[-n_features:][::-1]
            return X[:, top_indices], list(top_indices)
        except:
            return FeatureSelector._variance_selection(X, n_features)
    
    @staticmethod
    def _adaboost_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur AdaBoost"""
        try:
            ada = AdaBoostRegressor(n_estimators=50, random_state=42)
            ada.fit(X, y)
            importances = ada.feature_importances_
            top_indices = np.argsort(importances)[-n_features:][::-1]
            return X[:, top_indices], list(top_indices)
        except:
            return FeatureSelector._variance_selection(X, n_features)
    
    @staticmethod
    def _extra_trees_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur Extra Trees"""
        try:
            et = ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            et.fit(X, y)
            importances = et.feature_importances_
            top_indices = np.argsort(importances)[-n_features:][::-1]
            return X[:, top_indices], list(top_indices)
        except:
            return FeatureSelector._variance_selection(X, n_features)
    
    @staticmethod
    def _mutual_info_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur l'information mutuelle"""
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            top_indices = np.argsort(mi_scores)[-n_features:][::-1]
            return X[:, top_indices], list(top_indices)
        except:
            return FeatureSelector._variance_selection(X, n_features)
    
    @staticmethod
    def _f_regression_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> Tuple[np.ndarray, List[int]]:
        """Sélection basée sur le F-score (régression univariée)"""
        try:
            selector = SelectKBest(f_regression, k=n_features)
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            return X[:, selected_indices], list(selected_indices)
        except:
            return FeatureSelector._variance_selection(X, n_features)


class GeneticProgramming:
    """Algorithme de Genetic Programming avec parallélisation GPU/TPU/CPU"""
    
    def __init__(self,
                 dataframes,
                 population_size: int = 50,
                 max_generations: int = 100,
                 parent_selection_rate: float = 0.16,
                 mutation_rate: float = 0.3,
                 elitism_count: int = 8,
                 random_injection_count: int = 8,
                 fitness_threshold: float = 0.95,
                 max_depth_range: Tuple[int, int] = (1, 4),
                 max_children_range: Tuple[int, int] = (1, 4),
                 feature_algos: List[str] = None,
                 feature_range: Tuple[int, int] = (5, 100),
                 use_real_fitness: bool = False,
                 n_folds: int = 5,
                 use_gpu: bool = False,
                 use_tpu: bool = False,
                 n_jobs: int = -1):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.parent_selection_rate = parent_selection_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.random_injection_count = random_injection_count
        self.fitness_threshold = fitness_threshold
        self.max_depth_range = max_depth_range
        self.max_children_range = max_children_range
        self.feature_range = feature_range
        self.n_folds = n_folds
        self.dataframes = dataframes
        
        # Configuration TPU (priorité sur GPU)
        self.use_tpu = use_tpu and TPU_AVAILABLE
        if use_tpu and not TPU_AVAILABLE:
            print("WARNING: TPU requested but not available. Falling back to GPU/CPU.")
            self.use_tpu = False
        
        # Configuration GPU
        self.use_gpu = use_gpu and GPU_AVAILABLE and not self.use_tpu
        if use_gpu and not GPU_AVAILABLE and not self.use_tpu:
            print("WARNING: GPU requested but CuPy not available. Using CPU.")
            self.use_gpu = False
        
        # Configuration parallélisation CPU
        if n_jobs == -1:
            self.n_jobs = max(1, mp.cpu_count() - 1)
        else:
            self.n_jobs = max(1, n_jobs)
        
        # Afficher configuration
        accelerator = "TPU" if self.use_tpu else ("GPU" if self.use_gpu else "CPU")
        print(f"Configuration d'accélération:")
        print(f"  - Accélérateur: {accelerator}")
        if self.use_tpu:
            print(f"  - TPU devices: {len(jax.devices('tpu'))}")
        print(f"  - CPU workers: {self.n_jobs}")
        
        # Feature selection algorithms selon le papier
        if feature_algos is None:
            self.feature_algos = [
                'Variance', 'Pearson', 'RandomForest', 'GradientBoosting',
                'AdaBoost', 'ExtraTrees', 'MutualInfo', 'FRegression'
            ]
        else:
            self.feature_algos = feature_algos
        
        # Types d'omics selon le papier
        self.omics_types = ['Clinical Data', 'Pam50', 'Oncotype21']
        
        # Configuration pour fitness réelle
        self.use_real_fitness = use_real_fitness and SKSURV_AVAILABLE
        
        if self.use_real_fitness and not SKSURV_AVAILABLE:
            print("WARNING: Cannot use real fitness - scikit-survival not available")
            self.use_real_fitness = False
        
        self.population: List[TreeNode] = []
        self.fitness_scores: List[float] = []
        self.best_individual: Optional[TreeNode] = None
        self.best_fitness: float = 0.0
        self.generation: int = 0
    
    def initialize_population(self):
        """Initialisation de la population aléatoire"""
        print("Initialisation de la population...")
        self.population = []
        for _ in range(self.population_size):
            max_depth = random.randint(*self.max_depth_range)
            tree = self._create_random_tree(max_depth, current_depth=0)
            self.population.append(tree)
        print(f"Population de {self.population_size} individus créée")
    
    def _create_random_tree(self, max_depth: int, current_depth: int, parent_omics: List[str] = None) -> TreeNode:
        """
        Crée un arbre aléatoire récursivement
        Respecte les contraintes du papier:
        - Les feuilles contiennent un type d'omics
        - Pas de doublons d'omics sous le même parent
        - Chaque noeud à profondeur n doit avoir au moins un enfant à profondeur n-1
        """
        if parent_omics is None:
            parent_omics = []
        
        # Si on est à la profondeur maximale, créer une feuille
        if current_depth >= max_depth:
            available_omics = [o for o in self.omics_types if o not in parent_omics]
            if not available_omics:
                available_omics = self.omics_types
            
            omics = random.choice(available_omics)
            return TreeNode(
                max_depth=0,
                num_children=0,
                feature_selection_algo='',
                num_features=0,
                children=[],
                depth=current_depth,
                omics_type=omics
            )
        
        # Sinon, créer un noeud intermédiaire
        num_children = random.randint(*self.max_children_range)
        algo = random.choice(self.feature_algos)
        num_features = random.randint(*self.feature_range)
        
        node = TreeNode(
            max_depth=random.randint(current_depth + 1, max_depth),
            num_children=num_children,
            feature_selection_algo=algo,
            num_features=num_features,
            children=[],
            depth=current_depth,
            omics_type=None
        )
        
        # Créer les enfants en évitant les doublons d'omics
        used_omics = []
        for _ in range(num_children):
            child_max_depth = min(node.max_depth, max_depth)
            child = self._create_random_tree(child_max_depth, current_depth + 1, used_omics)
            node.children.append(child)
            if child.is_leaf():
                used_omics.append(child.omics_type)
        
        return node
    
    def fitness_calculation(self):
        """
        Calcul de la fitness pour toute la population
        Avec parallélisation pour accélérer le calcul
        """
        mode = "RÉELLE (Gradient Boosting Survival)" if self.use_real_fitness else "SIMULÉE"
        accelerator = "TPU" if self.use_tpu else ("GPU" if self.use_gpu else f"CPU ({self.n_jobs} workers)")
        print(f"\nGénération {self.generation}: Calcul de la fitness ({mode}, {accelerator})...")
        
        # Parallélisation du calcul de fitness
        if self.n_jobs > 1 and len(self.population) > 5:
            executor_class = ThreadPoolExecutor if self.use_real_fitness else ProcessPoolExecutor
            
            with executor_class(max_workers=self.n_jobs) as executor:
                fitness_func = partial(self._evaluate_fitness_wrapper, 
                                      use_real_fitness=self.use_real_fitness,
                                      use_gpu=self.use_gpu,
                                      use_tpu=self.use_tpu,
                                      n_folds=self.n_folds,
                                      max_depth_range=self.max_depth_range)
                
                self.fitness_scores = list(executor.map(fitness_func, self.population))
        else:
            self.fitness_scores = []
            for i, individual in enumerate(self.population):
                fitness = self._evaluate_fitness(individual)
                self.fitness_scores.append(fitness)
                
                if self.use_real_fitness and (i + 1) % 5 == 0:
                    print(f"  Progression: {i + 1}/{len(self.population)} individus évalués")
        
        # Mise à jour du meilleur individu
        max_fitness_idx = self.fitness_scores.index(max(self.fitness_scores))
        if self.fitness_scores[max_fitness_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[max_fitness_idx]
            self.best_individual = copy.deepcopy(self.population[max_fitness_idx])
        
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        print(f"Fitness moyenne: {avg_fitness:.4f}, Meilleure fitness: {self.best_fitness:.4f}")
    
    @staticmethod
    def _evaluate_fitness_wrapper(individual: TreeNode, 
                                  use_real_fitness: bool,
                                  use_gpu: bool,
                                  use_tpu: bool,
                                  n_folds: int,
                                  max_depth_range: Tuple[int, int]) -> float:
        """
        Wrapper statique pour l'évaluation de fitness (nécessaire pour la parallélisation)
        """
        temp_gp = GeneticProgramming(
            use_real_fitness=use_real_fitness,
            use_gpu=use_gpu,
            use_tpu=use_tpu,
            n_folds=n_folds,
            max_depth_range=max_depth_range,
            n_jobs=1
        )
        return temp_gp._evaluate_fitness(individual)
    
    def _evaluate_fitness(self, individual: TreeNode) -> float:
        """
        Évalue la fitness d'un individu
        
        Si use_real_fitness=True: Calcule le C-index avec Gradient Boosting Survival
        Sinon: Utilise une fitness simulée basée sur la structure
        """
        if self.use_real_fitness:
            return self._evaluate_real_fitness(individual)
        else:
            return self._evaluate_simulated_fitness(individual)
    
    def _evaluate_real_fitness(self, individual: TreeNode) -> float:
        """
        Calcule la fitness réelle avec Gradient Boosting Survival et validation croisée 5-fold
        Retourne le C-index moyen
        """
        try:
            # Intégrer les features selon l'arbre chromosomique
            times, events = self.survival_data
            
            # Créer un y dummy pour la sélection de features (on utilise times)
            y_dummy = times
            
            integrated_features = self._integrate_and_select_features(
                individual, self.omics_data, y_dummy
            )
            
            # Vérifier que nous avons des features
            if integrated_features.size == 0 or integrated_features.shape[1] == 0:
                return 0.0
            
            # Validation croisée 5-fold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            c_indices = []
            
            for train_idx, val_idx in kf.split(integrated_features):
                X_train = integrated_features[train_idx]
                X_val = integrated_features[val_idx]
                
                # Créer les structured arrays pour survival
                y_train = Surv.from_arrays(
                    event=events[train_idx].astype(bool),
                    time=times[train_idx]
                )
                y_val = Surv.from_arrays(
                    event=events[val_idx].astype(bool),
                    time=times[val_idx]
                )
                
                # Entraîner le modèle Gradient Boosting Survival
                gb_surv = GradientBoostingSurvivalAnalysis(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                
                gb_surv.fit(X_train, y_train)
                
                # Prédire et calculer le C-index
                predictions = gb_surv.predict(X_val)
                
                # Calculer le C-index
                c_index = concordance_index_censored(
                    events[val_idx].astype(bool),
                    times[val_idx],
                    predictions
                )[0]
                
                c_indices.append(c_index)
            
            # Retourner le C-index moyen
            mean_c_index = np.mean(c_indices)
            return mean_c_index
            
        except Exception as e:
            print(f"Error in real fitness calculation: {str(e)}")
            return 0.0
    
    def _evaluate_simulated_fitness(self, individual: TreeNode) -> float:
        """
        Évalue la fitness simulée basée sur la structure de l'arbre
        Utilisé quand use_real_fitness=False
        """
        nodes = individual.get_all_nodes()
        
        # Bonus pour diversité des omics
        leaf_nodes = [n for n in nodes if n.is_leaf()]
        unique_omics = len(set(n.omics_type for n in leaf_nodes))
        omics_diversity_score = unique_omics / len(self.omics_types)
        
        # Bonus pour profondeur intermédiaire (ni trop plat, ni trop profond)
        depth = individual.get_tree_depth()
        optimal_depth = (self.max_depth_range[0] + self.max_depth_range[1]) / 2
        depth_score = 1.0 - abs(depth - optimal_depth) / self.max_depth_range[1]
        
        # Pénalité pour complexité excessive
        complexity_penalty = len(nodes) / 50.0
        
        # Score basé sur le nombre de features à la racine
        feature_score = min(1.0, individual.num_features / 50.0)
        
        # Combinaison des scores
        score = (omics_diversity_score * 0.3 + 
                depth_score * 0.3 + 
                feature_score * 0.3 - 
                complexity_penalty * 0.1)
        
        # Normaliser et ajouter du bruit pour simulation
        score = max(0.0, min(1.0, score))
        score = score * 0.8 + random.uniform(0, 0.2)  # Simulation de variabilité
        
        return score
    
    def _integrate_and_select_features(self, individual: TreeNode, 
                                      omics_data: Dict[str, np.ndarray],
                                      y: np.ndarray) -> np.ndarray:
        """
        Intègre les données multi-omics selon l'arbre chromosomique (bottom-up)
        Avec support GPU/TPU pour les opérations de sélection
        """
        node_features = {}
        
        def process_node(node: TreeNode) -> np.ndarray:
            """Traite un noeud récursivement (bottom-up)"""
            if id(node) in node_features:
                return node_features[id(node)]
            
            if node.is_leaf():
                features = omics_data.get(node.omics_type, np.array([]))
                node_features[id(node)] = features
                return features
            
            child_features = []
            for child in node.children:
                child_result = process_node(child)
                if child_result.size > 0:
                    child_features.append(child_result)
            
            if not child_features:
                node_features[id(node)] = np.array([])
                return np.array([])
            
            concatenated = np.hstack(child_features)
            
            # Appliquer la sélection de features avec TPU/GPU si disponible
            selected_features, _ = FeatureSelector.select_features(
                X=concatenated,
                y=y,
                method=node.feature_selection_algo,
                n_features=min(node.num_features, concatenated.shape[1]),
                use_gpu=self.use_gpu,
                use_tpu=self.use_tpu
            )
            
            node_features[id(node)] = selected_features
            return selected_features
        
        final_features = process_node(individual)
        return final_features
    
    def check_stopping_criteria(self) -> bool:
        """Vérifie les critères d'arrêt"""
        if self.generation >= self.max_generations:
            print(f"\nCritère d'arrêt atteint: Nombre maximum de générations ({self.max_generations})")
            return True
        
        if self.best_fitness >= self.fitness_threshold:
            print(f"\nCritère d'arrêt atteint: Fitness threshold ({self.fitness_threshold})")
            return True
        
        return False
    
    def select_parents(self) -> List[TreeNode]:
        """
        Sélection des parents via élitisme et roulette wheel
        Selon le papier: 16% de la population comme parents, avec élitisme
        """
        num_parents = max(2, int(self.population_size * self.parent_selection_rate))
        
        # Élitisme: garder les meilleurs (elitism_count chromosomes)
        sorted_indices = sorted(range(len(self.fitness_scores)), 
                              key=lambda i: self.fitness_scores[i], reverse=True)
        parents = [copy.deepcopy(self.population[i]) for i in sorted_indices[:self.elitism_count]]
        
        # Roulette wheel pour le reste
        remaining = num_parents - self.elitism_count
        if remaining > 0:
            total_fitness = sum(self.fitness_scores)
            
            # Éviter division par zéro
            if total_fitness == 0:
                total_fitness = 1e-10
            
            for _ in range(remaining):
                pick = random.uniform(0, total_fitness)
                current = 0
                for i, fitness in enumerate(self.fitness_scores):
                    current += fitness
                    if current > pick:
                        parents.append(copy.deepcopy(self.population[i]))
                        break
        
        print(f"Sélection de {len(parents)} parents ({self.elitism_count} élites, {remaining} roulette wheel)")
        return parents
    
    def crossover(self, parents: List[TreeNode]) -> List[TreeNode]:
        """Opérateur de crossover avec deux types selon la profondeur"""
        offspring = []
        
        while len(offspring) < self.population_size - len(parents):
            parent1, parent2 = random.sample(parents, 2)
            
            depth1 = parent1.get_tree_depth()
            depth2 = parent2.get_tree_depth()
            
            if depth1 == depth2:
                # Crossover pour parents de même profondeur
                child1, child2 = self._crossover_same_depth(parent1, parent2)
            else:
                # Crossover pour parents de profondeurs différentes
                child1, child2 = self._crossover_different_depth(parent1, parent2)
            
            offspring.extend([child1, child2])
        
        print(f"Génération de {len(offspring)} descendants par crossover")
        return offspring[:self.population_size - len(parents)]
    
    def _crossover_same_depth(self, parent1: TreeNode, parent2: TreeNode) -> Tuple[TreeNode, TreeNode]:
        """Crossover pour parents de même profondeur"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Échange du plus grand sous-arbre
        nodes1 = child1.get_all_nodes()[1:]  # Exclure la racine
        nodes2 = child2.get_all_nodes()[1:]
        
        if nodes1 and nodes2:
            # Sélection des sous-arbres les plus larges
            node1 = max(nodes1, key=lambda n: len(n.get_all_nodes()))
            node2 = max(nodes2, key=lambda n: len(n.get_all_nodes()))
            
            # Échange
            node1.children, node2.children = node2.children, node1.children
            node1.num_children, node2.num_children = node2.num_children, node1.num_children
        
        return child1, child2
    
    def _crossover_different_depth(self, parent1: TreeNode, parent2: TreeNode) -> Tuple[TreeNode, TreeNode]:
        """Crossover pour parents de profondeurs différentes"""
        # Identifier le parent plus profond et le moins profond
        if parent1.get_tree_depth() > parent2.get_tree_depth():
            deeper, shallower = copy.deepcopy(parent1), copy.deepcopy(parent2)
        else:
            deeper, shallower = copy.deepcopy(parent2), copy.deepcopy(parent1)
        
        target_depth = shallower.get_tree_depth()
        nodes_deeper = deeper.get_all_nodes()
        
        # Trouver un sous-arbre de profondeur correspondante
        matching_nodes = [n for n in nodes_deeper if n.depth <= target_depth and n != deeper]
        
        if matching_nodes:
            target_node = random.choice(matching_nodes)
            # Remplacer le sous-arbre
            target_node.children = copy.deepcopy(shallower.children)
            target_node.num_children = shallower.num_children
        
        return deeper, shallower
    
    def mutation(self, offspring: List[TreeNode]):
        """
        Opérateur de mutation selon le papier:
        - Pour chaque noeud, générer un nombre aléatoire
        - Si < mutation_rate, muter le noeud
        - Noeuds intermédiaires/racine: régénération complète des sous-arbres
        - Feuilles: changement de type d'omics si alternatives disponibles
        """
        mutated_count = 0
        
        for individual in offspring:
            nodes = individual.get_all_nodes()
            
            for node in nodes:
                if random.random() < self.mutation_rate:
                    mutated_count += 1
                    
                    if not node.is_leaf():  # Noeud intermédiaire ou racine
                        # Régénération complète des sous-arbres
                        old_depth = node.depth
                        max_depth = node.max_depth if node.max_depth > old_depth else random.randint(*self.max_depth_range)
                        
                        # Régénérer les enfants
                        node.children = []
                        num_children = random.randint(*self.max_children_range)
                        node.num_children = num_children
                        
                        used_omics = []
                        for _ in range(num_children):
                            child = self._create_random_tree(max_depth, old_depth + 1, used_omics)
                            node.children.append(child)
                            if child.is_leaf():
                                used_omics.append(child.omics_type)
                        
                        # Possibilité de changer l'algorithme de sélection
                        node.feature_selection_algo = random.choice(self.feature_algos)
                        node.num_features = random.randint(*self.feature_range)
                        
                    else:  # Noeud feuille
                        # Changement de type d'omics
                        available_omics = [o for o in self.omics_types if o != node.omics_type]
                        if available_omics:
                            node.omics_type = random.choice(available_omics)
        
        print(f"Mutation appliquée à {mutated_count} noeuds")
    
    def run(self):
        """Exécution de l'algorithme de Genetic Programming"""
        print("=" * 60)
        print("DÉMARRAGE DE L'ALGORITHME DE GENETIC PROGRAMMING")
        print("=" * 60)
        
        # 1. Initialisation de la population aléatoire
        self.initialize_population()
        
        # 2. Fitness calculation initiale
        self.fitness_calculation()
        
        # Boucle évolutionnaire
        while True:
            self.generation += 1
            
            # 3. Stopping criteria
            if self.check_stopping_criteria():
                break
            
            # 4. Sélection des parents (16% ou constante modifiable)
            parents = self.select_parents()
            
            # 5. Crossover
            offspring = self.crossover(parents)
            
            # 6. Mutation
            self.mutation(offspring)
            
            # 7. Injection de chromosomes aléatoires (selon le papier)
            random_individuals = []
            for _ in range(self.random_injection_count):
                max_depth = random.randint(*self.max_depth_range)
                random_tree = self._create_random_tree(max_depth, current_depth=0)
                random_individuals.append(random_tree)
            
            # Nouvelle population = élites + descendants + random injection
            self.population = parents + offspring + random_individuals
            
            # S'assurer que la population ne dépasse pas la taille max
            if len(self.population) > self.population_size:
                self.population = self.population[:self.population_size]
            
            # 8. Retour à la fitness calculation
            self.fitness_calculation()
        
        print("\n" + "=" * 60)
        print("ALGORITHME TERMINÉ")
        print("=" * 60)
        print(f"Meilleure fitness atteinte: {self.best_fitness:.4f}")
        print(f"Nombre de générations: {self.generation}")
        
        return self.best_individual, self.best_fitness


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("EXEMPLE D'UTILISATION AVEC FICHIERS CSV")
    print("="*60)
    
    # =========================================================================
    # OPTION 1: Chargement depuis un répertoire (méthode simple)
    # =========================================================================
    print("\nOPTION 1: Chargement automatique depuis répertoire")
    print("-"*60)
    
    # Spécifier le répertoire contenant vos fichiers CSV
    data_directory = "./data"  # MODIFIEZ CE CHEMIN
    
    # Vérifier si le répertoire existe
    if os.path.exists(data_directory):
        try:
            # Charger automatiquement tous les fichiers
            # Les colonnes de survie seront auto-détectées ou générées
            loader = CSVDataLoader.load_from_directory(
                directory=data_directory,
                time_column=None,             # Auto-détection
                event_column=None,            # Auto-détection
                files_to_load=[               # Fichiers à charger
                    'everything.csv',
                    'oncotype21.csv',
                    'pam50.csv',
                    'union_pam50_oncotype.csv',
                    'intersection_pam50_oncotype.csv'
                ],
                generate_synthetic_survival=True  # Générer si absent
            )
            
            print(f"\n✓ Données chargées avec succès!")
            print(f"\nTypes d'omics disponibles:")

            #print(loader.data_frames)
            
            # Créer et exécuter le Genetic Programming
            print(f"\n{'='*60}")
            print("EXÉCUTION DU GENETIC PROGRAMMING")
            print(f"{'='*60}")
            
            # Extraire les types d'omics disponibles
            available_omics = loader.files
            
            gp = GeneticProgramming(
                population_size=20,
                max_generations=10,
                parent_selection_rate=0.16,
                mutation_rate=0.3,
                elitism_count=3,
                random_injection_count=3,
                fitness_threshold=0.95,
                max_depth_range=(1, 3),
                max_children_range=(1, 3),
                feature_range=(5, 30),
                use_real_fitness=SKSURV_AVAILABLE,  # Active si disponible
                dataframes=loader.data_frames,
                n_folds=3,
                use_gpu=False,
                use_tpu=False,
                n_jobs=-1
            )
            
            best_solution, best_fitness = gp.run()
            
            print(f"\n{'='*60}")
            print("RÉSULTATS")
            print(f"{'='*60}")
            print(f"{'C-index' if SKSURV_AVAILABLE else 'Fitness'}: {best_fitness:.4f}")
            print(f"Profondeur de l'arbre: {best_solution.get_tree_depth()}")
            print(f"Algorithme racine: {best_solution.feature_selection_algo}")
            print(f"Features sélectionnées: {best_solution.num_features}")
            
            # Analyser les omics utilisés
            leaf_nodes = [n for n in best_solution.get_all_nodes() if n.is_leaf()]
            omics_used = set(n.omics_type for n in leaf_nodes)
            print(f"\nOmics intégrés dans la meilleure solution:")
            for omics in sorted(omics_used):
                print(f"  - {omics}")
        
        except Exception as e:
            print(f"\n✗ Erreur lors du chargement: {str(e)}")
            print(f"\nVérifications:")
            print(f"  1. Le répertoire '{data_directory}' contient vos fichiers CSV")
            print(f"  2. Les fichiers ont les bons noms")
            print(f"  3. Les colonnes sont correctes (Hugo_Symbol ou Sample ID)")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Répertoire '{data_directory}' introuvable")
        print(f"\nPassage à l'OPTION 2...")