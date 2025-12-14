import random
import copy
from typing import List, Tuple, Optional
from dataclasses import dataclass

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


class GeneticProgramming:
    """Algorithme de Genetic Programming"""
    
    def __init__(self,
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
                 omics_types: List[str] = None,
                 feature_range: Tuple[int, int] = (5, 100)):
        
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
        
        # Feature selection algorithms selon le papier
        if feature_algos is None:
            self.feature_algos = [
                'Variance', 'Pearson', 'RandomForest', 'GradientBoosting',
                'AdaBoost', 'ExtraTrees', 'MutualInfo', 'FRegression'
            ]
        else:
            self.feature_algos = feature_algos
        
        # Types d'omics selon le papier (miRNA, gene expression, methylation)
        if omics_types is None:
            self.omics_types = ['miRNA', 'GeneExpression', 'Methylation']
        else:
            self.omics_types = omics_types
        
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
        """Calcul de la fitness pour toute la population"""
        print(f"\nGénération {self.generation}: Calcul de la fitness...")
        self.fitness_scores = []
        
        for individual in self.population:
            fitness = self._evaluate_fitness(individual)
            self.fitness_scores.append(fitness)
        
        # Mise à jour du meilleur individu
        max_fitness_idx = self.fitness_scores.index(max(self.fitness_scores))
        if self.fitness_scores[max_fitness_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[max_fitness_idx]
            self.best_individual = copy.deepcopy(self.population[max_fitness_idx])
        
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        print(f"Fitness moyenne: {avg_fitness:.4f}, Meilleure fitness: {self.best_fitness:.4f}")
    
    def _evaluate_fitness(self, individual: TreeNode) -> float:
        """
        Évalue la fitness d'un individu
        Dans le papier, la fitness est le C-index d'un modèle de survie
        avec validation croisée 5-fold
        
        Cette implémentation est une SIMULATION pour démonstration.
        À REMPLACER par votre vrai calcul de C-index avec vos données réelles.
        """
        # SIMULATION: Dans la vraie implémentation, vous devriez:
        # 1. Parcourir l'arbre de bas en haut (bottom-up)
        # 2. À chaque feuille, récupérer les données d'omics correspondantes
        # 3. À chaque noeud intermédiaire:
        #    - Concaténer les features des enfants
        #    - Appliquer l'algorithme de sélection de features
        #    - Sélectionner num_features features
        # 4. À la racine, entraîner un modèle Gradient Boosting Survival
        # 5. Calculer le C-index avec validation croisée 5-fold
        
        # Pour la simulation, on évalue basé sur la structure
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
    # Création et exécution de l'algorithme avec paramètres du papier
    gp = GeneticProgramming(
        population_size=50,          # Taille de population du papier
        max_generations=100,         # Nombre de générations du papier
        parent_selection_rate=0.16,  # 16% comme spécifié
        mutation_rate=0.3,           # Taux de mutation du papier (0.3 = 30%)
        elitism_count=8,             # 8 chromosomes élites (16% de 50)
        random_injection_count=8,    # 8 nouveaux chromosomes aléatoires par génération
        fitness_threshold=0.95,
        max_depth_range=(1, 4),      # Profondeur entre 1 et 4 selon le papier
        max_children_range=(1, 4),   # Nombre d'enfants entre 1 et 4
        feature_range=(5, 100)       # Plage de sélection de features
    )
    
    best_solution, best_fitness = gp.run()
    
    print(f"\n{'='*60}")
    print("ANALYSE DE LA MEILLEURE SOLUTION")
    print(f"{'='*60}")
    print(f"Fitness (C-index simulé): {best_fitness:.4f}")
    print(f"Profondeur de l'arbre: {best_solution.get_tree_depth()}")
    print(f"Nombre total de noeuds: {len(best_solution.get_all_nodes())}")
    print(f"\nRacine de l'arbre:")
    print(f"  - Max depth: {best_solution.max_depth}")
    print(f"  - Nombre d'enfants: {best_solution.num_children}")
    print(f"  - Algorithme: {best_solution.feature_selection_algo}")
    print(f"  - Nombre de features: {best_solution.num_features}")
    
    # Analyser les types d'omics utilisés
    leaf_nodes = [n for n in best_solution.get_all_nodes() if n.is_leaf()]
    omics_used = set(n.omics_type for n in leaf_nodes)
    print(f"\nTypes d'omics intégrés: {', '.join(sorted(omics_used))}")
    print(f"Nombre de feuilles: {len(leaf_nodes)}")
    
    print(f"\n{'='*60}")
    print("NOTE: Cette implémentation utilise une fitness SIMULÉE.")
    print("Pour une utilisation réelle, remplacez _evaluate_fitness()")
    print("par un calcul de C-index avec vos données multi-omics.")
    print(f"{'='*60}")
