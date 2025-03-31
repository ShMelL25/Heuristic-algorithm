import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Callable
from tqdm.auto import tqdm

class FishSchoolSearch:
    def __init__(self):
        """Initialize the Fish School Search optimizer"""
        self.history = {'positions': [], 'fitness': []}  # Optimization history
        self.best_position = None      # Best solution found
        self.best_fitness = float('inf')  # Best fitness value
        self.prev_total_weight = 0.0   # Total weight from previous iteration

    def _initialize_population(self, population_size: int, 
                             search_space: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize fish population within search space boundaries
        
        Args:
            population_size: Number of fish in the population
            search_space: List of (min, max) tuples for each dimension
            
        Returns:
            Tuple of (positions, velocities) arrays
        """
        dimensions = len(search_space)
        
        # Generate random positions within bounds
        positions = np.vstack([
            np.random.uniform(low, high, population_size)
            for low, high in search_space
        ])
        
        # Initialize velocities (1% of each dimension's range)
        velocity_scales = 0.01 * np.array([high - low for low, high in search_space])
        velocities = np.random.uniform(-1, 1, (dimensions, population_size)) * velocity_scales[:, np.newaxis]
        
        return positions, velocities

    def _evaluate_fitness(self, positions: np.ndarray, 
                         fitness_func: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        Evaluate fitness for all fish
        
        Args:
            positions: Current fish positions (dimensions × population_size)
            fitness_func: Function to minimize
            
        Returns:
            Array of fitness values (population_size,)
        """
        return np.array([fitness_func(pos) for pos in positions.T])

    def _update_velocities(self, velocities: np.ndarray, 
                          positions: np.ndarray,
                          search_space: List[Tuple[float, float]],
                          step_size: float) -> np.ndarray:
        """
        Update fish velocities with random movement
        
        Args:
            velocities: Current velocities
            positions: Current positions
            search_space: Search space boundaries
            step_size: Movement step size
            
        Returns:
            Updated velocities
        """
        # Limit maximum velocity to 5% of dimension range
        max_velocity = 0.05 * np.array([high - low for low, high in search_space])[:, np.newaxis]
        random_dir = np.random.uniform(-1, 1, positions.shape)
        new_velocities = velocities + step_size * random_dir
        return np.clip(new_velocities, -max_velocity, max_velocity)

    def _calculate_weights(self, current_fitness: np.ndarray, 
                         new_fitness: np.ndarray, 
                         weight_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate fish weights based on fitness improvement
        
        Args:
            current_fitness: Fitness before movement
            new_fitness: Fitness after movement
            weight_scale: Base weight scaling factor
            
        Returns:
            Tuple of (weights, delta_fitness)
        """
        delta = current_fitness - new_fitness
        delta_max = np.max(np.abs(delta)) if np.max(np.abs(delta)) > 0 else 1.0
        weights = delta / delta_max + weight_scale
        return weights, delta

    def _instinctive_movement(self, positions: np.ndarray, 
                            delta: np.ndarray,
                            search_space: List[Tuple[float, float]]) -> np.ndarray:
        """
        Perform collective instinctive movement
        
        Args:
            positions: Current positions
            delta: Fitness improvements
            search_space: Search space boundaries
            
        Returns:
            New positions after instinctive movement
        """
        if np.sum(np.abs(delta)) > 0:
            movement = np.sum(positions * delta, axis=1) / np.sum(delta)
            new_positions = positions + movement[:, np.newaxis]
        else:
            new_positions = positions.copy()
        
        return self._clip_to_bounds(new_positions, search_space)

    def _volitive_movement(self, positions: np.ndarray, 
                         weights: np.ndarray,
                         step_size: float,
                         search_space: List[Tuple[float, float]]) -> np.ndarray:
        """
        Perform collective volitive movement (expansion/contraction)
        
        Args:
            positions: Current positions
            weights: Fish weights
            step_size: Movement step size
            search_space: Search space boundaries
            
        Returns:
            New positions after volitive movement
        """
        # Calculate school barycenter
        barycenter = np.sum(positions * weights, axis=1) / np.sum(weights)
        distances = norm(positions - barycenter[:, np.newaxis], axis=0)
        mean_dist = np.mean(distances)
        
        # Determine movement direction
        if np.sum(weights) > self.prev_total_weight:
            direction = barycenter[:, np.newaxis] - positions  # Contraction
        else:
            direction = positions - barycenter[:, np.newaxis]  # Expansion
            
        # Normalize direction and calculate step
        norm_dir = direction / (norm(direction, axis=0)[np.newaxis, :])
        step = step_size * (distances / mean_dist) * norm_dir
        
        return self._clip_to_bounds(positions + step, search_space)

    def _clip_to_bounds(self, positions: np.ndarray, 
                       search_space: List[Tuple[float, float]]) -> np.ndarray:
        """
        Clip positions to stay within search space bounds
        
        Args:
            positions: Current positions
            search_space: Search space boundaries
            
        Returns:
            Clipped positions
        """
        lows = np.array([dim[0] for dim in search_space])[:, np.newaxis]
        highs = np.array([dim[1] for dim in search_space])[:, np.newaxis]
        return np.clip(positions, lows, highs)

    def optimize(self, fitness_func: Callable[[np.ndarray], float], 
                search_space: List[Tuple[float, float]], 
                population_size: int = 50, 
                iterations: int = 100,
                individual_step: float = 0.1,
                volitive_step: float = 0.05,
                weight_scale: float = 1.0,
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the Fish School Search optimization
        
        Args:
            fitness_func: Function to minimize (accepts N-dim vector)
            search_space: List of (min, max) tuples for each dimension
            population_size: Number of fish in the school
            iterations: Number of optimization iterations
            individual_step: Individual movement step size
            volitive_step: Volitive movement step size
            weight_scale: Base weight scaling factor
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (best_position, best_fitness)
        """
        # Initialize population
        positions, velocities = self._initialize_population(population_size, search_space)
        fitness = self._evaluate_fitness(positions, fitness_func)
        
        # Set initial best solution
        best_idx = np.argmin(fitness)
        self.best_position = positions[:, best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.prev_total_weight = population_size * weight_scale
        
        # Optimization loop
        for _ in tqdm(range(iterations), disable=not verbose):
            # 1. Individual movement
            velocities = self._update_velocities(velocities, positions, search_space, individual_step)
            new_positions = self._clip_to_bounds(positions + velocities, search_space)
            new_fitness = self._evaluate_fitness(new_positions, fitness_func)
            
            # 2. Calculate weights
            weights, delta = self._calculate_weights(fitness, new_fitness, weight_scale)
            
            # 3. Instinctive movement
            instinctive_pos = self._instinctive_movement(new_positions, delta, search_space)
            instinctive_fitness = self._evaluate_fitness(instinctive_pos, fitness_func)
            
            # 4. Update best solution
            current_best_idx = np.argmin(instinctive_fitness)
            if instinctive_fitness[current_best_idx] < self.best_fitness:
                self.best_position = instinctive_pos[:, current_best_idx].copy()
                self.best_fitness = instinctive_fitness[current_best_idx]
            
            # 5. Volitive movement
            positions = self._volitive_movement(instinctive_pos, weights, volitive_step, search_space)
            fitness = self._evaluate_fitness(positions, fitness_func)
            
            self.prev_total_weight = np.sum(weights)
            
        return self.best_position, self.best_fitness