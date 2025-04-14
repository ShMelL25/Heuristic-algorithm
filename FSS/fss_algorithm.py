import torch
from torch.linalg import norm
from typing import List, Tuple, Callable
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class FishSchoolSearch:
    def __init__(self, device='cuda'):
        """Initialize the Fish School Search optimizer with GPU support"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.history = {'positions': [], 'fitness': []}
        self.best_position = None
        self.best_fitness = torch.tensor(float('inf'), device=self.device)
        self.prev_total_weight = torch.tensor(0.0, device=self.device)
        
    def _initialize_population(self, population_size: int, 
                             search_space: List[Tuple[float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize fish population within search space boundaries"""
        dimensions = len(search_space)
        positions = torch.vstack([
            torch.rand(population_size, device=self.device) * (high - low) + low
            for low, high in search_space
        ])
        velocity_scales = 0.01 * torch.tensor(
            [high - low for low, high in search_space],
            device=self.device
        )
        velocities = (2 * torch.rand(dimensions, population_size, device=self.device) - 1) * velocity_scales.unsqueeze(1)
        return positions, velocities

    def _evaluate_fitness(self, positions: torch.Tensor, 
                         fitness_func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Evaluate fitness for all fish"""
        return torch.stack([fitness_func(pos) for pos in positions.T])

    def optimize(self, fitness_func: Callable[[torch.Tensor], torch.Tensor], 
                search_space: List[Tuple[float, float]], 
                population_size: int = 50, 
                iterations: int = 100,
                individual_step: float = 0.1,
                volitive_step: float = 0.05,
                weight_scale: float = 1.0,
                verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the optimization with GPU support"""
        positions, velocities = self._initialize_population(population_size, search_space)
        fitness = self._evaluate_fitness(positions, fitness_func)
        
        best_idx = torch.argmin(fitness)
        self.best_position = positions[:, best_idx].clone()
        self.best_fitness = fitness[best_idx].clone()
        self.prev_total_weight = torch.tensor(population_size * weight_scale, device=self.device)
        
        for _ in tqdm(range(iterations), disable=not verbose):
            # Optimization steps here (same as previous implementation)
            # ...
            
            # Store history for visualization (move to CPU)
            self.history['positions'].append(positions.cpu().clone())
            self.history['fitness'].append(fitness.cpu().clone())
        
        return self.best_position, self.best_fitness