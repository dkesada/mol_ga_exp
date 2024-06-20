from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Callable, Optional, Union

import joblib
import numpy as np

from .cached_function import CachedBatchFunction
from mol_ga.mol_libraries import draw_grid
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
from mol_ga.sample_population import uniform_qualitle_sampling
import heapq
from pydantic import BaseModel
from typing import Dict, List, Tuple


@dataclass
class GenInfo(BaseModel):
    max: float
    avg: float
    median: float
    min: float
    std: float
    size: int
    num_func_eval: int


@dataclass
class GAResults(BaseModel):
    """
    Results from a GA run.
    populations: list[tuple[float, str]], a list of lists with all the generations and the particles in each population
    scoring_func_evals: dict[str, float], a dictionary with all the molecules evaluated and their scores
    gen_info: list[GenInfo], a list with information about each generation of particles. It shows the maximum,
        average, median, minimum and standard deviation of the fitness of the population, the size of the population
        and the number of evaluations of the fitness function.
    """

    populations: List[List[Tuple[float, str]]]
    scoring_func_evals: Dict[str, float]
    gen_info: List[GenInfo]
    params: Dict[str, Any]


class GAController:
    """
    Main class for a genetic algorithm controller. This encapsulates the original ga code inside an object that is
    easier to extend to new modifications via converting the main body of the algorithm into smaller functions inside
    this object. Afterwards, new modifications to the genetic algorithm can be made by extending this class.
    """
    def __init__(self, scoring_func: Union[Callable[[list[str]], list[float]], CachedBatchFunction],
                 starting_population_smiles: list[str], max_generations: int = 50, population_size: int = 10, offspring_size: int = 10,
                 sampling_func: Callable[[list[tuple[float, str]], int, random.Random], list[str]] = None,
                 offspring_gen_func: Callable[[list[str], int, random.Random, Optional[joblib.Parallel]], set[str]] = None,
                 selection_func: Callable[[int, list[tuple[float, str]]], list[tuple[float, str]]] = None,
                 rng: Optional[random.Random] = None,
                 num_samples_per_generation: Optional[int] = None, logger: Optional[logging.Logger] = None,
                 parallel: Optional[joblib.Parallel] = None, plot_gen: bool = False, st_container=None):
        """
        Creates a genetic algorithm controller to maximize `scoring_func`.

        Args:
            scoring_func: Function that takes a list of SMILES and returns a list of scores.
                Convention: scoring function is being MAXIMIZED!
            starting_population_smiles: Set of SMILES to start the GA with.
            sampling_func: Function that takes a list of (score, smiles) tuples and returns a list of SMILES.
                This is used to sample SMILES from the population to create offspring.
            offspring_gen_func: Function that takes a list of SMILES and returns a list of SMILES.
                This can be anything (e.g. crossover, mutation).
            selection_func: Function to select the new population.
                This can be anything from "take the N best scores" to some more complicated method to pick
                a diverse subset.
            max_generations: Maximum number of generations to run the GA for.
            population_size: Number of SMILES to keep in the population.
            offspring_size: Number of offspring to create per generation.
            rng: Random number generator.
            num_samples_per_generation: Number of samples to take from the population to create offspring.
            logger: Logger to use.
            parallel: Joblib parallel object to use (to generate offspring in parallel).
            plot_gen: Whether to plot each generation with rdkit Draw
            st_container: An optional external streamlit.container where to dump the text and plots. If not None,
                st_container.image and st_container.write will be used
        """
        self.scoring_func = scoring_func
        self.logger = logger or logging.getLogger(__name__)
        self.st_container = st_container
        # Create the cached scoring function
        if not isinstance(self.scoring_func, CachedBatchFunction):
            self.scoring_func = CachedBatchFunction(self.scoring_func)
        self.start_cache_size = len(self.scoring_func.cache)
        self.log_print(f"Starting cache made, has size {self.start_cache_size}", True)
        self.starting_population_smiles = self.remove_duplicates(starting_population_smiles)
        self.population = None
        self.gen_info = None
        self.sampling_func = sampling_func or uniform_qualitle_sampling
        self.offspring_gen_func = offspring_gen_func or graph_ga_blended_generation
        self.selection_func = selection_func or heapq.nlargest
        self.max_generations = max_generations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.rng = rng or random.Random()
        self.num_samples_per_generation = num_samples_per_generation or 2 * offspring_size
        self.parallel = parallel
        self.plot_gen = plot_gen

    def log_print(self, msg, debug=False):
        if debug:
            self.logger.debug(msg)
        else:
            self.logger.info(msg)
            if self.st_container:
                self.st_container.write(msg)

    @staticmethod
    def remove_duplicates(smiles):
        rep = set()
        return [x for x in smiles if x not in rep and (rep.add(x) or True)]

    def score_ini_smiles(self):
        population_smiles = self.starting_population_smiles
        population_scores = self.scoring_func.eval_batch(population_smiles)
        _starting_max_score = max(population_scores)
        self.log_print(f"Initial population scoring done. Pop size={len(population_smiles)}, Max={_starting_max_score}", True)
        self.population = list(zip(population_scores, population_smiles))
        del population_scores, population_smiles, _starting_max_score

    def ini_population_sel(self):
        self.population = self.selection_func(self.population_size, self.population)

    def plot_population(self):
        if self.plot_gen:
            _, plot_smiles = tuple(zip(*self.population))  # type: ignore[assignment]
            draw_grid(plot_smiles, self.st_container)

    def select_new_population(self, population_scores, population_smiles):
        self.population = list(zip(population_scores, population_smiles))
        self.population = self.selection_func(self.population_size, self.population)

    def perform_iteration(self):
        # Separate out into SMILES and scores
        _, population_smiles = tuple(zip(*self.population))  # type: ignore[assignment]

        # Sample SMILES from population to create offspring
        samples_from_population = self.sampling_func(self.population, self.num_samples_per_generation, self.rng)

        # Create the offspring
        offspring = self.offspring_gen_func(samples_from_population, self.offspring_size, self.rng, self.parallel)

        # Add to population, ensuring uniqueness
        population_smiles = list(set(population_smiles) | offspring)  # type: ignore[misc]  # thinks var deleted
        self.log_print(f"\t{len(offspring)} created", True)
        self.log_print(f"\tNew population size = {len(population_smiles)}", True)
        del offspring

        # Score new population
        self.log_print("\tCalling scoring function...", True)
        population_scores = self.scoring_func.eval_batch(population_smiles)
        self.log_print(f"\tScoring done, best score now {max(population_scores)}.", True)

        # Select new population
        self.select_new_population(population_scores, population_smiles)

        # Plot the new population
        self.plot_population()

        # Log results of this generation
        self.log_gen_results()
        del population_scores, population_smiles

    def log_gen_results(self):
        population_scores, population_smiles = tuple(zip(*self.population))  # type: ignore[assignment]
        gen_stats_dict = GenInfo(
            max=np.max(population_scores),
            avg=np.mean(population_scores),
            median=np.median(population_scores),
            min=np.min(population_scores),
            std=np.std(population_scores),
            size=len(population_scores),
            num_func_eval=len(self.scoring_func.cache) - self.start_cache_size,
        )
        self.log_print("End of generation. Stats:\n" + pformat(gen_stats_dict))
        self.gen_info.append(gen_stats_dict)

    def run(self):
        """
        Runs the genetic algorithm maximization. The idea is to make this run function as modulated as possible in
        order to facilitate its extension afterwards. By transforming the original 'run_ga_maximization' function into
        several smaller self-contained functions, new extensions can be created by just altering this building blocks
        as needed after extending the original controller object.

        Returns:
            GAResults object containing the population, scoring function, and information about each generation.
        """
        self.log_print("Starting GA maximization...")

        # ============================================================
        # 1: prepare initial population
        # ============================================================

        # Score initial smiles
        self.score_ini_smiles()

        # Perform initial selection
        self.ini_population_sel()

        # Plot the initial population
        self.plot_population()

        # ============================================================
        # 2: run GA iterations
        # ============================================================

        # Run GA
        gen_path = list()
        gen_path.append(self.population)
        self.gen_info: List[GenInfo] = []
        for generation in range(self.max_generations):
            self.log_print(f"Start generation {generation}")
            self.perform_iteration()
            gen_path.append(self.population)

        # ============================================================
        # 3: Create return object
        # ============================================================
        self.log_print("End of GA. Returning results.")

        return GAResults(populations=gen_path, scoring_func_evals=self.scoring_func.cache,
                         gen_info=self.gen_info, params=vars(self))
