from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Callable, Optional, Union
from collections import deque

import joblib

from .cached_function import CachedBatchFunction
from mol_ga.restart_ga import RGAResults
from mol_ga.ga_controller import GAController


class TRGAController(GAController):
    """
    Class for a genetic algorithm controller with a tabu list fitness function and random restarts.
    """
    def __init__(self, scoring_func: Union[Callable[[list[str]], list[float]], CachedBatchFunction],
                 tabu_func: Callable[[str], None],
                 starting_population_smiles: list[str],
                 max_generations: int, population_size: int, offspring_size: int, restarts: int = 4,
                 conv_it: int = 5, conv_th: float = 0.3, tabu: bool = True, ini_rand: bool = False,
                 sampling_func: Callable[[list[tuple[float, str]], int, random.Random], list[str]] = None,
                 offspring_gen_func: Callable[[list[str], int, random.Random, Optional[joblib.Parallel]], set[str]] = None,
                 selection_func: Callable[[int, list[tuple[float, str]]], list[tuple[float, str]]] = None,
                 rng: Optional[random.Random] = None, num_samples_per_generation: Optional[int] = None,
                 logger: Optional[logging.Logger] = None, parallel: Optional[joblib.Parallel] = None,
                 plot_gen: boolean = False, st_container=None):
        """
        Creates a tabu random restart genetic algorithm controller to maximize `scoring_func`.

        Args:
            scoring_func: Function that takes a list of SMILES and returns a list of scores.
                Convention: scoring function is being MAXIMIZED!
            tabu_func: Function that adds a given SMILES to the tabu list of seen molecules
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
            restarts: Number of random restarts to perform
            conv_it: Number of iterations without improvement before convergence is assumed
            conv_th: Threshold of minimum improvement for considering convergence
            tabu: Whether or not to keep a tabu list
            ini_rand: Whether or not to select random molecules as the starting population for each restart
            rng: Random number generator.
            num_samples_per_generation: Number of samples to take from the population to create offspring.
            logger: Logger to use.
            parallel: Joblib parallel object to use (to generate offspring in parallel).
            plot_gen: Whether to plot each generation with rdkit Draw
        """
        GAController.__init__(self, scoring_func=scoring_func, starting_population_smiles=starting_population_smiles,
                              sampling_func=sampling_func, offspring_gen_func=offspring_gen_func,
                              selection_func=selection_func, max_generations=max_generations,
                              population_size=population_size, offspring_size=offspring_size, rng=rng,
                              num_samples_per_generation=num_samples_per_generation, logger=logger,
                              parallel=parallel, plot_gen=plot_gen, st_container=st_container)
        self.tabu_func = tabu_func
        self.restarts = restarts
        self.conv_it = conv_it
        self.conv_th = conv_th
        self.tabu = tabu
        self.ini_rand = ini_rand
        self.ini_full_population = None

    @staticmethod
    def random_sel(sel_size, population, rng):
        """ Sample 'sel_size' individuals from 'population' using the 'rng' generator """
        return rng.sample(population, sel_size)

    def ini_population_sel(self):
        if self.ini_rand:
            self.population = self.random_sel(self.population_size, self.ini_full_population, self.rng)
        else:
            self.population = self.selection_func(self.population_size, self.ini_full_population)  # Deterministic

    def run(self):
        """
        Runs the tabu random restart genetic algorithm maximization.

        Returns:
            GAResults object containing the population, scoring function, and information about each generation.
        """
        if self.tabu:
            self.log_print("Starting TRGA maximization...")
        else:
            self.log_print("Starting RGA maximization...")

        # ============================================================
        # 1: prepare initial population
        # ============================================================

        # Score initial smiles
        self.score_ini_smiles()
        self.ini_full_population = self.population.copy()

        populations: list[list[tuple[float, str]]] = []
        scoring_func_evals: list[dict[str, float]] = []
        run_info: list[list[dict[str, Any]]] = []

        # Begin the different random restarts
        for i in range(self.restarts):
            self.log_print(f"-------- Beginning restart number {i} --------")

            # Perform initial population selection
            self.ini_population_sel()

            # Plot the initial population
            self.plot_population()

            # ============================================================
            # 2: run GA iterations
            # ============================================================

            # Run GA
            max_vals = deque([-float('inf')] * self.conv_it, maxlen=self.conv_it)  # The queue of maximum values obtained
            convergence = False
            generation = 0
            gen_path = list()
            gen_path.append(self.population)
            self.gen_info: list[dict[str, Any]] = []

            while generation < self.max_generations and not convergence:
                self.log_print(f"Start generation {generation}")
                self.perform_iteration()
                gen_path.append(self.population)

                # Check convergence criteria
                max_vals.append(self.gen_info[-1]['max'])  # Add the maximum value to the queue
                if max_vals[-1] - max_vals[0] < self.conv_th:  # Check if we haven't improved conv_th in conv_it generations
                    convergence = True
                    self.log_print(f"Convergence criteria reached. No improvement of at least {self.conv_th} found in {self.conv_it} generations.")

                generation += 1

            # Update the tabu list with the best molecule found in this restart
            if self.tabu:
                smile = max(self.population)[1]
                self.tabu_func(smile)

            # Log results of this restart
            populations.append(gen_path)
            scoring_func_evals.append(self.scoring_func.cache)
            run_info.append(self.gen_info)

        # ============================================================
        # 3: Create return object
        # ============================================================
        if self.tabu:
            self.log_print("End of TRGA. Returning results.")
        else:
            self.log_print("End of RGA. Returning results.")

        return RGAResults(populations=populations, scoring_func_evals=scoring_func_evals, run_info=run_info)
