from __future__ import annotations

import heapq

from mol_ga.general_ga import GAResults, run_ga_maximization
from mol_ga.restart_ga import RGAResults, run_trga_maximization
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
from mol_ga.sample_population import uniform_qualitle_sampling


def default_ga(
    starting_population_smiles: list[str],
    scoring_function,
    max_generations: int,
    offspring_size: int,
    offspring_gen_func=graph_ga_blended_generation,
    population_sampling_function=uniform_qualitle_sampling,
    population_size=10_000,
    **kwargs,
) -> GAResults:
    """Genetic algorithm with default parameters for maximizing a scoring function."""
    return run_ga_maximization(
        starting_population_smiles=set(starting_population_smiles),
        scoring_func=scoring_function,
        offspring_gen_func=offspring_gen_func,
        sampling_func=population_sampling_function,
        selection_func=heapq.nlargest,
        max_generations=max_generations,
        population_size=population_size,
        offspring_size=offspring_size,
        **kwargs,
    )


# Function that takes a list and returns that same list in the same order but without duplicates. This is
# needed because sets are unordered by default, so if I keep the original set implementation this will create
# non-reproducible results when sampling the random initial sets from the restarts, because when you convert the set
# back to a list and sample from it, elements can be in different positions
def remove_duplicates(smiles):
    rep = set()
    return [x for x in smiles if x not in rep and (rep.add(x) or True)]


def restart_ga(
    starting_population_smiles: list[str],
    scoring_function,
    tabu_function,
    max_generations: int,
    offspring_size: int,
    offspring_gen_func=graph_ga_blended_generation,
    population_sampling_function=uniform_qualitle_sampling,
    population_size=10_000,
    restarts=4,
    conv_it=5,
    conv_th=0.3,
    tabu=True,
    **kwargs,
) -> RGAResults:
    """Genetic algorithm with default parameters for maximizing a scoring function."""
    return run_trga_maximization(
        starting_population_smiles=remove_duplicates(starting_population_smiles),
        scoring_func=scoring_function,
        tabu_func=tabu_function,
        offspring_gen_func=offspring_gen_func,
        sampling_func=population_sampling_function,
        selection_func=heapq.nlargest,
        max_generations=max_generations,
        population_size=population_size,
        offspring_size=offspring_size,
        restarts=restarts,
        conv_it=conv_it,
        conv_th=conv_th,
        tabu=tabu,
        **kwargs,
    )
