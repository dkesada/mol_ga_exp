# Molecule Genetic Algorithms (mol_ga)

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A simple, lightweight python package for genetic algorithms on molecules.

❗ This is an experimental, modified version of mol_ga: Simple, lightweight package for genetic algorithms on molecules.
The original algorithm was coded so that the main flow of the algorithm was performed by a single, long function from 
start to finish. When extending the original algorithm to add random restarts and a tabu list, this meant duplicating
a lot of code. This was refactored into classes in an object-oriented paradigm. Now, there are two controllers: 
GAController and TRGAController that can be imported from mol_ga. After initializing these objects, the optimization is
performed simply by calling the run() method inside the controller object.

Inheritance and the controller classes make it simple to extend the functionality of mol_ga, but I've opted to cut off 
backwards compatibility. Other mol_ga versions that use the old function-based optimization will not be compatible with
this fork of the package.

## Installation

Install using pip from this GitHub repo:

```bash
pip install git+https://github.com/dkesada/mol_ga_exp.git
```

## Why mol_ga?

There are many reasons why the original mol_ga framework is a good option in molecular optimization. This repo is 
an extension of the original one showing that it indeed was very flexible and allowed for the implementation of 
new functionalities. Please, check the [original mol_ga repository](https://github.com/AustinT/mol_ga) for further 
information about this package.

## Quick Example

Using this library is simple, but it is different from the original mol_ga package:

```python
import joblib
from rdkit import Chem
from rdkit.Chem import QED

from mol_ga import mol_libraries, GAController

# Function to optimize: we choose QED.
# mol_ga is designed for batch functions so it inputs a list of SMILES and outputs a list of floats.
f_opt = lambda s_list: [QED.qed(Chem.MolFromSmiles(s)) for s in s_list]

# Starting molecules: we choose random molecules from ZINC
# (we provide an easy handle for this)
start_smiles = mol_libraries.random_zinc(100)

# Run GA with fast parallel generation
with joblib.Parallel(n_jobs=-1) as parallel:
    ctrl = GAController(
            scoring_func=f_opt,
            starting_population_smiles=start_smiles,
            max_generations=20,
            offspring_size=10,
            population_size=10,
            parallel=parallel,
            plot_gen=False
        )
    
    res = ctrl.run()

# Print the best molecule
print(max(res.populations[-1]))
```

Output (remember it is random so results will vary between runs and between machines):

`(0.933266275123922, 'CC(C)(Nc1ccccc1)C(=O)N1CCC2COCCC2C1')`

In addition, I've also added plotting of the molecules in each generation and support for streamlit. If you pass a 
streamlit.container to the controller constructor, the plotting of the molecules in each generation will be shown in 
that container. There is also now a new controller called TRGAController that allows for random restarts and several 
optimization runs. Please, check the constructors of the 
[GAController](https://github.com/dkesada/mol_ga_exp/blob/main/mol_ga/ga_controller.py) and the 
[TRGAController](https://github.com/dkesada/mol_ga_exp/blob/main/mol_ga/trga_controller.py) classes to see the available 
options.

## Contributing

The changes I made to the original code are not backwards compatible and modify the fundamental workflow of the package, 
so I will not be making any PR's to the original repo.
