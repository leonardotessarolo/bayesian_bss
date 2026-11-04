import jax
jax.config.update("jax_enable_x64", True)

import sys
import dill
from pathlib import Path
sys.path.append('/Users/leonardo.tessarolo/git/bayesian_bss/')

import math
import numpy as np
import argparse

from src import LoggingUtilities, ContinuousExperimentExecutor, GenerateConfigs

# Number of sources and observations
NSOURCES=2
NOBS=1000

# Degrees of freedom for evaluation
SOURCE_DFS = [
    3.0,
    3.296130,
    3.747975,
    4.505851,
    6.152115
]

# Mixing matrix configuration
A = np.array([
    [math.sqrt(2)/2, -math.sqrt(2)/2],
    [math.sqrt(2)/2, math.sqrt(2)/2]
])

# Ideal separating matrix
B = np.linalg.inv(A)

def arguments():
    
    parser = argparse.ArgumentParser(
        description="Continuous experiment."
    )

    parser.add_argument(
        "--experiment-name",
        required=True,
        type=str,
        help="Experiment name.",
    )
    
    parser.add_argument(
        "--first-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize a new experiment.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Master seed."
    )

    parser.add_argument(
        "--create-folder-structure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the required output folders.",
    )

    parser.add_argument(
        "--n-realizations",
        type=int,
        default=50,
        help="Number of realizations.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default='./output',
        help="Base output directory."
    )

    return parser.parse_args()


if __name__ == "__main__":
    
    # Get inputted arguments
    args = arguments()

    # Save to variables
    FIRST_RUN = args.first_run
    CREATE_FOLDER_STRUCTURE = args.create_folder_structure
    EXPERIMENT_NAME = args.experiment_name
    N_REALIZATIONS = args.n_realizations
    OUTPUT_DIR = args.output_dir
    SEED = args.seed
    base_output_path=Path(OUTPUT_DIR)

    # Experiment directory
    experiment_dir = base_output_path / EXPERIMENT_NAME

    # Get configurations
    if FIRST_RUN:
         exec_configs = GenerateConfigs.get_continuous_configs(
            n_realizations=N_REALIZATIONS,
            base_output_path=base_output_path,
            experiment_name=EXPERIMENT_NAME,
            experiment_dir=experiment_dir,
            create_folder_structure=CREATE_FOLDER_STRUCTURE,
            nsources=NSOURCES,
            nobs=NOBS,
            A=A,
            B=B,
            source_dfs=SOURCE_DFS,
            seed=SEED
        )
    else:
        with (experiment_dir/'execution_config.pkl').open('rb') as f:
            exec_configs = dill.load(f)
    
    # Initialize folder structure, if so specified
    if CREATE_FOLDER_STRUCTURE:
        # Creates base output path and experiment dir
        if not base_output_path.is_dir():
            base_output_path.mkdir()
        if not experiment_dir.is_dir():
            experiment_dir.mkdir()

        # Creates initialization results folder
        initialization_dir = experiment_dir / 'initial_mode_seeking'
        if not initialization_dir.is_dir():
            initialization_dir.mkdir()

        # Creates folders for individual realizations
        for r in range(exec_configs['general']['n_realizations']):
            # Overall folder for realization
            realization_dir = experiment_dir / str(r)
            if not realization_dir.is_dir():
                realization_dir.mkdir()
            # Initialize success flag
            flag_path = realization_dir / "success_flag.pkl"
            if not flag_path.exists():
                with flag_path.open("wb") as f:
                    dill.dump("UNFINISHED", f)

    # Instantiate logger
    LoggingUtilities.configure_logging(
        log_dir=experiment_dir,
        log_type='continuous'
    )

    # Initialize executor and run
    ex = ContinuousExperimentExecutor(
        cfg=exec_configs,
        initialize=FIRST_RUN,
    )
    ex.run()