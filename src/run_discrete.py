import jax
jax.config.update("jax_enable_x64", True)


import sys
import dill
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # bayesian_bss/
sys.path.append(str(REPO_ROOT))

import numpy as np
import argparse

from src import DiscreteExperimentExecutor, LoggingUtilities, GenerateConfigs

def arguments():
    
    parser = argparse.ArgumentParser(
        description="Discrete experiment."
    )

    parser.add_argument(
        "--experiment-name",
        required=True,
        type=str,
        help="Experiment name.",
    )

    parser.add_argument(
        "--mode",
        choices=["narrow", "broad"],
        default="narrow",
        help="Execution mode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Master seed."
    )
    
    parser.add_argument(
        "--first-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize a new experiment.",
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
        default='./discrete_experiment/output',
        help="Base output directory."
    )

    return parser.parse_args()


if __name__ == "__main__":
    
    # Get inputted arguments
    args = arguments()

    # Save to variables
    FIRST_RUN = args.first_run
    MODE = args.mode
    CREATE_FOLDER_STRUCTURE = args.create_folder_structure
    EXPERIMENT_NAME = args.experiment_name
    N_REALIZATIONS = args.n_realizations
    OUTPUT_DIR=args.output_dir
    SEED = args.seed
    
    # Get base output path
    base_output_path = Path(OUTPUT_DIR)

    # Experiment directory
    experiment_dir = base_output_path / EXPERIMENT_NAME

    # Number of sources and observations
    NSOURCES=2
    NOBS=1000

    # Mixing matrix configuration
    A = np.array([
        [1, 1],
        [-0.5, 0.5]
    ])

    # Ideal separating matrix
    B = np.linalg.inv(A)

    # Get configurations
    if FIRST_RUN:
         exec_configs = GenerateConfigs.get_discrete_configs(
              mode=MODE,
              n_realizations=N_REALIZATIONS,
              base_output_path=base_output_path,
              experiment_name=EXPERIMENT_NAME,
              experiment_dir=experiment_dir,
              create_folder_structure=CREATE_FOLDER_STRUCTURE,
              nsources=NSOURCES,
              nobs=NOBS,
              A=A,
              B=B,
              seed=SEED
         )
    else:
        with (experiment_dir/'execution_config.pkl').open('rb') as f:
            exec_configs = dill.load(f)
        
        # Point the saved paths at the current layout (configs saved under final_experiment/ hold stale relative paths)
        exec_configs['general']['base_output_path'] = base_output_path
        exec_configs['general']['experiment_dir'] = experiment_dir
    
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
        log_type='discrete'
    )

    # Initialize executor and run
    ex = DiscreteExperimentExecutor(
        cfg=exec_configs,
        initialize=FIRST_RUN
    )
    ex.run()