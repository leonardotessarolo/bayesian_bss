import os
import gc
import pandas as pd
import numpy as np
import pathos
import functools
import dill
import jax.numpy as jnp
from .estimator import BayesianEstimators, MMSEBarkerMHEstimator, ImportanceSamplingEstimator
from .contour_line import PosteriorContourLines
from .utilities import PosteriorUtilities
from tqdm import tqdm 
import time
import jax
import psutil
import gc
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def jax_buffer_report(tag, top=12):
    arrs = jax.live_arrays()
    counts, nbytes = Counter(), Counter()
    for a in arrs:
        key = (tuple(a.shape), str(a.dtype))
        counts[key] += 1
        nbytes[key] += a.nbytes
    total = sum(nbytes.values())
    print(f"\n[{tag}] live_arrays={len(arrs)}  total={total/2**30:.3f} GiB")
    print(f"RSS: {psutil.Process(os.getpid()).memory_info().rss/2**30:.2f} GiB")
    for key, nb in nbytes.most_common(top):
        print(f"   {counts[key]:>6} x {str(key[0]):<22} {key[1]:<8} {nb/2**20:>}")

def find_retainers(shape=(4, 500, 2, 2), depth=3):
    arrs = [a for a in jax.live_arrays() if a.shape == shape]
    print(f"{len(arrs)} live arrays of shape {shape}")
    if not arrs:
        return
    seen, frontier = set(), [arrs[0]]
    for level in range(depth):
        nxt = []
        kinds = Counter()
        for obj in frontier:
            for r in gc.get_referrers(obj):
                if id(r) in seen:
                    continue
                seen.add(id(r))
                kinds[type(r).__name__] += 1
                nxt.append(r)
                if level >= 1 and not isinstance(r, (list, dict, tuple)):
                    print(f"  L{level}: {type(r).__module__}.{type(r).__name__} "
                          f"{repr(r)[:160]}")
        print(f"level {level}: {dict(kinds)}")
        frontier = nxt[:40]          # cap the fan-out



class DiscreteExperimentExecutor:

    def __init__(
        self,
        cfg,
        initialize=True
    ):
        if initialize:
            # Save cfg as attribute
            self.cfg=cfg

            # Save test cases as attribute
            self.test_cases = list(self.cfg['sim']['test_cases'].keys())
            
            # Get source and mixture signals for all realizations
            self.__initialize_signals()

            # Get source model and prior distributions
            self.__get_test_case_distributions()

            # Get execution functions for test cases
            self.__get_exec_fns()

            # Run initial mode seeking
            # self.__run_initial_mode_seeking()

            # Get initial starting points for Gradient Ascent and MCMC
            self.__get_initial_conditions()

            # Save cfg and signals
            self.__save_initializations()

        else:
            self.cfg=cfg

            self.__read_signals()

    
    def __initialize_signals(
        self
    ):
        
        # Parse config object
        A = self.cfg['general']['A']
        n_realizations = self.cfg['general']['n_realizations']
        n_sources = self.cfg['general']['n_sources']
        n_obs = self.cfg['general']['n_obs']

        # Initialize random seeds
        seeds = [x for x in range(n_realizations)]
        
        # Initialize dict to keep signals
        signals = {}
        
        # Iterate over realizations and generate signals
        for r in range(n_realizations):
            realization = {}
            # Generate sources for each specified configuration
            for s_name, s_obj in self.cfg['sim']['sources'].items():
                # Get source realization
                s = s_obj.get_realization(
                    nsources=n_sources,
                    nobs=n_obs,
                    seed=seeds[r]
                )
                # Get mixtures
                x = A@s
                # Save signals
                realization[s_name] = {
                    's': s,
                    'x': x
                }
                signals[str(r)] = realization

        self.signals=signals
        
    
    def __get_test_case_distributions(
        self
    ):
        
        # Iterate over test cases and save distributions
        for test_case, test_case_cfgs in self.cfg['sim']['test_cases'].items():
            # Source and prior
            source = self.cfg['sim']['source_model']
            prior = self.cfg['sim']['priors'][test_case_cfgs['prior']]
            
            # Get source model pdfs
            source_model_pdf = source.get()
            source_model_pdf_derivative = source.get_derivative()

            # Get prior pdfs
            prior_pdf = prior.get()
            prior_pdf_derivative = prior.get_derivative()

            # Configurations for MCMC sampling and Gradient Ascent optimization
            mcmc_configs, grad_asc_configs =  BayesianEstimators.generate_configs(
                source_pdf=source_model_pdf,
                source_pdf_derivative=source_model_pdf_derivative,
                prior_pdf=prior_pdf,
                prior_pdf_derivative=prior_pdf_derivative,
                normalize_posterior=self.cfg['general']['normalize_posterior'],
                n_samples_mcmc=self.cfg['mcmc']['n_samples'],
                exploration_var_mcmc=self.cfg['mcmc']['exploration_var'],
                parallel_chains_mcmc=self.cfg['mcmc']['parallel_chains'],
                max_it_mcmc=self.cfg['mcmc']['max_it'],
                R_hat_thresh_mcmc=self.cfg['mcmc']['R_hat_thresh'],
                R_hat_evaluation_step_mcmc=self.cfg['mcmc']['R_hat_evaluation_step'],
                R_hat_persistance_mcmc=self.cfg['mcmc']['R_hat_persistance'],
                R_hat_minimum_burn_in_mcmc=self.cfg['mcmc']['R_hat_minimum_burn_in'],
                auto_adjust_exploration_var_mcmc=self.cfg['mcmc']['auto_adjust_exploration_var'],
                progress_bar_mcmc=self.cfg['mcmc']['progress_bar'],
                target_accept_prob_mcmc=self.cfg['mcmc']['target_accept_prob'],
                learning_rate_grad_asc=self.cfg['map']['learning_rate'],
                stopping_thresh_grad_asc=self.cfg['map']['stopping_thresh'],
                max_it_grad_asc=self.cfg['map']['max_it'],
                min_it_grad_asc=self.cfg['map']['min_it'],
                stopping_criterion_persistance_its_grad_asc=self.cfg['map']['stopping_criterion_persistance_its']
            )
            
            # Set configuration in estimators to run both
            mcmc_configs['run_mcmc'] = True
            grad_asc_configs['run_grad_asc'] = True

            # Save MAP and MCMC configs
            self.cfg['sim']['test_cases'][test_case]['mcmc_configs']=mcmc_configs
            self.cfg['sim']['test_cases'][test_case]['grad_asc_configs']=grad_asc_configs

    def __get_exec_fns(
        self
    ):
        def __run_estimators(
            mcmc_configs,
            grad_asc_configs,
            initial_conditions,
            s,
            x
        ):
            """
                Runs estimators for one realization
            """
            # Run estimators
            mmse_estimator, map_estimator = BayesianEstimators.run(
                s=s,
                x=x,
                mmse_configs=mcmc_configs,
                map_configs=grad_asc_configs,
                initial_conditions=initial_conditions
            )

            return mmse_estimator, map_estimator

        def __run_contour(
            u_lims,
            v_lims,
            central_point,
            contour_grid_points,
            source_model_pdf,
            prior_pdf,
            A,
            n_obs,
            n_workers,
            x
        ):
            """
                Runs posteriori grid for one realization
            """
            # Initialize object used to obtain contour lines
            c_lines = PosteriorContourLines(
                u_lims=u_lims,
                v_lims=v_lims,
                n_points=contour_grid_points,
                central_point=central_point,
                source_pdf_fn=source_model_pdf,
                prior_pdf_fn=prior_pdf,
            )
            
            # Evaluate posterior in neighborhood of true point
            c_lines.get_posteriori_neighborhood(
                A=A,
                x=x,
                N=n_obs,
                njobs=n_workers
            )
            
            # Maximum a posteriori point
            c_lines.get_max_point()

            return c_lines

        
        # Iterate over test cases and obtain functions to be executed for each realization
        for test_case, test_case_cfgs in self.cfg['sim']['test_cases'].items():
            
            # Get log posterior function for test case
            # Source and prior
            log_posterior_fn = PosteriorUtilities.get_log_posterior_fn(
                source_pdf_fn=self.cfg['sim']['source_model'].get(),
                prior_pdf_fn=self.cfg['sim']['priors'][test_case_cfgs['prior']].get()
            )

            # Bayesian estimators execution function - regular executions
            bayesian_estimators_fn = functools.partial(
                __run_estimators,
                test_case_cfgs['mcmc_configs'],
                test_case_cfgs['grad_asc_configs']
            )

            # Posteriori grid execution function
            source_type = self.cfg['sim']['test_cases'][test_case]['source']
            posteriori_grid_fn = functools.partial(
                __run_contour,
                self.cfg['contour'][source_type]['u_lims'],
                self.cfg['contour'][source_type]['v_lims'],
                self.cfg['contour'][source_type]['central_point'],
                self.cfg['contour'][source_type]['contour_grid_points'],
                test_case_cfgs['grad_asc_configs']['source_pdf'],
                test_case_cfgs['grad_asc_configs']['prior_pdf'],
                self.cfg['general']['A'],
                self.cfg['general']['n_obs'],
                self.cfg['general']['n_workers'],
            )

            # Save functions
            self.cfg['sim']['test_cases'][test_case]['log_posterior_fn']=log_posterior_fn
            self.cfg['sim']['test_cases'][test_case]['bayesian_estimators_fn']=bayesian_estimators_fn
            self.cfg['sim']['test_cases'][test_case]['posteriori_grid_fn']=posteriori_grid_fn


    def __run_initial_mode_seeking(
        self
    ):
        def __run_realization(
            cfg,
            realization_info,
        ):
            """
                Runs initial mode seeking for one realization
            """
            # Parse realization info
            realization_name = realization_info[0]
            realization_cfgs = realization_info[-1]
            
            # Iterate over test cases and execute experiment
            realization_results = {}
            for test_case, test_case_cfgs in self.cfg['sim']['test_cases'].items():
                # Source for test case
                test_case_source = test_case_cfgs['source']

                # Obtain sources and mixtures
                s = realization_cfgs[test_case_source]['s']
                x = realization_cfgs[test_case_source]['x']
                
                # Retrieve function for executing bayesian estimators 
                bayesian_estimators_fn = test_case_cfgs['initialization_bayesian_estimators_fn']

                # Run bayesian estimators
                _ , map_estimator = bayesian_estimators_fn(
                    s=s,
                    x=x,
                    initial_conditions={
                        'map': [self.cfg['general']['B']]
                    }
                )
                
                # Updates realization cfgs
                realization_results[test_case] = {
                    's': s,
                    'x': x,
                    'map_estimator': map_estimator
                }
            
            # Saves to realization_cfgs
            realization_cfgs['results'] = realization_results

            # Save results
            realization_dir = cfg['general']['experiment_dir']  / 'initial_mode_seeking' / realization_name
            with (realization_dir/'results_raw.pkl').open('wb') as f:
                dill.dump(realization_cfgs, f)

            return realization_results
        

        def __parse_initial_mode_seeking_results(
            results
        ):
            # Raw results
            parsed_results = {
                'raw_results': results
            }

            # Estimated mode for each test case
            estimated_modes = {
                test_case: np.median(
                    np.array([
                        res[test_case]['map_estimator'].B_est for res in results
                    ]),
                    axis=0
                ) for test_case in self.test_cases
            }

            # Save to parsed results
            parsed_results['estimated_modes']=estimated_modes

            # Save modes per test case
            for test_case, mode in estimated_modes.items():
                self.cfg['sim']['test_cases'][test_case]['estimated_mode']=mode
                

            return parsed_results
    


        # Create execution functions for each realization
        realization_fn = functools.partial(
            __run_realization,
            self.cfg
        )
        
        # Run experiment for each realization
        iterable = list(self.__get_iterable())[:self.cfg['initial_mode_seeking']['n_realizations']]
        if (self.cfg['general']['n_workers'] > 1) and (self.cfg['initial_mode_seeking']['n_realizations'] > 1):
            with pathos.pools.ProcessPool(self.cfg['general']['n_workers']) as p:
                results = p.map(
                    realization_fn, 
                    iterable
                )
        else:
            results = [
                realization_fn(elem) for elem in iterable
            ]

        # Debug
        # iterable = list(self.__get_iterable())[:self.cfg['initial_mode_seeking']['n_realizations']]
        # results = [
        #     realization_fn(elem) for elem in iterable
        # ]

        # Save as attribute
        self.initial_mode_seeking_results = __parse_initial_mode_seeking_results(
            results=results
        )


    def __get_initial_conditions(
        self
    ):
        
        # Find starting points randomly
        if self.cfg['initialization']['strategy'] == 'random':
            # Iterate test cases, and for each test case find parallel_chains initial points with finite log posterior
            iterable = self.cfg['sim']['test_cases'].items()
            if self.cfg['general']['initialization_progress_bar']:
                iterable = tqdm(iterable)
            for test_case, test_case_cfgs in iterable:
                test_case_cfgs['initial_conditions'] = {}
                for r in range(self.cfg['general']['n_realizations']):
                    initial_condition = np.empty(
                        shape = (0,) + self.cfg['general']['B'].shape
                    )
                    # Get x
                    source_type = self.cfg['sim']['test_cases'][test_case]['source']
                    x = self.signals[str(r)][source_type]['x']

                    # Draw while enough points are not found
                    draw_points = self.cfg['mcmc']['parallel_chains']
                    while len(initial_condition) < self.cfg['mcmc']['parallel_chains']:
                        # Draw points
                        random_draw = np.asarray([
                            self.cfg['mcmc']['starting_distribution']() for _ in range(2*draw_points)
                        ])

                        # Get posteriors for points
                        posteriors = np.array([
                            self.cfg['sim']['test_cases'][test_case]['log_posterior_fn'](
                                x=x,
                                B=B_0
                            ) for B_0 in random_draw
                        ])

                        # Finite idxs
                        finite_idxs = np.where(np.array(posteriors) > -np.inf)[0]
                        if len(finite_idxs) > 0:
                            initial_condition = np.concatenate(
                                [
                                    initial_condition,
                                    random_draw[finite_idxs]
                                ],
                                axis=0
                            )

                        # Number of points to draw
                        draw_points = max(
                            0,
                            self.cfg['mcmc']['parallel_chains'] - len(initial_condition)
                        )

                    if len(initial_condition) > self.cfg['mcmc']['parallel_chains']:
                        initial_condition = initial_condition[:self.cfg['mcmc']['parallel_chains']]

                    # Save to dict
                    test_case_cfgs['initial_conditions'][str(r)] = {
                        'map': [initial_condition[0]],
                        'mmse': initial_condition,
                    }
                
                # Save to object attribute
                self.cfg['sim']['test_cases'][test_case] = test_case_cfgs

        # Starting points from Gaussian Mixture Model analysis
        elif self.cfg['initialization']['strategy'] == 'gmm':
            # Path for starting points
            gmm_means_path = self.cfg['general']['base_output_path'] / '{}/{}.pkl'.format(
                self.cfg['initialization']['starting_value_experiment'],
                'gmm_means_starting_points'
            )

            # Read gmm means
            with (gmm_means_path).open('rb') as f:
                gmm_means = dill.load(f)

            iterable = self.cfg['sim']['test_cases'].items()
            for test_case, test_case_cfgs in iterable:
                # Starting point for map is gaussian mean
                map_initial_condition = gmm_means[test_case]

                # Starting points for mmse are random points around gaussian mean
                mmse_initial_condition = np.asarray([
                    gmm_means[test_case] + self.cfg['mcmc']['starting_distribution']() for _ in range(self.cfg['mcmc']['parallel_chains'])
                ])

                # Save to dict
                test_case_cfgs['initial_conditions'] = {
                    'map': [map_initial_condition],
                    'mmse': mmse_initial_condition,
                }

                # Save to object attribute
                self.cfg['sim']['test_cases'][test_case] = test_case_cfgs

        

    def __save_initializations(
        self
    ):
        # Get experiment_dir
        experiment_dir = self.cfg['general']['experiment_dir']
        
        # Iterate over realizations and save signals
        for r, res in self.signals.items():
            # Get dir for saving realization results
            realization_dir = experiment_dir / r

            # Save results
            with (realization_dir/'signals.pkl').open('wb') as f:
                dill.dump(res, f)

        # Save updated config file
        with (experiment_dir/'execution_config.pkl').open('wb') as f:
                dill.dump(self.cfg, f)


    def __read_signals(
        self
    ):
        
        # Get realization iterable
        realizations = range(self.cfg['general']['n_realizations'])

        # Get experiment dir
        experiment_dir = self.cfg['general']['experiment_dir']

        # Read signals
        signals = {}
        for r in realizations:
            # Get dir for reading realization results
            realization_dir = experiment_dir / str(r)

            # Read signals
            with (realization_dir/'signals.pkl').open('rb') as f:
                signals[str(r)] = dill.load(f)
            
        self.signals=signals

    def __get_iterable(
            self
    ):
        # Get realization iterable
        realizations = range(self.cfg['general']['n_realizations'])

         # Get experiment dir
        experiment_dir = self.cfg['general']['experiment_dir']

        # Iterate and identify finished realizations
        finished_realizations = []
        for r in realizations:
            
            # Get dir for reading realization results
            realization_dir = experiment_dir / str(r)

            # Read success flag
            with (realization_dir/'success_flag.pkl').open('rb') as f:
                success_flag = dill.load(f)

            if success_flag=='SUCCESS':
                finished_realizations.append(str(r))
        
        
        # Get realizations to run
        unfinished_realizations_signals = {
            k:v for k, v in self.signals.items() if k not in finished_realizations
        }

        # Print status
        print('#'*100)
        print('Total realizations: {}'.format(self.cfg['general']['n_realizations']))
        print('Finished realizations: {}'.format(len(finished_realizations)))
        print('Unfinished realizations: {}'.format(len(unfinished_realizations_signals.keys())))
        print('#'*100)

        return unfinished_realizations_signals.items()
    

    def run(
        self
    ):

        def __run_realization(
            cfg,
            realization_info,
        ):
            """
                Runs experiment for one realization
            """
            
            # Parse realization info
            realization_name = realization_info[0]
            realization_cfgs = realization_info[-1]
            
            # Log realization
            logger.info('Running realization {}.'.format(realization_name))

            # Iterate over test cases and execute experiment
            realization_results = {}
            iterable=self.cfg['sim']['test_cases'].items()
            for test_case, test_case_cfgs in iterable:
                # Log realization
                logger.info('Running test case {}.'.format(test_case))
                # Source for test case
                test_case_source = test_case_cfgs['source']

                # Obtain sources and mixtures
                s = realization_cfgs[test_case_source]['s']
                x = realization_cfgs[test_case_source]['x']
                
                # Retrieve initial conditions
                if self.cfg['initialization']['strategy']=='random':
                    initial_conditions = test_case_cfgs['initial_conditions'][str(realization_name)]
                if self.cfg['initialization']['strategy']=='gmm':
                    initial_conditions = test_case_cfgs['initial_conditions']

                # Retrieve function for executing bayesian estimators 
                bayesian_estimators_fn = test_case_cfgs['bayesian_estimators_fn']

                # Run bayesian estimators
                mmse_estimator, map_estimator = bayesian_estimators_fn(
                    s=s,
                    x=x,
                    initial_conditions=initial_conditions
                )
                
                # Retrieve function for executing posteriori grid calculation
                posteriori_grid_fn = test_case_cfgs['posteriori_grid_fn']
                
                # Run posteriori grid calculation
                start_grid = time.time_ns()
                posteriori_grid = posteriori_grid_fn(x=x)
                end_grid = time.time_ns()
                logger.info('Time spent in grid (seconds): {}'.format(round((end_grid-start_grid)/1E9, 2)))

                # Updates realization cfgs
                realization_results[test_case] = {
                    's': s,
                    'x': x,
                    'mmse_estimator': mmse_estimator,
                    'map_estimator': map_estimator,
                    'posteriori_grid': posteriori_grid
                }

            # Saves to realization_cfgs
            realization_cfgs['results'] = realization_results

            # Save results
            start_saving = time.time_ns()
            realization_dir = cfg['general']['experiment_dir']  / realization_name
            with (realization_dir/'results_raw.pkl').open('wb') as f:
                dill.dump(realization_cfgs, f)
            end_saving = time.time_ns()
            logger.info('Time spent saving (seconds): {}'.format(round((end_saving-start_saving)/1E9, 2)))

            with (realization_dir/'success_flag.pkl').open('wb') as f:
                dill.dump('SUCCESS', f)

            # Delete as results have already been written
            del realization_cfgs['results']
            del realization_results
            del mmse_estimator, map_estimator, posteriori_grid
            gc.collect()

            logger.info("JAX live arrays: %d", len(jax.live_arrays()))
            # find_retainers()
            
            # Clear jax caches
            jax.clear_caches()

            # jax_buffer_report(f"realization {realization_name}")


        # Create execution functions for each realization
        realization_fn = functools.partial(
            __run_realization,
            self.cfg
        )
        
        # Run experiment for each realization
        iterable = self.__get_iterable()
        gc.collect()
        gc.freeze()
        if self.cfg['general']['realizations_progress_bar']:
            print('Executando experimento')
            iterable = tqdm(iterable)
        for elem in iterable:
            try:
                realization_fn(elem)
            except Exception:
                logger.exception(
                    "Realization %s failed",
                    elem[0]
                )
                raise



    def rerun_contour(
        self,
        test_cases
    ):
        def __run_realization(
            cfg,
            test_cases,
            realization_info,
        ):
            """
                Runs experiment for one realization
            """

            # Parse realization info
            realization_name = realization_info[0]
            realization_cfgs = realization_info[-1]

            # Read success flag
            realization_dir = self.cfg['general']['experiment_dir'] / realization_name
            with (realization_dir/'results_raw.pkl').open('rb') as f:
                realization_cfgs = dill.load(f)
            

            # start_time = time.time()
            
            # Iterate over test cases and execute experiment
            realization_results = {}
            # last_time = start_time
            for test_case, test_case_cfgs in self.cfg['sim']['test_cases'].items():

                if test_case not in test_cases:
                    continue
                
                # Source for test case
                test_case_source = test_case_cfgs['source']

                # Obtain sources and mixtures
                s = realization_cfgs[test_case_source]['s']
                x = realization_cfgs[test_case_source]['x']
                
                # Retrieve function for executing posteriori grid calculation
                posteriori_grid_fn = test_case_cfgs['posteriori_grid_fn']
                
                # Run posteriori grid calculation
                posteriori_grid = posteriori_grid_fn(x=x)
                
                # Updates realization cfgs
                realization_cfgs['results'][test_case]['posteriori_grid'] = posteriori_grid


            # Save results
            realization_dir = cfg['general']['experiment_dir']  / realization_name
            with (realization_dir/'results_raw.pkl').open('wb') as f:
                dill.dump(realization_cfgs, f)

            with (realization_dir/'success_flag.pkl').open('wb') as f:
                dill.dump('SUCCESS', f)
            
                

        # Initializations that aren't for signals #

        # Get source model and prior distributions
        self.__get_test_case_distributions()

        # Get execution functions for test cases
        self.__get_exec_fns()

        # Get experiment_dir
        experiment_dir = self.cfg['general']['experiment_dir']

        # Save updated config file
        with (experiment_dir/'execution_config.pkl').open('wb') as f:
                dill.dump(self.cfg, f)


        # Create execution functions for each realization
        realization_fn = functools.partial(
            __run_realization,
            self.cfg,
            test_cases
        )
        
        # Run experiment for each realization
        iterable = self.__get_iterable()
        with pathos.pools.ProcessPool(self.cfg['general']['n_workers']) as p:
            p.map(
                realization_fn, 
                iterable
            )

        
        # DEBUG
        # results = []
        # iterable = self.__get_iterable()
        # for realization_info in iterable:
            
        #     results.append(
        #         realization_fn(
        #             realization_info=realization_info
        #         )
        #     )
            
        # # Parse results object
        # self.results = {}
        # for d in results:
        #     self.results[d[0]] = d[1]

        # # Save results
        # self.__save_results()



class DiscreteExperimentParser:
    def __init__(
        self,
        experiment_dir
    ):
        # Save experiment dir
        self.experiment_dir = experiment_dir

        # Test case to source map
        self.test_case_source_map = {
            'i': 'perfect_model',
            'ii': 'perfect_model',
            'iii': 'perfect_model',
            'iv': 'perfect_model',
            'v': 'slightly_misspecified_model',
            'vi': 'slightly_misspecified_model',
            'vii': 'slightly_misspecified_model',
            'viii': 'slightly_misspecified_model',
            'ix': 'largely_misspecified_model',
            'x': 'largely_misspecified_model',
            'xi': 'largely_misspecified_model',
            'xii': 'largely_misspecified_model',
        }


    def __read_execution_configs(self):
        # Read experiment configs
        with (self.experiment_dir/'execution_config.pkl').open('rb') as f:
                self.execution_config = dill.load(f)

    def __get_finished_realizations(
        self,
        verbose
    ):
        # Get realization dirs
        realizations = [x[1] for x in os.walk(self.experiment_dir)][0]
        realizations = [
            r for r in realizations if r not in [
                'analysis_all_realizations',
                'hypothesis_tests',
                'initial_mode_seeking'
            ]
        ]

        # Iterate in realizations and read success flags
        finished_realizations = []
        non_executed_realizations = []
        for r in realizations:
            # Get directory for realization
            realization_dir = self.experiment_dir / r

            # Read success flag
            with (realization_dir/'success_flag.pkl').open('rb') as f:
                success_flag = dill.load(f)

            if success_flag=='SUCCESS':
                finished_realizations.append(str(r))
            else:
                non_executed_realizations.append(str(r))
        
        self.finished_realizations = finished_realizations
        self.non_executed_realizations=non_executed_realizations
        
        if verbose:
            print('#'*100)
            print('Total of {} realizations with executed results'.format(len(finished_realizations)))
            print('-'*100)
            print('Total of {} realizations without executed results'.format(len(non_executed_realizations)))
            print('#'*100)


    def __parse_raw_results(
        self,
        verbose
    ):
        def __parse_mmse(
            mmse_estimator,
            s,
            x
        ):
            # Get MMSE estimate
            B_est_mmse = mmse_estimator.mcmc_results['B_est']

            # Get error for MMSE estimate
            mmse_error_norm = np.linalg.norm(
                np.subtract(
                    B_est_mmse,
                    self.execution_config['general']['B']
                )
            )/np.linalg.norm(self.execution_config['general']['B'])

            # Get estimated source
            s_est_mmse = B_est_mmse@x

            # Get error for estimated source
            s_est_error_norm = np.linalg.norm(
                np.subtract(
                    s_est_mmse,
                    s
                )
            )/np.linalg.norm(s)

            # Get R-hats
            diagnostics = mmse_estimator.diagnostics
            # diagnostics = mmse_estimator.R_hats
            converged = mmse_estimator.converged
            samples = mmse_estimator.samples

            return B_est_mmse, mmse_error_norm, s_est_mmse, s_est_error_norm, diagnostics, converged, samples

        def __parse_map(
            map_estimator,
            s,
            x
        ):
            # Get results for analyzed model (best model by default)
            B_est_map = map_estimator.gradient_ascent_results[0]['B_est']

            # Get error for MAP estimate
            map_error_norm = np.linalg.norm(
                np.subtract(
                    B_est_map,
                    self.execution_config['general']['B']
                )
            )/np.linalg.norm(self.execution_config['general']['B'])

            # Get estimated source
            s_est_map = B_est_map@x

            # Get error for estimated source
            s_est_error_norm = np.linalg.norm(
                np.subtract(
                    s_est_map,
                    s
                )
            )/np.linalg.norm(s)

            # Get execution logs
            logs = map_estimator.gradient_ascent_results[0]['logs']

            return B_est_map, map_error_norm, s_est_map, s_est_error_norm, logs
        
        def __parse_posteriori_grid(
            posteriori_grid,
            s,
            x
        ):
            
            # Get basis for symmetric space
            symmetric_basis = np.array([
                [0, 1],
                [1, 0]
            ])

            # Get basis for skew-symmetric space
            skew_symmetric_basis = np.array([
                [0, -1],
                [1, 0]
            ])

            # Get maximum u and v
            u_max = posteriori_grid.u_max
            v_max = posteriori_grid.v_max

            # Get results for posteriori grid
            B_est_posteriori = self.execution_config['general']['B'] + u_max*symmetric_basis + v_max*skew_symmetric_basis
            
            # Get recovered sources
            s_est_posteriori = B_est_posteriori@x

            # Get error for posteriori estimate
            posteriori_error_norm = np.linalg.norm(
                np.subtract(
                    B_est_posteriori,
                    self.execution_config['general']['B']
                )
            )/np.linalg.norm(self.execution_config['general']['B'])

            # Get error for estimated source
            s_est_error_norm = np.linalg.norm(
                np.subtract(
                    s_est_posteriori,
                    s
                )
            )/np.linalg.norm(s)

            return B_est_posteriori, posteriori_error_norm, s_est_posteriori, s_est_error_norm, u_max, v_max

        def __parse_realization(
            realization_name
        ):
            
            # Read raw results for realization
            realization_dir = self.experiment_dir / realization_name
            with (realization_dir/'results_raw.pkl').open('rb') as f:
                results_raw = dill.load(f)

            # Test cases
            test_cases = list(
                self.execution_config['sim']['test_cases'].keys()
            )

            # Initialize dictionary to store parsed results
            parsed_results = {}
            signals = {}
            for t in test_cases:
                parsed_results[t] = {
                    'realization': realization_name,
                    'mmse': {}, 
                    'map': {}, 
                    'posteriori_grid': {}
                }
                signals[t] = {
                    's': None,
                    'x': None,
                }
        
            # Iterate through test cases and parse objects
            for test_case, realizations_results in parsed_results.items():
                # Get objects
                mmse_estimator=results_raw['results'][test_case]['mmse_estimator']
                map_estimator=results_raw['results'][test_case]['map_estimator']
                posteriori_grid=results_raw['results'][test_case]['posteriori_grid']
                
                # Get signals
                s = results_raw[self.test_case_source_map[test_case]]['s']
                x = results_raw[self.test_case_source_map[test_case]]['x']

                # Save signals
                signals[test_case]['s'] = s
                signals[test_case]['x'] = x

                # Parse mmse
                B_est, B_error_norm, s_est, s_error_norm, diagnostics, converged, samples = __parse_mmse(
                    mmse_estimator=mmse_estimator,
                    s=s,
                    x=x
                )
                parsed_results[test_case]['mmse']['B_estimates'] = B_est
                parsed_results[test_case]['mmse']['B_errors'] = B_error_norm
                parsed_results[test_case]['mmse']['s_estimates'] = s_est
                parsed_results[test_case]['mmse']['s_errors'] = s_error_norm
                parsed_results[test_case]['mmse']['diagnostics'] = diagnostics
                parsed_results[test_case]['mmse']['converged'] = converged
                parsed_results[test_case]['mmse']['samples'] = samples
            
                # Parse map
                B_est, B_error_norm, s_est, s_error_norm, logs = __parse_map(
                    map_estimator=map_estimator,
                    s=s,
                    x=x
                )
                parsed_results[test_case]['map']['B_estimates'] = B_est
                parsed_results[test_case]['map']['B_errors'] = B_error_norm
                parsed_results[test_case]['map']['s_estimates'] = s_est
                parsed_results[test_case]['map']['s_errors'] = s_error_norm
                parsed_results[test_case]['map']['logs'] = logs

                # Parse posteriori grid objects
                # Save u_vec and v_vec
                parsed_results[test_case]['posteriori_grid']['u_vec'] = posteriori_grid.u_vec
                parsed_results[test_case]['posteriori_grid']['v_vec'] = posteriori_grid.v_vec
                # Parse posteriori grid
                parsed_results[test_case]['posteriori_grid']['grids'] = posteriori_grid.posterior_grid
                B_est, B_error_norm, s_est, s_error_norm, u_max, v_max = __parse_posteriori_grid(
                    posteriori_grid=posteriori_grid,
                    s=s,
                    x=x
                )
                parsed_results[test_case]['posteriori_grid']['maximums'] = (u_max, v_max)
                parsed_results[test_case]['posteriori_grid']['B_estimates'] = B_est
                parsed_results[test_case]['posteriori_grid']['B_errors'] = B_error_norm
                parsed_results[test_case]['posteriori_grid']['s_estimates'] = s_est
                parsed_results[test_case]['posteriori_grid']['s_errors'] = s_error_norm
            
            # Delete residual objects
            del mmse_estimator
            del map_estimator
            del posteriori_grid

            self.signals = signals

            return parsed_results
        
        def __format_results(
            results
        ):
            # Test cases
            test_cases = list(
                self.execution_config['sim']['test_cases'].keys()
            )
            # Iterate in test cases and results and create parsed dict with
            # test case as first dimension
            parsed_results = {
                t: {
                    'realization': [],
                    'mmse': {}, 
                    'map': {}, 
                    'posteriori_grid': {}
                } for t in test_cases
            }
            
            for t in test_cases:
                # Initialize parsed fields
                parsed_results[t]['mmse']['B_estimates'] = []
                parsed_results[t]['mmse']['B_errors'] = []
                parsed_results[t]['mmse']['s_estimates'] = []
                parsed_results[t]['mmse']['s_errors'] = []
                parsed_results[t]['mmse']['diagnostics'] = []
                parsed_results[t]['mmse']['converged'] = []
                parsed_results[t]['mmse']['samples'] = []
                parsed_results[t]['map']['B_estimates'] = []
                parsed_results[t]['map']['B_errors'] = []
                parsed_results[t]['map']['s_estimates'] = []
                parsed_results[t]['map']['s_errors'] = []
                parsed_results[t]['map']['logs'] = []
                parsed_results[t]['posteriori_grid']['grids'] = []
                parsed_results[t]['posteriori_grid']['maximums'] = []
                parsed_results[t]['posteriori_grid']['B_estimates'] = []
                parsed_results[t]['posteriori_grid']['B_errors'] = []
                parsed_results[t]['posteriori_grid']['s_estimates'] = []
                parsed_results[t]['posteriori_grid']['s_errors'] = []
                # Iterate in results and save to parsed_results
                for r in results:
                    parsed_results[t]['realization'].append(r[t]['realization'])
                    parsed_results[t]['mmse']['B_estimates'].append(r[t]['mmse']['B_estimates'])
                    parsed_results[t]['mmse']['B_errors'].append(r[t]['mmse']['B_errors'])
                    parsed_results[t]['mmse']['s_estimates'].append(r[t]['mmse']['s_estimates'])
                    parsed_results[t]['mmse']['s_errors'].append(r[t]['mmse']['s_errors'])
                    parsed_results[t]['mmse']['diagnostics'].append(r[t]['mmse']['diagnostics'])
                    parsed_results[t]['mmse']['converged'].append(r[t]['mmse']['converged'])
                    parsed_results[t]['mmse']['samples'].append(r[t]['mmse']['samples'])
                    parsed_results[t]['map']['B_estimates'].append(r[t]['map']['B_estimates'])
                    parsed_results[t]['map']['B_errors'].append(r[t]['map']['B_errors'])
                    parsed_results[t]['map']['s_estimates'].append(r[t]['map']['s_estimates'])
                    parsed_results[t]['map']['s_errors'].append(r[t]['map']['s_errors'])
                    parsed_results[t]['map']['logs'].append(r[t]['map']['logs'])
                    parsed_results[t]['posteriori_grid']['grids'].append(r[t]['posteriori_grid']['grids'])
                    parsed_results[t]['posteriori_grid']['maximums'].append(r[t]['posteriori_grid']['maximums'])
                    parsed_results[t]['posteriori_grid']['B_estimates'].append(r[t]['posteriori_grid']['B_estimates'])
                    parsed_results[t]['posteriori_grid']['B_errors'].append(r[t]['posteriori_grid']['B_errors'])
                    parsed_results[t]['posteriori_grid']['s_estimates'].append(r[t]['posteriori_grid']['s_estimates'])
                    parsed_results[t]['posteriori_grid']['s_errors'].append(r[t]['posteriori_grid']['s_errors'])
                
            # Get individual estimates
            for test_case, _ in parsed_results.items():
                # Iterate matrix indices to retrieve mmse and map estimates for individual coefficients
                it_shape = (
                    self.execution_config['general']['n_sources'],
                    self.execution_config['general']['n_sources']
                )
                for i, j in np.ndindex(it_shape):
                    # Get mmse
                    parsed_results[test_case]['mmse'][
                        'b{}{}_estimates'.format(
                            i+1, j+1
                        )
                    ] = np.array(parsed_results[test_case]['mmse']['B_estimates'])[:, i, j]
                    
                    
                    # Get map
                    parsed_results[test_case]['map'][
                        'b{}{}_estimates'.format(
                            i+1, j+1
                        )
                    ] = np.array(parsed_results[test_case]['map']['B_estimates'])[:, i, j]

                # Get average grid and maximum points for posteriori grid
                # u
                parsed_results[test_case]['posteriori_grid']['u_max'] = np.array(
                    parsed_results[test_case]['posteriori_grid']['maximums']
                )[:, 0]
                # v
                parsed_results[test_case]['posteriori_grid']['v_max'] = np.array(
                    parsed_results[test_case]['posteriori_grid']['maximums']
                )[:, -1]
                # average grid
                parsed_results[test_case]['posteriori_grid']['average_grid'] = np.mean(
                    a = parsed_results[test_case]['posteriori_grid']['grids'],
                    axis=0
                )
            
            # Save to attribute
            self.parsed_results=parsed_results

        

        # Iterate over realizations and parse
        realizations_results = []
        for r in tqdm(self.finished_realizations):
            realizations_results.append( 
                __parse_realization(
                    realization_name=r
                )
            )

        # Put results in final parsed format
        __format_results(
            results=realizations_results
        )
            
            
    def parse(
        self,
        verbose=False
    ):
        
        # Read execution configurations
        self.__read_execution_configs()

        # Read raw results for realizations
        self.__get_finished_realizations(verbose=verbose)

        # Parse realizations results
        self.__parse_raw_results(verbose=verbose)


class ContinuousExperimentExecutor:

    def __init__(
        self,
        cfg,
        initialize=True
    ):
        if initialize:
            # Save cfg as attribute
            self.cfg=cfg
            
            # Get source and mixture signals for all realizations
            self.__initialize_signals()

            # Get source model and prior distributions
            self.__get_execution_objects()

            # Get execution functions for test cases
            self.__get_exec_fns()

            # Get initial starting points for Gradient Ascent and MCMC
            self.__get_initial_conditions()

            # Save cfg and signals
            self.__save_initializations()

        else:
            self.cfg=cfg

            self.__read_signals()

    
    def __initialize_signals(
        self
    ):
        
        # Parse config object
        A = self.cfg['general']['A']
        n_realizations = self.cfg['general']['n_realizations']
        n_sources = self.cfg['general']['n_sources']
        n_obs = self.cfg['general']['n_obs']

        # Initialize random seeds
        seeds = [x for x in range(n_realizations)]
        
        # Initialize dict to keep signals
        signals = {}
        
        # Iterate over realizations and generate signals
        for r in range(n_realizations):
            realization = {}
            # Generate sources for each specified configuration
            for s_name, s_obj in self.cfg['sources'].items():
                # Get source realization
                s = s_obj.get_realization(
                    nsources=n_sources,
                    nobs=n_obs,
                    seed=seeds[r]
                )
                # Get mixtures
                x = A@s
                # Save signals
                realization[s_name] = {
                    's': s,
                    'x': x
                }
                signals[str(r)] = realization

        self.signals=signals
        
    
    def __get_execution_objects(
        self
    ):
        # Enrich prior variations
        if len(self.cfg['prior_variation']) > 0:
            for variation, variation_params in self.cfg['prior_variation'].items():
                if len(variation_params)>0:
                    # Get target priors
                    self.cfg['prior_variation'][variation]['target_priors'] = variation_params['target_param_fn'](
                        target_params=variation_params['target_param_values']
                    )
                    # Instantiate sampler for each source-model (baseline model assumed to be equal to source)
                    self.cfg['prior_variation'][variation]['mcmc_samplers'] = {}
                    for s_name, s_obj in self.cfg['sources'].items():
                        # Instantiate MCMC sampler
                        self.cfg['prior_variation'][variation]['mcmc_samplers'][s_name] = MMSEBarkerMHEstimator(
                            n_samples=self.cfg['mcmc']['n_samples'],
                            source_pdf_fn=s_obj.get(),
                            prior_pdf_fn=variation_params['base_prior'].get(),
                            exploration_var=self.cfg['mcmc']['exploration_var'],
                            parallel_chains=self.cfg['mcmc']['parallel_chains'],
                            max_it=self.cfg['mcmc']['max_it'],
                            R_hat_thresh=self.cfg['mcmc']['R_hat_thresh'],
                            R_hat_evaluation_step=self.cfg['mcmc']['R_hat_evaluation_step'],
                            R_hat_persistance=self.cfg['mcmc']['R_hat_persistance'],
                            R_hat_minimum_burn_in=self.cfg['mcmc']['R_hat_minimum_burn_in'],
                            auto_adjust_exploration_var=self.cfg['mcmc']['auto_adjust_exploration_var'],
                            progress_bar=self.cfg['mcmc']['progress_bar'],
                            target_accept_prob=self.cfg['mcmc']['target_accept_prob']
                        )
         # Enrich likelihood variations           
        if len(self.cfg['likelihood_variation']) > 0:
            pass
        

    def __get_exec_fns(
        self
    ):
        def __run_mcmc(
            mmse_estimator,
            initial_conditions,
            x
        ):
            """
                Runs mcmc for one realization
            """
            # run mcmc
            samples, diagnostics, logs = mmse_estimator.fit(
                x=x,
                initial_condition=initial_conditions
            )

            return samples, diagnostics, logs

        
        # Iterate over prior variations and obtain functions to be executed for each realization
        if len(self.cfg['prior_variation']) > 0:
            for variation, variation_params in self.cfg['prior_variation'].items():
                if len(variation_params)>0:
                    self.cfg['prior_variation'][variation]['base_log_posteriors'] = {}
                    self.cfg['prior_variation'][variation]['mcmc_execution_functions'] = {}
                    for s_name, s_obj in self.cfg['sources'].items():
                        # Instantiate log posterior function
                        self.cfg['prior_variation'][variation]['base_log_posteriors'][s_name] = PosteriorUtilities.get_log_posterior_fn(
                            source_pdf_fn=s_obj.get(),
                            prior_pdf_fn=variation_params['base_prior'].get(),
                        )

                        # MCMC execution function - baseline posteriors
                        self.cfg['prior_variation'][variation]['mcmc_execution_functions'][s_name] = functools.partial(
                            __run_mcmc,
                            self.cfg['prior_variation'][variation]['mcmc_samplers'][s_name]
                        )

        # Iterate over likelihood variations and obtain functions to be executed for each realization
        if len(self.cfg['likelihood_variation']) > 0:
            pass


    def __get_initial_conditions(
        self
    ):
        self.initial_conditions = np.array([
            np.add(
                self.cfg['mcmc']['starting_distribution'](),
                self.cfg['general']['B']
            ) for c in range(self.cfg['mcmc']['parallel_chains'])
        ])
        

    def __save_initializations(
        self
    ):
        # Get experiment_dir
        experiment_dir = self.cfg['general']['experiment_dir']
        
        # Iterate over realizations and save signals
        for r, res in self.signals.items():
            # Get dir for saving realization results
            realization_dir = experiment_dir / r

            # Save results
            with (realization_dir/'signals.pkl').open('wb') as f:
                dill.dump(res, f)

        # Save updated config file
        with (experiment_dir/'execution_config.pkl').open('wb') as f:
                dill.dump(self.cfg, f)


    def __read_signals(
        self
    ):
        
        # Get realization iterable
        realizations = range(self.cfg['general']['n_realizations'])

        # Get experiment dir
        experiment_dir = self.cfg['general']['experiment_dir']

        # Read signals
        signals = {}
        for r in realizations:
            # Get dir for reading realization results
            realization_dir = experiment_dir / str(r)

            # Read signals
            with (realization_dir/'signals.pkl').open('rb') as f:
                signals[str(r)] = dill.load(f)
            
        self.signals=signals

    def __get_iterable(
            self
    ):
        # Get realization iterable
        realizations = range(self.cfg['general']['n_realizations'])

         # Get experiment dir
        experiment_dir = self.cfg['general']['experiment_dir']

        # Iterate and identify finished realizations
        finished_realizations = []
        for r in realizations:
            
            # Get dir for reading realization results
            realization_dir = experiment_dir / str(r)

            # Read success flag
            with (realization_dir/'success_flag.pkl').open('rb') as f:
                success_flag = dill.load(f)

            if success_flag=='SUCCESS':
                finished_realizations.append(str(r))
        
        
        # Get realizations to run
        unfinished_realizations_signals = {
            k:v for k, v in self.signals.items() if k not in finished_realizations
        }

        # Print status
        print('#'*100)
        print('Total realizations: {}'.format(self.cfg['general']['n_realizations']))
        print('Finished realizations: {}'.format(len(finished_realizations)))
        print('Unfinished realizations: {}'.format(len(unfinished_realizations_signals.keys())))
        print('#'*100)

        return unfinished_realizations_signals.items()
    

    def run(
        self
    ):

        def __run_realization(
            cfg,
            realization_info,
        ):
            """
                Runs experiment for one realization
            """
            
            # Parse realization info
            realization_name = realization_info[0]
            realization_cfgs = realization_info[-1]
            
            # Log realization
            logger.info('Running realization {}.'.format(realization_name))

            # Iterate over specified analyses and perform operations
            realization_results = {}
            # Iterate over prior variations and obtain functions to be executed for each realization
            if len(self.cfg['prior_variation']) > 0:
                realization_results['prior_variation'] = {}
                for variation, variation_params in self.cfg['prior_variation'].items():
                    # Log realization
                    logger.info('Running prior variation: {}.'.format(variation))
                    realization_results['prior_variation'][variation] = {}
                    if len(variation_params)>0:
                        for s_name, s_obj in self.cfg['sources'].items():
                            logger.info('Running analysis for source: {}.'.format(s_name))
                            # Obtain sources and mixtures
                            s = realization_cfgs[s_name]['s']
                            x = realization_cfgs[s_name]['x']

                            # Get function for executing MCMC
                            mcmc_execution_function = variation_params['mcmc_execution_functions'][s_name]

                            # Execute MCMC
                            start_mcmc = time.time_ns()
                            samples, diagnostics, logs = mcmc_execution_function(
                                initial_conditions=self.initial_conditions,
                                x=x
                            )
                            end_mcmc = time.time_ns()
                            logger.info('Time spent in MCMC (seconds): {}'.format(round((end_mcmc-start_mcmc)/1E9, 2)))

                            # Execute importance sampling
                            start_is = time.time_ns()
                            is_results = ImportanceSamplingEstimator.run(
                                samples=self.cfg['prior_variation'][variation]['mcmc_samplers'][s_name].mcmc_results['samples'],
                                diagnostics=diagnostics,
                                baseline_prior=variation_params['base_prior'],
                                x=x,
                                target_priors=variation_params['target_priors'],
                                prior_control_params=variation_params['target_param_values'],
                                baseline_source_model=s_obj,
                                target_source_models=None,
                                source_model_control_params=None
                            )
                            end_is = time.time_ns()
                            logger.info('Time spent in IS (seconds): {}'.format(round((end_is-start_is)/1E9, 2)))

                            # Updates realization cfgs
                            realization_results['prior_variation'][variation][s_name] = {
                                's': s,
                                'x': x,
                                'merged_baseline_samples': self.cfg['prior_variation'][variation]['mcmc_samplers'][s_name].mcmc_results['samples'],
                                'raw_baseline_samples': samples,
                                'mcmc':{
                                    'diagnostics': diagnostics,
                                    'logs': logs
                                }
                            }
                            realization_results['prior_variation'][variation][s_name].update(is_results['prior_variation'])

            # Saves to realization_cfgs
            realization_cfgs['results'] = realization_results

            # Save results
            start_saving = time.time_ns()
            realization_dir = cfg['general']['experiment_dir']  / realization_name
            with (realization_dir/'results_raw.pkl').open('wb') as f:
                dill.dump(realization_cfgs, f)
            end_saving = time.time_ns()
            logger.info('Time spent saving (seconds): {}'.format(round((end_saving-start_saving)/1E9, 2)))

            with (realization_dir/'success_flag.pkl').open('wb') as f:
                dill.dump('SUCCESS', f)

            # Delete as results have already been written
            del realization_cfgs['results']
            del realization_results
            del mcmc_execution_function
            sampler = self.cfg['prior_variation'][variation]['mcmc_samplers'][s_name]
            sampler.samples = None
            sampler.logs = None
            sampler.diagnostics = None
            sampler.mcmc_results = None
            del samples, diagnostics, logs
            gc.collect()

            logger.info("JAX live arrays: %d", len(jax.live_arrays()))
            # find_retainers()
            
            # Clear jax caches
            jax.clear_caches()



        # Create execution functions for each realization
        realization_fn = functools.partial(
            __run_realization,
            self.cfg
        )
        
        # Run experiment for each realization
        iterable = self.__get_iterable()
        if self.cfg['general']['realizations_progress_bar']:
            print('Executando experimento')
            iterable = tqdm(iterable)
        for elem in iterable:
            try:
                realization_fn(elem)
            except Exception:
                logger.exception(
                    "Realization %s failed",
                    elem[0]
                )
                raise
