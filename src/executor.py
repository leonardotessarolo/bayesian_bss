import os
import pandas as pd
import numpy as np
import pathos
import functools
import dill
import jax.numpy as jnp
from .estimator import BayesianEstimators
from .contour_line import PosteriorContourLines
from .utilities import PosteriorUtilities
from tqdm import tqdm 
import time

class ExperimentExecutor:

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

            ########################################
            # Configs for initialization execution #
            ########################################

            # # Configurations for MCMC sampling and Gradient Ascent optimization
            # initialization_mcmc_configs, initialization_grad_asc_configs =  BayesianEstimators.generate_configs(
            #     source_pdf=source_model_pdf,
            #     source_pdf_derivative=source_model_pdf_derivative,
            #     prior_pdf=prior_pdf,
            #     prior_pdf_derivative=prior_pdf_derivative,
            #     normalize_posterior=self.cfg['general']['normalize_posterior'],
            #     n_samples_mcmc=self.cfg['mcmc']['n_samples'],
            #     exploration_var_mcmc=self.cfg['mcmc']['exploration_var'],
            #     parallel_chains_mcmc=self.cfg['mcmc']['parallel_chains'],
            #     max_it_mcmc=self.cfg['mcmc']['max_it'],
            #     R_hat_thresh_mcmc=self.cfg['mcmc']['R_hat_thresh'],
            #     R_hat_evaluation_step_mcmc=self.cfg['mcmc']['R_hat_evaluation_step'],
            #     R_hat_persistance_mcmc=self.cfg['mcmc']['R_hat_persistance'],
            #     R_hat_minimum_burn_in_mcmc=self.cfg['mcmc']['R_hat_minimum_burn_in'],
            #     learning_rate_grad_asc=self.cfg['initial_mode_seeking']['learning_rate'],
            #     stopping_thresh_grad_asc=self.cfg['initial_mode_seeking']['stopping_thresh'],
            #     max_it_grad_asc=self.cfg['initial_mode_seeking']['max_it'],
            #     stopping_criterion_persistance_its_grad_asc=self.cfg['initial_mode_seeking']['stopping_criterion_persistance_its']
            # )

            # # Set configuration in estimators to run both
            # initialization_mcmc_configs['run_mcmc'] = False
            # initialization_grad_asc_configs['run_grad_asc'] = True

            # # Save MAP and MCMC configs
            # self.cfg['sim']['test_cases'][test_case]['initialization_mcmc_configs']=initialization_mcmc_configs
            # self.cfg['sim']['test_cases'][test_case]['initialization_grad_asc_configs']=initialization_grad_asc_configs

            ##################################
            # Configs for standard execution #
            ##################################

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
                print('Executando inicialização')
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
                        random_draw = jnp.asarray([
                            self.cfg['mcmc']['starting_distribution']() for _ in range(2*draw_points)
                        ])

                        # Get posteriors for points
                        posteriors = jnp.array([
                            self.cfg['sim']['test_cases'][test_case]['log_posterior_fn'](
                                x=x,
                                B=B_0
                            ) for B_0 in random_draw
                        ])

                        # Finite idxs
                        finite_idxs = jnp.where(np.array(posteriors) > -np.inf)[0]
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
                mmse_initial_condition = jnp.asarray([
                    self.cfg['general']['B'] + self.cfg['mcmc']['starting_distribution']() for _ in range(self.cfg['mcmc']['parallel_chains'])
                ])

                # Save to dict
                test_case_cfgs['initial_conditions'] = {
                    'map': [map_initial_condition],
                    'mmse': mmse_initial_condition,
                }

                # print(test_case_cfgs['initial_conditions'])

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
            
            # Iterate over test cases and execute experiment
            realization_results = {}
            iterable=self.cfg['sim']['test_cases'].items()
            if self.cfg['general']['test_cases_progress_bar']:
                iterable=tqdm(iterable)
            for test_case, test_case_cfgs in iterable:
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
                print('Time spent in grid: {}'.format((end_grid-start_grid)/1E9)) 

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
            print('Start saving')
            realization_dir = cfg['general']['experiment_dir']  / realization_name
            with (realization_dir/'results_raw.pkl').open('wb') as f:
                dill.dump(realization_cfgs, f)
            print('End saving')

            with (realization_dir/'success_flag.pkl').open('wb') as f:
                dill.dump('SUCCESS', f)


        # Create execution functions for each realization
        realization_fn = functools.partial(
            __run_realization,
            self.cfg
        )
        
        # Run experiment for each realization
        
        iterable = self.__get_iterable()
        # if (self.cfg['general']['n_workers'] > 1) and (self.cfg['general']['n_realizations'] > 1):
        #     with pathos.pools.ProcessPool(self.cfg['general']['n_workers']) as p:
        #         p.map(
        #             realization_fn, 
        #             iterable
        #         )
        # else:
        if self.cfg['general']['realizations_progress_bar']:
            print('Executando experimento')
            iterable = tqdm(iterable)
        for elem in iterable:
            realization_fn(elem)

        # Debug
        # iterable = self.__get_iterable()
        # _ = [
        #     realization_fn(elem) for elem in iterable
        # ]



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



class ExperimentParser:
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

    def __read_raw_results(
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
        realizations_results = {}
        finished_realizations = []
        non_executed_realizations = []
        for r in tqdm(realizations):
            # Get directory for realization
            realization_dir = self.experiment_dir / r
            try:
                # Read raw results
                with (realization_dir/'results_raw.pkl').open('rb') as f:
                        realizations_results[r] = dill.load(f)
            except:
                # non_executed_realizations.append(r)
                pass

            # Read success flag
            with (realization_dir/'success_flag.pkl').open('rb') as f:
                success_flag = dill.load(f)

            if success_flag=='SUCCESS':
                finished_realizations.append(str(r))
            else:
                non_executed_realizations.append(str(r))
        
        # Save to object attributes
        self.realizations_results = realizations_results
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

            return B_est_mmse, mmse_error_norm, s_est_mmse, s_est_error_norm, diagnostics, converged

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

            return B_est_map, map_error_norm, s_est_map, s_est_error_norm
        
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

        # Test cases
        test_cases = list(
            self.execution_config['sim']['test_cases'].keys()
        )

        # Initialize dictionary to store parsed results
        parsed_results = {}
        for t in test_cases:
            parsed_results[t] = {
                'realization': [],
                'mmse': {'object': []}, 
                'map': {'object': []}, 
                'posteriori_grid': {'object': []}
            }
        
        # Iterate through realizations results and get objects for each test case
        for r, results in self.realizations_results.items():
            # Iterate through test cases
            
            for test_case in parsed_results.keys():
                # Log realization
                parsed_results[test_case]['realization'].append(r)

                # Read mmse estimator
                parsed_results[test_case]['mmse']['object'].append(results['results'][test_case]['mmse_estimator'])

                # Read map estimator
                parsed_results[test_case]['map']['object'].append(results['results'][test_case]['map_estimator'])

                # Read posteriori_grid
                parsed_results[test_case]['posteriori_grid']['object'].append(results['results'][test_case]['posteriori_grid'])
                
        # Iterate through test cases and parse objects
        for test_case, realizations_results in tqdm(parsed_results.items()):
            # Initialize parsed fields
            parsed_results[test_case]['mmse']['B_estimates'] = []
            parsed_results[test_case]['mmse']['B_errors'] = []
            parsed_results[test_case]['mmse']['s_estimates'] = []
            parsed_results[test_case]['mmse']['s_errors'] = []
            parsed_results[test_case]['mmse']['diagnostics'] = []
            parsed_results[test_case]['mmse']['converged'] = []
            parsed_results[test_case]['map']['B_estimates'] = []
            parsed_results[test_case]['map']['B_errors'] = []
            parsed_results[test_case]['map']['s_estimates'] = []
            parsed_results[test_case]['map']['s_errors'] = []
            parsed_results[test_case]['posteriori_grid']['grids'] = []
            parsed_results[test_case]['posteriori_grid']['maximums'] = []
            parsed_results[test_case]['posteriori_grid']['B_estimates'] = []
            parsed_results[test_case]['posteriori_grid']['B_errors'] = []
            parsed_results[test_case]['posteriori_grid']['s_estimates'] = []
            parsed_results[test_case]['posteriori_grid']['s_errors'] = []


            
            # Parse estimator objects
            for r, mmse_estimator, map_estimator, posteriori_grid in zip(
                realizations_results['realization'],
                realizations_results['mmse']['object'],
                realizations_results['map']['object'],
                realizations_results['posteriori_grid']['object']
            ):
                # Get signals
                s = self.realizations_results[r][self.test_case_source_map[test_case]]['s']
                x = self.realizations_results[r][self.test_case_source_map[test_case]]['x']

                # Parse mmse
                B_est, B_error_norm, s_est, s_error_norm, diagnostics, converged = __parse_mmse(
                    mmse_estimator=mmse_estimator,
                    s=s,
                    x=x
                )
                parsed_results[test_case]['mmse']['B_estimates'].append(B_est)
                parsed_results[test_case]['mmse']['B_errors'].append(B_error_norm)
                parsed_results[test_case]['mmse']['s_estimates'].append(s_est)
                parsed_results[test_case]['mmse']['s_errors'].append(s_error_norm)
                parsed_results[test_case]['mmse']['diagnostics'].append(diagnostics)
                parsed_results[test_case]['mmse']['converged'].append(converged)
            
                # Parse map
                B_est, B_error_norm, s_est, s_error_norm = __parse_map(
                    map_estimator=map_estimator,
                    s=s,
                    x=x
                )
                parsed_results[test_case]['map']['B_estimates'].append(B_est)
                parsed_results[test_case]['map']['B_errors'].append(B_error_norm)
                parsed_results[test_case]['map']['s_estimates'].append(s_est)
                parsed_results[test_case]['map']['s_errors'].append(s_error_norm)

                # Parse posteriori grid objects
                # Save u_vec and v_vec
                parsed_results[test_case]['posteriori_grid']['u_vec'] = posteriori_grid.u_vec
                parsed_results[test_case]['posteriori_grid']['v_vec'] = posteriori_grid.v_vec
                # Parse posteriori grid
                parsed_results[test_case]['posteriori_grid']['grids'].append(posteriori_grid.posterior_grid)
                B_est, B_error_norm, s_est, s_error_norm, u_max, v_max = __parse_posteriori_grid(
                    posteriori_grid=posteriori_grid,
                    s=s,
                    x=x
                )
                parsed_results[test_case]['posteriori_grid']['maximums'].append((u_max, v_max))
                parsed_results[test_case]['posteriori_grid']['B_estimates'].append(B_est)
                parsed_results[test_case]['posteriori_grid']['B_errors'].append(B_error_norm)
                parsed_results[test_case]['posteriori_grid']['s_estimates'].append(s_est)
                parsed_results[test_case]['posteriori_grid']['s_errors'].append(s_error_norm)
            
        # Get individual estimates
        for test_case, realizations_results in parsed_results.items():
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
        self.parsed_results = parsed_results
            
            
    def parse(
        self,
        verbose=False
    ):
        
        # Read execution configurations
        self.__read_execution_configs()

        # Read raw results for realizations
        self.__read_raw_results(verbose=verbose)

        # Parse realizations results
        self.__parse_raw_results(verbose=verbose)