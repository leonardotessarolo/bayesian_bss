import os
import pandas as pd
import numpy as np
import pathos
import functools
import dill
from .estimator import BayesianEstimators
from .contour_line import PosteriorContourLines

class ExperimentExecutor:

    # TODO: repetir simulações com matriz quase-singular

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
            self.__get_test_case_distributions()

            # Get execution functions for test cases
            self.__get_exec_fns()

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
                burn_in_mcmc=self.cfg['mcmc']['burn_in'],
                learning_rate_grad_asc=self.cfg['map']['learning_rate'],
                stopping_thresh_grad_asc=self.cfg['map']['stopping_thresh'],
                max_it_grad_asc=self.cfg['map']['max_it']
            )

            # Save MAP and MCMC configs
            self.cfg['sim']['test_cases'][test_case]['mcmc_configs']=mcmc_configs
            self.cfg['sim']['test_cases'][test_case]['grad_asc_configs']=grad_asc_configs

    def __get_exec_fns(
        self
    ):
        def __run_estimators(
            mcmc_configs,
            grad_asc_configs,
            initial_B,
            n_workers,
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
                initial_B=initial_B,
                n_jobs=n_workers
            )

            return mmse_estimator, map_estimator

        def __run_contour(
            u_lims,
            v_lims,
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
                central_point=(0,0),
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
            
            # Bayesian estimators execution function
            bayesian_estimators_fn = functools.partial(
                __run_estimators,
                test_case_cfgs['mcmc_configs'],
                test_case_cfgs['grad_asc_configs'],
                self.cfg['general']['initial_B'],
                self.cfg['general']['n_workers']
            )

            # Posteriori grid execution function
            posteriori_grid_fn = functools.partial(
                __run_contour,
                self.cfg['contour']['u_lims'],
                self.cfg['contour']['v_lims'],
                self.cfg['contour']['contour_grid_points'],
                test_case_cfgs['grad_asc_configs']['source_pdf'],
                test_case_cfgs['grad_asc_configs']['prior_pdf'],
                self.cfg['general']['A'],
                self.cfg['general']['n_obs'],
                self.cfg['general']['n_workers'],
            )

            # Save functions
            self.cfg['sim']['test_cases'][test_case]['bayesian_estimators_fn']=bayesian_estimators_fn
            self.cfg['sim']['test_cases'][test_case]['posteriori_grid_fn']=posteriori_grid_fn



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

    # def __save_results(
    #     self
    # ):

    #     # Iterate over realizations and save
    #     for r, res in self.results.items():
    #         # Get dir for saving realization results
    #         realization_dir = self.cfg['general']['experiment_dir'] / r

    #         # Save results
    #         with (realization_dir/'results_raw.pkl').open('wb') as f:
    #             dill.dump(res, f)
    

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
            for test_case, test_case_cfgs in self.cfg['sim']['test_cases'].items():
                # Source for test case
                test_case_source = test_case_cfgs['source']

                # Obtain sources and mixtures
                s = realization_cfgs[test_case_source]['s']
                x = realization_cfgs[test_case_source]['x']
                
                # Retrieve function for executing bayesian estimators 
                bayesian_estimators_fn = test_case_cfgs['bayesian_estimators_fn']

                # Run bayesian estimators
                mmse_estimator, map_estimator = bayesian_estimators_fn(
                    s=s,
                    x=x
                )
                
                # Retrieve function for executing posteriori grid calculation
                posteriori_grid_fn = test_case_cfgs['posteriori_grid_fn']
                
                # Run posteriori grid calculation
                posteriori_grid = posteriori_grid_fn(x=x)
                
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
            realization_dir = cfg['general']['experiment_dir']  / realization_name
            with (realization_dir/'results_raw.pkl').open('wb') as f:
                dill.dump(realization_cfgs, f)

            with (realization_dir/'success_flag.pkl').open('wb') as f:
                dill.dump('SUCCESS', f)
            

            # return (realization_name, realization_cfgs)
                

        # Create execution functions for each realization
        realization_fn = functools.partial(
            __run_realization,
            self.cfg
        )
        
        # Run experiment for each realization
        # iterable = self.signals.items()
        iterable = self.__get_iterable()
        with pathos.pools.ProcessPool(self.cfg['general']['n_workers']) as p:
            # results=p.map(
            #     realization_fn, 
            #     iterable
            # )
            p.map(
                realization_fn, 
                iterable
            )

        
        # DEBUG
        # results = []
        # for realization_info in self.signals.items():
            
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
        realizations_results = {}
        non_executed_realizations = []
        for r in realizations:
            # Get directory for realization
            realization_dir = self.experiment_dir / r
            try:
                # Read raw results
                with (realization_dir/'results_raw.pkl').open('rb') as f:
                        realizations_results[r] = dill.load(f)
            except:
                non_executed_realizations.append(r)
        
        # Save to object attributes
        self.realizations_results = realizations_results
        self.non_executed_realizations=non_executed_realizations
        
        if verbose:
            print('#'*100)
            print('Total of {} realizations with executed results'.format(len(realizations_results.keys())))
            print('-'*100)
            print('Total of {} realizations without executed results'.format(len(non_executed_realizations)))
            print('#'*100)

    def __parse_raw_results(
        self,
        verbose
    ):
        def __parse_mmse(
            mmse_estimator,
            A
        ):
            # Get MMSE estimate
            B_est_mmse = mmse_estimator.mcmc_results[0]['B_est']
            # s_est_mmse = B_est_mmse@test_case_results['x']

            # Get error for MMSE estimate
            mmse_error_norm = np.linalg.norm(
            np.subtract(
                    B_est_mmse,
                    np.linalg.inv(A)
                )
            )/np.linalg.norm(np.linalg.inv(A))

            return B_est_mmse, mmse_error_norm

        def __parse_map(
            map_estimator,
            A
        ):
            # Get results for analyzed model (best model by default)
            B_est_map = map_estimator.gradient_ascent_results[0]['B_est']
            # s_est_map = B_est_map@test_case_results['x']

            # Get error for MAP estimate
            map_error_norm = np.linalg.norm(
            np.subtract(
                    B_est_map,
                    np.linalg.inv(A)
                )
            )/np.linalg.norm(np.linalg.inv(A))

            return B_est_map, map_error_norm
            
        # Test cases
        test_cases = [
            'i', 'ii', 'iii', 'iv',
            'v', 'vi', 'vii', 'viii',
            'ix', 'x', 'xi', 'xii'
        ]

        # Initialize dictionary to store parsed results
        parsed_results = {}
        for t in test_cases:
            parsed_results[t] = {
                'mmse': {'object': []}, 
                'map': {'object': []}, 
                'posteriori_grid': {'object': []}
            }

        # Iterate through realizations results and get objects for each test case
        for r, results in self.realizations_results.items():
            # Iterate through test cases
            for test_case in parsed_results.keys():
                # Read mmse estimator
                parsed_results[test_case]['mmse']['object'].append(results['results'][test_case]['mmse_estimator'])

                # Read map estimator
                parsed_results[test_case]['map']['object'].append(results['results'][test_case]['map_estimator'])

                # Read posteriori_grid
                parsed_results[test_case]['posteriori_grid']['object'].append(results['results'][test_case]['posteriori_grid'])
                
        # Iterate through test cases and parse objects
        for test_case, realizations_results in parsed_results.items():
            # Initialize parsed fields
            parsed_results[test_case]['mmse']['estimates'] = []
            parsed_results[test_case]['mmse']['errors'] = []
            parsed_results[test_case]['map']['estimates'] = []
            parsed_results[test_case]['map']['errors'] = []
            parsed_results[test_case]['posteriori_grid']['grids'] = []
            parsed_results[test_case]['posteriori_grid']['maximums'] = []
            
            # Parse mmse estimator objects
            for mmse_estimator in realizations_results['mmse']['object']:
                # Parse mmse
                B_est, error_norm = __parse_mmse(
                    mmse_estimator=mmse_estimator,
                    A=self.execution_config['general']['A']
                )
                parsed_results[test_case]['mmse']['estimates'].append(B_est)
                parsed_results[test_case]['mmse']['errors'].append(error_norm)
            
            # Parse mmse estimator objects
            for map_estimator in realizations_results['map']['object']:
                # Parse map
                B_est, error_norm = __parse_map(
                    map_estimator=map_estimator,
                    A=self.execution_config['general']['A']
                )
                parsed_results[test_case]['map']['estimates'].append(B_est)
                parsed_results[test_case]['map']['errors'].append(error_norm)

            # Parse posteriori grid objects
            for posteriori_grid in realizations_results['posteriori_grid']['object']:
                # Save u_vec and v_vec
                parsed_results[test_case]['posteriori_grid']['u_vec'] = posteriori_grid.u_vec
                parsed_results[test_case]['posteriori_grid']['v_vec'] = posteriori_grid.v_vec
                # Parse posteriori grid
                parsed_results[test_case]['posteriori_grid']['grids'].append(posteriori_grid.posterior_grid)
                parsed_results[test_case]['posteriori_grid']['maximums'].append(
                    (
                         posteriori_grid.u_vec[posteriori_grid.max_post_point[0]],
                        posteriori_grid.v_vec[posteriori_grid.max_post_point[-1]]
                    )
                )
            
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
                ] = np.array(parsed_results[test_case]['mmse']['estimates'])[:, i, j]
                
                # Get map
                parsed_results[test_case]['map'][
                    'b{}{}_estimates'.format(
                        i+1, j+1
                    )
                ] = np.array(parsed_results[test_case]['map']['estimates'])[:, i, j]

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