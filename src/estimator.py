import pandas as pd
import numpy as np
import functools
import jax.numpy as jnp
import jax
from numpyro.diagnostics import gelman_rubin, effective_sample_size
from numpyro.infer import MCMC, BarkerMH
import time

from .utilities import PosteriorUtilities

class BayesianEstimators:

    @staticmethod
    def generate_configs(
        source_pdf,
        source_pdf_derivative,
        prior_pdf,
        prior_pdf_derivative,
        normalize_posterior,
        n_samples_mcmc,
        exploration_var_mcmc,
        parallel_chains_mcmc,
        max_it_mcmc,
        R_hat_thresh_mcmc,
        R_hat_evaluation_step_mcmc,
        R_hat_persistance_mcmc,
        R_hat_minimum_burn_in_mcmc,
        auto_adjust_exploration_var_mcmc,
        progress_bar_mcmc,
        target_accept_prob_mcmc,
        learning_rate_grad_asc,
        stopping_thresh_grad_asc,
        max_it_grad_asc,
        stopping_criterion_persistance_its_grad_asc,
        is_natural_gradient_grad_asc=False
    ):

        # Create MCMC config
        mcmc_configs = {
            'n_samples': n_samples_mcmc,
            'source_pdf_fn': source_pdf,
            'prior_pdf_fn': prior_pdf,
            'exploration_var': exploration_var_mcmc,
            'parallel_chains': parallel_chains_mcmc,
            'max_it': max_it_mcmc,
            'R_hat_thresh': R_hat_thresh_mcmc,
            'R_hat_evaluation_step': R_hat_evaluation_step_mcmc,
            'R_hat_persistance': R_hat_persistance_mcmc,
            'R_hat_minimum_burn_in': R_hat_minimum_burn_in_mcmc,
            'auto_adjust_exploration_var':auto_adjust_exploration_var_mcmc,
            'progress_bar': progress_bar_mcmc,
            'target_accept_prob':target_accept_prob_mcmc
        }

        # Create Gradient Ascent config
        grad_asc_configs = {
            'learning_rate': learning_rate_grad_asc,
            'thresh': stopping_thresh_grad_asc,
            'max_it': max_it_grad_asc,
            'stopping_criterion_persistance_its': stopping_criterion_persistance_its_grad_asc,
            'source_pdf': source_pdf,
            'source_pdf_derivative': source_pdf_derivative,
            'prior_pdf': prior_pdf,
            'prior_pdf_derivative': prior_pdf_derivative,
            'normalize_posterior': normalize_posterior,
            'is_natural_gradient': is_natural_gradient_grad_asc
        }

        return mcmc_configs, grad_asc_configs
        
        
    @staticmethod
    def run(
        s,
        x,
        mmse_configs,
        map_configs,
        initial_conditions
    ):
        
        # Initialize MH estimator
        # mmse_estimator = MMSEMetropolisHastingsEstimator(
        #     n_samples=mmse_configs['n_samples'],
        #     source_pdf_fn=mmse_configs['source_pdf_fn'],
        #     prior_pdf_fn=mmse_configs['prior_pdf_fn'],
        #     exploration_var=mmse_configs['exploration_var'],
        #     parallel_chains=mmse_configs['parallel_chains'],
        #     max_it=mmse_configs['max_it'],
        #     R_hat_thresh=mmse_configs['R_hat_thresh'],
        #     R_hat_evaluation_step=mmse_configs['R_hat_evaluation_step'],
        #     R_hat_persistance=mmse_configs['R_hat_persistance'],
        #     R_hat_minimum_burn_in=mmse_configs['R_hat_minimum_burn_in']
        # )
        start_mmse = time.time_ns()
        mmse_estimator = MMSEBarkerMHEstimator(
            n_samples=mmse_configs['n_samples'],
            source_pdf_fn=mmse_configs['source_pdf_fn'],
            prior_pdf_fn=mmse_configs['prior_pdf_fn'],
            exploration_var=mmse_configs['exploration_var'],
            parallel_chains=mmse_configs['parallel_chains'],
            max_it=mmse_configs['max_it'],
            R_hat_thresh=mmse_configs['R_hat_thresh'],
            R_hat_evaluation_step=mmse_configs['R_hat_evaluation_step'],
            R_hat_persistance=mmse_configs['R_hat_persistance'],
            R_hat_minimum_burn_in=mmse_configs['R_hat_minimum_burn_in'],
            auto_adjust_exploration_var=mmse_configs['auto_adjust_exploration_var'],
            progress_bar=mmse_configs['progress_bar'],
            target_accept_prob=mmse_configs['target_accept_prob'],
        )
        if mmse_configs['run_mcmc']:
            # Execute MCMC estimation
            mmse_estimator.fit(
                x=x,
                initial_condition=initial_conditions['mmse']
            )
        end_mmse = time.time_ns()
        print('Time spent in MCMC: {}'.format((end_mmse-start_mmse)/1E9))        

        # Initialize MAP estimator
        start_map = time.time_ns()
        map_estimator = MAPGradientAscentEstimator(
            learning_rate=map_configs['learning_rate'],
            thresh=map_configs['thresh'],
            max_it=map_configs['max_it'],
            stopping_criterion_persistance_its=map_configs['stopping_criterion_persistance_its'],
            source_pdf_fn=map_configs['source_pdf'],
            source_pdf_derivative_fn=map_configs['source_pdf_derivative'],
            prior_pdf_fn=map_configs['prior_pdf'],
            prior_pdf_derivative_fn=map_configs['prior_pdf_derivative'],
            normalize_posterior=map_configs['normalize_posterior'],
            is_natural_gradient=map_configs['is_natural_gradient']
        )
        
        if map_configs['run_grad_asc']:
            # Run optimization
            map_estimator.fit(
                s=s,
                x=x,
                initial_condition=initial_conditions['map']
            )
        end_map = time.time_ns()

        
        print('Time spent in GA: {}'.format((end_map-start_map)/1E9))

        return mmse_estimator, map_estimator


class MAPGradientAscentEstimator:
    
    def __init__(
        self,
        learning_rate: float,
        thresh: float,
        max_it: int,
        stopping_criterion_persistance_its: int,
        source_pdf_fn,
        source_pdf_derivative_fn,
        prior_pdf_fn,
        prior_pdf_derivative_fn,
        normalize_posterior,
        is_natural_gradient: bool=False   
    ):
        self.log_posterior_fn=PosteriorUtilities.get_log_posterior_fn(
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn,
            use_jax=True
        )
        self.learning_rate=learning_rate
        self.thresh=thresh
        self.max_it=max_it
        self.stopping_criterion_persistance_its=stopping_criterion_persistance_its
        self.is_natural_gradient=is_natural_gradient
        self.source_pdf_fn=source_pdf_fn
        self.source_pdf_derivative_fn=source_pdf_derivative_fn
        self.prior_pdf_fn=prior_pdf_fn
        self.prior_pdf_derivative_fn=prior_pdf_derivative_fn
        self.normalize_posterior=normalize_posterior
    
    
    def fit(
        self,
        s,
        x,
        initial_condition='random'
    ):
        
        def __fit(
            source_pdf,
            source_pdf_derivative,
            prior_pdf,
            prior_pdf_derivative,
            learning_rate,
            thresh,
            max_it,
            stopping_criterion_persistance_its,
            is_natural_gradient,
            s,
            x,
            initial_value
        ):
            
            # Number of observations and sources
            NOBS=s.shape[-1]
            NSOURCES=s.shape[0]

            # Inicialização aleatória de B
            B = initial_value
            logs = pd.DataFrame()

            continue_opt=True
            n=1
            last_posteriori = -np.inf
            non_increasing_iterations=0
            while continue_opt:
                
                # Determinação de gradiente
                deltaB = NOBS*np.transpose(np.linalg.inv(B))
                deltaB = deltaB + (prior_pdf_derivative(B=B)/prior_pdf(B=B))
                y=B@x
                g = source_pdf_derivative(y)/source_pdf(y)
                deltaB = deltaB + g@x.T
                # import pdb; pdb.set_trace()
                # for t in range(NOBS):
                #     x_t = x[:,t]
                #     y_t = B@x_t
                #     g_y = np.array([
                #         source_pdf_derivative(y_t_i)/source_pdf(y_t_i) for y_t_i in y_t
                #     ]).reshape((NSOURCES,1))
                #     deltaB = deltaB + (g_y@x_t.reshape((1,NSOURCES)))
                
                # Normalize posterior update, if so specified
                if self.normalize_posterior:
                    deltaB = deltaB/NOBS

                # Atualização de matriz de separação B
                B = B + learning_rate*deltaB

                # Cálculo de posteriori para registros
                posteriori = self.log_posterior_fn(
                    x=x,
                    B=B
                )

                # Normalize posterior, if so specified
                if self.normalize_posterior:
                    posteriori = posteriori/NOBS
                
                logs = pd.concat(
                    [
                        logs,
                        pd.DataFrame(
                            index=[n],
                            data={
                                'iteration': [n+1],
                                'detB': [np.linalg.det(B)],
                                'log_posterior': [posteriori],
                                'B': [B],
                                'gradient': [deltaB]
                            }
                        )
                    ]
                )


                if n==1:
                    n+=1
                    continue_opt=True
                    last_posteriori=posteriori
                else:
                    n+=1
                    # B_increment = learning_rate * np.sqrt(
                    #     np.sum(
                    #         np.square(deltaB)
                    #     )
                    # )
                    # Verify magnitude of increment to B
                    # if (
                    #     B_increment/np.square(NSOURCES)
                    # ) < thresh:
                        
                    #     non_increasing_iterations += 1
                    # else:
                    #     non_increasing_iterations = 0

                    if (
                        posteriori-last_posteriori < thresh
                    ):
                        non_increasing_iterations += 1
                    else:
                        non_increasing_iterations = 0

                    # Stopping criterion
                    # continue_opt = (n<max_it) and (non_increasing_iterations < stopping_criterion_persistance_its)

                    continue_opt = (n<max_it) and (non_increasing_iterations < stopping_criterion_persistance_its)
                    last_posteriori=posteriori

            return np.array(B), logs
            

        def __parse_GradientAscent_results(
            gradient_ascent_results
        ):

            parsed_results = []
            for i in range(len(gradient_ascent_results)):
                # Get samples and logs
                B_est=gradient_ascent_results[i][0]
                logs=gradient_ascent_results[i][-1]
                
                # Get maximum posterior in realization
                max_posterior = logs.log_posterior.max()

                # Create dict with parsed results
                parsed_results.append(
                    {
                        'result_number': i,
                        'logs': logs,
                        'B_est': B_est,
                        'max_posterior': max_posterior
                    }
                )

            # Get best model index in realizations
            self.B_est_idx = np.argmax([
                r['max_posterior'] for r in parsed_results
            ])

            # Get best model in realizations
            self.B_est = parsed_results[self.B_est_idx]['B_est']

            # Save parsed results
            self.gradient_ascent_results = parsed_results
        
        # Pin static arguments
        exec_fn = functools.partial(
            __fit,
            self.source_pdf_fn,
            self.source_pdf_derivative_fn,
            self.prior_pdf_fn,
            self.prior_pdf_derivative_fn,
            self.learning_rate,
            self.thresh,
            self.max_it,
            self.stopping_criterion_persistance_its,
            self.is_natural_gradient,
            s,
            x
        )
        
        # Initial conditions
        initial_B=initial_condition
        
        # Execute Optimizations
        results = [
            exec_fn(B_0) for B_0 in initial_B
        ]

        __parse_GradientAscent_results(
            gradient_ascent_results=results
        )



class MMSEMetropolisHastingsEstimator:
    
    def __init__(
        self,
        n_samples: int,
        source_pdf_fn,
        prior_pdf_fn,
        exploration_var,
        parallel_chains,
        max_it,
        R_hat_thresh=1.01,
        R_hat_evaluation_step=100,
        R_hat_persistance=5,
        R_hat_minimum_burn_in=1000
    ):
        self.n_samples=n_samples
        self.log_posterior_fn=PosteriorUtilities.get_log_posterior_fn(
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn
        )
        self.Q=self.__get_Q(
            exploration_var=exploration_var
        )
        self.R_hat_thresh=R_hat_thresh
        self.parallel_chains=parallel_chains
        self.max_it=max_it
        self.n_samples_per_chain = n_samples//parallel_chains
        self.R_hat_minimum_evaluation=R_hat_minimum_burn_in + self.n_samples_per_chain
        self.R_hat_persistance=R_hat_persistance
        self.R_hat_evaluation_step=R_hat_evaluation_step
    

    def __get_Q(
        self,
        exploration_var
    ):
        """
            This method obtains the Q function for metropolis-hastings algorithm. The Q function takes in a current value
            of the estimated parameters and proposes a new one.
        """
        def __proposal_fn(
            exploration_var,
            B
        ):
            # Calculate shift in parameter
            shift = np.random.normal(
                loc=0,
                scale=np.sqrt(exploration_var),
                size=B.shape
            )
        
            # Sum shift to obtain new parameter value
            new_B = np.add(
                B, shift
            )
        
            return new_B

        return lambda B: __proposal_fn(
            exploration_var=exploration_var,
            B=B
        )
    
    
    def fit(
        self,
        x,
        initial_condition
    ):
        
        def __get_samples(
            log_posterior_fn,
            x,
            n_samples: int,
            initial_value,
            chain
        ):
            
            # Number of observations and sources
            NOBS=x.shape[-1]
            NSOURCES=x.shape[0]

            # Inicialização aleatória de B
            B = initial_value
            logs = pd.DataFrame()

            max_posterior = -np.inf
            n=1
            MH_samples = []
            last_B = initial_value
            new_B = None
            logs = pd.DataFrame()
            for i in range(n_samples):
                
                # Get new B
                new_B = self.Q(last_B)
                
                # Get proposal alpha
                alpha_proposal = log_posterior_fn(x, new_B) - log_posterior_fn(x, last_B)
                
                # Check whether it is greater than 0 and anti-transform alpha to yield a probability
                alpha = 1 if alpha_proposal > 0 else np.exp(alpha_proposal)
            
                # Get random uniform sample and check whether it is smaller than alpha. This will
                # say if new sample is accepted
                choice_sample = np.random.uniform(0,1)
                save_new_sample = choice_sample < alpha
                
                # Save new sample or current one
                if save_new_sample:
                    MH_samples.append(new_B)
                    last_B=new_B
                else:
                    MH_samples.append(last_B)

                # Save to log
                logs = pd.concat(
                    [
                        logs,
                        pd.DataFrame(
                            index=[i],
                            data={
                                'inside_iteration': [i+1],
                                'B_sample': [last_B],
                                'alpha': [alpha],
                                'save_new_sample': [save_new_sample],
                                'log_posterior': [log_posterior_fn(x, last_B)/NOBS]
                            }
                        )
                    ]
                )
            
            logs['chain'] = chain
            
            return np.array(MH_samples), logs
        
        def __get_diagnostics(
            samples,
            iteration
        ):
            iteration_samples = samples[0].shape[0]
            diagnostics = {'iteration': [iteration]}
            for i,j in np.ndindex((samples[0].shape[-2], samples[0].shape[-1])):
                # Get samples across chains for coefficient ij
                bij_samples = np.array([
                    samples[c][:,i,j] for c in range(self.parallel_chains)
                ])

                # Evaluate r-hat
                diagnostics['r_hat_b_{}{}'.format(i+1, j+1)] = [gelman_rubin(x=bij_samples)]

                # Get ESS
                diagnostics['ess_b_{}{}'.format(i+1, j+1)] = [float(effective_sample_size(bij_samples))]

            return pd.DataFrame(
                index=[0],
                data=diagnostics
            )
        

        def __get_next_iteration(
            diagnostics,
            it
        ):

            # Evaluate persistance
            if diagnostics.shape[0] < self.R_hat_persistance:
                stop=False
            # If at least R_hat_persistance evaluations held:
            else:
                if it < self.max_it:
                    # Evaluate if all R_hats converged for R_hat_persistance evaluations
                    R_hat_cols = [c for c in diagnostics.columns if 'r_hat' in c]
                    if np.all(
                        diagnostics[
                            R_hat_cols
                        ].values[-self.R_hat_persistance:,:] < self.R_hat_thresh
                    ):
                        stop=True
                        self.converged=True
                    else:
                        stop=False

                # Evaluate maximum iterations
                else:
                    stop=True
                    self.converged=False

            # Increment iteration
            if not stop:
                it += self.R_hat_evaluation_step

            return stop, it
            


        def __parse_MCMC_results(
            chains_samples,
            diagnostics,
            logs 
        ):
            
            # Merge samples from different chains into one vector
            merged_samples = np.concatenate(
                [
                    s[-self.n_samples_per_chain:,:,:] for s in chains_samples
                ],
                axis=0
            )

            # Get index for which maximum posterior was found
            max_posterior_idx = logs[
                logs.iteration > logs.iteration.max() - self.n_samples_per_chain
            ].log_posterior.idxmax()

            # Get max posterior sample and index
            max_posterior = logs.loc[max_posterior_idx]['log_posterior']
            max_posterior_B = logs.loc[max_posterior_idx]['B_sample']

            # Get Monte Carlo MMSE estimate
            B_est_mmse = np.mean(
                merged_samples,
                axis=0
            )

            # Average alpha in stationarity
            average_alpha = logs[
                logs.iteration > logs.iteration.max() - self.n_samples_per_chain
            ].alpha.mean()

            # Store parsed results
            parsed_results = {
                'samples': merged_samples,
                'diagnostics': diagnostics,
                'logs': logs,
                'B_est': B_est_mmse,
                'max_posterior': max_posterior,
                'max_posterior_B': max_posterior_B,
                'average_alpha': average_alpha
            }

            # Save parsed results
            self.mcmc_results = parsed_results


        # Run initial period (minimum R-hat evaluation samples)
        results_minimum_eval = [
            __get_samples(
                log_posterior_fn=self.log_posterior_fn,
                x=x,
                n_samples=self.R_hat_minimum_evaluation,
                initial_value=B0,
                chain=c
            ) for c, B0 in zip(
                range(self.parallel_chains), 
                initial_condition
            )
        ]

        # Append samples
        samples = np.array([
            tup[0] for tup in results_minimum_eval
        ])

        # Append logs
        logs = pd.concat(
            [
                tup[-1] for tup in results_minimum_eval
            ],
            axis=0
        )

        # Get R-hats
        diagnostics = __get_diagnostics(
            samples=np.array([
                samples_chain[-self.n_samples_per_chain:,:,:] for samples_chain in samples
            ]),
            iteration=self.R_hat_minimum_evaluation
        )

        # Run loop
        it = self.R_hat_minimum_evaluation + self.R_hat_evaluation_step
        stopping_condition=False
        while not stopping_condition:
            # Run incremental period for evaluating R-hat
            results_incremental_eval = [
                __get_samples(
                    log_posterior_fn=self.log_posterior_fn,
                    x=x,
                    n_samples=self.R_hat_evaluation_step,
                    initial_value=samples[c][-1,:,:],
                    chain=c
                ) for c in range(self.parallel_chains)
            ]

            # Append samples
            samples_incremental = np.array([
                tup[0] for tup in results_incremental_eval
            ])
            samples = np.array([
                np.concatenate(
                    [
                        s,
                        s_inc
                    ],
                    axis=0
                ) for s, s_inc in zip(
                    samples, samples_incremental
                )
            ])

            # Append logs
            logs_incremental = pd.concat(
                [
                    tup[-1] for tup in results_incremental_eval
                ],
                axis=0
            )
            logs = pd.concat(
                [
                    logs,
                    logs_incremental
                ],
                axis=0
            ).reset_index(
                drop=True
            )

            # Get R-hats
            diagnostics = pd.concat(
                [
                    diagnostics,
                    __get_diagnostics(
                        samples=np.array([
                            samples_chain[-self.n_samples_per_chain:,:,:] for samples_chain in samples
                        ]),
                        iteration=it
                    )
                ]               
            ).reset_index(
                drop=True
            )

            # Evaluate next iteration
            stopping_condition, it = __get_next_iteration(
                diagnostics=diagnostics,
                it=it
            )

        # Get iterations per chain
        logs['iteration'] = logs.groupby(
            by='chain'
        ).cumcount() + 1
        

        # Parse results
        __parse_MCMC_results(
            chains_samples=samples,
            diagnostics=diagnostics,
            logs=logs
        )

        self.samples=samples
        self.diagnostics=diagnostics
        self.logs=logs

        return samples, diagnostics, logs
    


class MMSEBarkerMHEstimator:
    """
    MMSE estimator built on top of the Barker proposal (Livingstone & Zanella, 2022),
    using NumPyro's gradient-based `BarkerMH` kernel.
    """

    def __init__(
        self,
        n_samples: int,
        source_pdf_fn,
        prior_pdf_fn,
        exploration_var,
        parallel_chains,
        max_it,
        R_hat_thresh=1.01,
        R_hat_evaluation_step=100,
        R_hat_persistance=5,
        R_hat_minimum_burn_in=1000,
        auto_adjust_exploration_var=False,
        progress_bar=False,
        target_accept_prob=0.5,
    ):
        self.n_samples=n_samples
        self.log_posterior_fn=PosteriorUtilities.get_log_posterior_fn(
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn,
            use_jax=True
        )
        # `exploration_var` is reused as the *initial* Barker step size (it is then
        # adapted by dual-averaging during burn-in). We keep the parameter name so
        # the constructor signature matches the random-walk estimator.
        self.step_size=self.__get_step_size(
            exploration_var=exploration_var
        )
        self.R_hat_thresh=R_hat_thresh
        self.parallel_chains=parallel_chains
        self.max_it=max_it
        self.n_samples_per_chain = n_samples//parallel_chains
        self.R_hat_minimum_evaluation=R_hat_minimum_burn_in + R_hat_evaluation_step
        self.R_hat_minimum_burn_in=R_hat_minimum_burn_in
        self.R_hat_persistance=R_hat_persistance
        self.R_hat_evaluation_step=R_hat_evaluation_step
        self.converged=False
        self.auto_adjust_exploration_var=auto_adjust_exploration_var
        self.progress_bar=progress_bar
        self.target_accept_prob=target_accept_prob

    def __get_step_size(
        self,
        exploration_var
    ):
        """
            Maps the random-walk exploration variance to an equivalent initial Barker
            step size (a scalar scale). Barker adapts this during warm-up, so it only
            sets the starting point of the proposal scale.
        """
        return float(np.sqrt(exploration_var))

    def __get_kernel(
        self,
        x
    ):
        """
            Builds the NumPyro BarkerMH kernel. The kernel is driven by a potential
            function (negative log-posterior) rather than a proposal Q: the Barker
            rule chooses the *sign* of a symmetric perturbation using the gradient of
            the log-posterior, which is what gives it its robustness to tuning.
        """
        # Convert mixture signals to jax numpy array
        x_jax = jnp.asarray(x)

        return BarkerMH(
            potential_fn=lambda B: -self.log_posterior_fn(x_jax, B),
            step_size=self.step_size,
            adapt_step_size=self.auto_adjust_exploration_var,
            adapt_mass_matrix=self.auto_adjust_exploration_var,
            dense_mass=False,
            target_accept_prob=self.target_accept_prob   # asymptotically optimal for the Barker proposal
        )

    def fit(
        self,
        x,
        initial_condition
    ):
        def __get_samples(
            mcmc,
            init_params,
            warmup: bool
        ):
            """
                Runs one block of BarkerMH sampling across all chains at once
                (vectorized over chains) and returns the drawn samples together
                with the per-draw acceptance probabilities.
            """
            # Continue from the last adapted state when this is not the warm-up block
            if warmup:
                mcmc.warmup(
                    jax.random.PRNGKey(0),
                    init_params=init_params,
                    extra_fields=('accept_prob','potential_energy'),
                    collect_warmup=True
                )
                # Retrieve samples and extra fields
                warmup_samples = np.array(
                    mcmc.get_samples(group_by_chain=True)
                )
                warmup_alpha = np.array(
                    mcmc.get_extra_fields(group_by_chain=True)['accept_prob']
                )
                warmup_energy = np.array(
                    mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
                )
                
                mcmc.post_warmup_state=mcmc.last_state
                mcmc.run(
                    mcmc.post_warmup_state.rng_key,
                    extra_fields=('accept_prob','potential_energy'),
                    init_params=init_params
                )
                
                block_samples = np.concatenate(
                    [
                        warmup_samples,
                        np.array(
                            mcmc.get_samples(group_by_chain=True)
                        )
                    ],
                    axis=1
                )
                block_alpha = np.concatenate(
                    [
                        warmup_alpha,
                        np.array(
                            mcmc.get_extra_fields(group_by_chain=True)['accept_prob']
                        )
                    ],
                    axis=1
                )
                block_energy = np.concatenate(
                    [
                        warmup_energy,
                        np.array(
                            mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
                        )
                    ],
                    axis=1
                )
            else:
                mcmc.post_warmup_state=mcmc.last_state
                mcmc.run(
                    mcmc.post_warmup_state.rng_key,
                    extra_fields=('accept_prob','potential_energy'),
                    init_params=init_params
                )
                
                block_samples = np.array(
                    mcmc.get_samples(group_by_chain=True)
                )
                block_alpha = np.array(
                    mcmc.get_extra_fields(group_by_chain=True)['accept_prob']
                )
                
                block_energy = np.array(
                    mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
                )
            
            return mcmc, block_samples, block_alpha, block_energy

        def __get_logs(
            samples,
            alphas,
            energies
        ):
            """
                Rebuilds the per-draw log DataFrame with the same columns as the
                random-walk estimator: inside_iteration, B_sample, alpha,
                save_new_sample and log_posterior (per chain).
            """

            NOBS=x.shape[-1]
            logs = pd.DataFrame()
            
            for c in range(self.parallel_chains):
                chain_samples = samples[c]
                chain_alpha = alphas[c]
                chain_energy = energies[c]

                # save_new_sample: the state moved w.r.t. the previous draw
                moved = np.concatenate([
                    [True],
                    np.any(
                        np.abs(np.diff(chain_samples, axis=0)) > 0,
                        axis=tuple(range(1, chain_samples.ndim))
                    )
                ])
                # Per-draw log-posterior (normalized by NOBS, as in the original).
                # log_posterior_fn is the JAX version; passing NumPy arrays is fine
                # (jnp promotes them) and float(...) pulls the scalar back to Python.
                # log_posterior = np.array([
                #     float(self.log_posterior_fn(x, B))/NOBS for B in chain_samples
                # ])
                
                logs = pd.concat([
                    logs,
                    pd.DataFrame(
                        data={
                            'inside_iteration': np.arange(1, chain_samples.shape[0]+1),
                            'chain': c,
                            # 'B_sample': list(chain_samples),
                            'alpha': chain_alpha,
                            'save_new_sample': moved,
                            'log_posterior': -chain_energy/NOBS,
                            'potential_energy': chain_energy
                        }
                    )
                ], axis=0)

            return logs.reset_index(drop=True)

        def __get_diagnostics(
            samples,
            it
        ):
            """
                Evaluates R-hat coefficient-wise across chains, on the last
                `n_samples_per_chain` draws of each chain.
            """
            window = np.array([
                samples[c][-self.n_samples_per_chain:] for c in range(self.parallel_chains)
            ])
            diagnostics = {'iteration': [it]}

            for i,j in np.ndindex((window.shape[-2], window.shape[-1])):
                # Samples across chains for coefficient ij -> (chains, n_samples)
                bij_samples = window[:, :, i, j]
                # Get R-hat
                diagnostics['r_hat_b_{}{}'.format(i+1, j+1)] = [float(gelman_rubin(bij_samples))]
                # Get ESS
                diagnostics['ess_b_{}{}'.format(i+1, j+1)] = [float(effective_sample_size(bij_samples))]

            return pd.DataFrame(
                index=[0],
                data=diagnostics
            )

        def __get_next_iteration(
            diagnostics_df,
            it
        ):
            # Evaluate persistance
            # if (diagnostics_df.shape[0] < self.R_hat_persistance):
            #     stop=False
            # If at least R_hat_persistance evaluations held:
            # else:
            if it < self.max_it:
                # Evaluate if all R_hats converged for R_hat_persistance evaluations
                R_hat_cols = [c for c in diagnostics_df.columns if 'r_hat' in c]
                if np.all(
                    diagnostics_df[
                        R_hat_cols
                    ].values[-self.R_hat_persistance:,:] < self.R_hat_thresh
                ):
                    stop=True
                    self.converged=True
                else:
                    stop=False
            # Evaluate maximum iterations
            else:
                stop=True
                self.converged=False
            # Increment iteration
            if not stop:
                it += self.R_hat_evaluation_step
            return stop, it

        def __parse_MCMC_results(
            chains_samples,
            diagnostics_df,
            logs
        ):
            # Merge samples from different chains into one vector (stationary window)
            merged_samples = np.concatenate(
                [
                    s[-self.n_samples_per_chain:,:,:] for s in chains_samples
                ],
                axis=0
            )

            # Get index for which maximum posterior was found (stationary window)
            stationary = logs[
                logs.iteration > logs.iteration.max() - self.n_samples_per_chain
            ]
            max_posterior_idx = stationary.log_posterior.idxmax()

            # Get max posterior value
            max_posterior = logs.loc[max_posterior_idx]['log_posterior']
            # max_posterior_B = logs.loc[max_posterior_idx]['B_sample']

            # Get Monte Carlo MMSE estimate
            B_est_mmse = np.mean(
                merged_samples,
                axis=0
            )

            # Average alpha in stationarity
            average_alpha = stationary.alpha.mean()

            # Store parsed results
            parsed_results = {
                'samples': merged_samples,
                'diagnostics': diagnostics_df,
                'logs': logs,
                'B_est': B_est_mmse,
                'max_posterior': max_posterior,
                # 'max_posterior_B': max_posterior_B,
                'average_alpha': average_alpha
            }

            # Save parsed results
            self.mcmc_results = parsed_results

        
        # Build the (single) BarkerMH kernel and MCMC driver. Warm-up performs the
        # step-size / mass-matrix adaptation; each subsequent `mcmc.run` continues
        # the adapted chains for one R-hat evaluation block.
        kernel = self.__get_kernel(x=x)
        mcmc = MCMC(
            kernel,
            num_warmup=self.R_hat_minimum_burn_in,
            num_samples=self.R_hat_evaluation_step,
            num_chains=self.parallel_chains,
            chain_method='sequential',
            progress_bar=self.progress_bar
        )
        # Run initial warm-up period
        mcmc, samples_block, alpha_block, energy_block = __get_samples(
            mcmc=mcmc,
            init_params=initial_condition,
            warmup=True
        )
        samples = samples_block
        alpha = alpha_block
        energy = energy_block

        # Get diagnostics for initial run
        diagnostics_df = __get_diagnostics(
            samples=samples,
            it=self.R_hat_minimum_burn_in + self.R_hat_evaluation_step
        )
        
        # Run loop
        it = self.R_hat_minimum_evaluation + self.R_hat_evaluation_step
        stopping_condition=False if it < self.max_it else True
        while not stopping_condition:
            # Run incremental period for evaluating R-hat
            mcmc, samples_block, alpha_block, energy_block = __get_samples(
                mcmc=mcmc,
                init_params=initial_condition,
                warmup=False
            )
            
            # Append samples / acceptance probabilities along the sample axis
            samples = np.concatenate(
                [
                    samples,
                    samples_block
                ],
                axis=1
            )
            alpha = np.concatenate(
                [
                    alpha,
                    alpha_block
                ],
                axis=1
            )
            energy = np.concatenate(
                [
                    energy,
                    energy_block
                ],
                axis=1
            )
            
            # Get diagnostics
            diagnostics_df = pd.concat(
                [
                    diagnostics_df,
                    __get_diagnostics(
                        samples=samples,
                        it=it
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )
            
            # Evaluate next iteration
            stopping_condition, it = __get_next_iteration(
                diagnostics_df=diagnostics_df,
                it=it
            )

        # Rebuild logs from the accumulated samples / acceptance probabilities
        logs = __get_logs(
            samples=samples,
            alphas=alpha,
            energies=energy
        )

        # Get iterations per chain
        logs['iteration'] = logs.groupby(
            by='chain'
        ).cumcount() + 1

        # Re-shape into the list-of-chains layout expected by the parser / caller
        chains_samples = [samples[c] for c in range(self.parallel_chains)]

        # Parse results
        __parse_MCMC_results(
            chains_samples=samples,
            diagnostics_df=diagnostics_df,
            logs=logs
        )
        # self.chains_samples = chains_samples
        self.samples=samples
        self.diagnostics=diagnostics_df
        self.logs=logs
        
        return samples, diagnostics_df, logs

        