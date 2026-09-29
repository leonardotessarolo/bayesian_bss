import pandas as pd
import numpy as np
import functools
import jax.numpy as jnp
import jax
from numpyro.diagnostics import gelman_rubin, effective_sample_size
from numpyro.infer import MCMC, BarkerMH
import time
import itertools

from .utilities import PosteriorUtilities
import psutil
import os
from collections import Counter

import logging

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
        min_it_grad_asc,
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
            'min_it': min_it_grad_asc,
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
        logger.info('Time spent in MCMC (seconds): {}'.format(round((end_mmse-start_mmse)/1E9, 2)))
        # jax_buffer_report(f"MCMC run")

        # Initialize MAP estimator
        start_map = time.time_ns()
        map_estimator = MAPGradientAscentEstimator(
            learning_rate=map_configs['learning_rate'],
            thresh=map_configs['thresh'],
            max_it=map_configs['max_it'],
            min_it=map_configs['min_it'],
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

        logger.info('Time spent in Gradient Ascent (seconds): {}'.format(round((end_map-start_map)/1E9, 2)))
        # jax_buffer_report(f"GA run")

        return mmse_estimator, map_estimator


class MAPGradientAscentEstimator:
    
    def __init__(
        self,
        learning_rate: float,
        thresh: float,
        max_it: int,
        min_it: int,
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
        self.min_it=min_it
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
            min_it,
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
            n=0
            last_posteriori = -np.inf
            non_increasing_iterations=0
            logs_list = []
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
                # if self.normalize_posterior:
                #     deltaB = deltaB/NOBS

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
                
                logs_list.append({
                    'iteration': n,
                    'detB': np.linalg.det(B),
                    'log_posterior': float(posteriori),
                    'B': np.asarray(B, copy=True),
                    'gradient': np.asarray(deltaB, copy=True)
                })


                if n==0:
                    n+=1
                    continue_opt=True
                    last_posteriori=posteriori
                else:
                    n+=1
                    if (
                        posteriori-last_posteriori < thresh
                    ):
                        non_increasing_iterations += 1
                    else:
                        non_increasing_iterations = 0

                    continue_opt = (
                        (n<max_it) and (non_increasing_iterations < stopping_criterion_persistance_its)
                    ) or (n<min_it)
                    last_posteriori=posteriori
            
            logs = pd.DataFrame(logs_list)
            
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
            self.min_it,
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
                            'chain': float(c),
                            # 'B_sample': list(chain_samples),
                            'alpha': np.asarray(chain_alpha),
                            'save_new_sample': np.asarray(moved),
                            'log_posterior': np.asarray(-chain_energy/NOBS),
                            'potential_energy': np.asarray(chain_energy)
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
        # chains_samples = [samples[c] for c in range(self.parallel_chains)]

        # Parse results
        __parse_MCMC_results(
            chains_samples=samples,
            diagnostics_df=diagnostics_df,
            logs=logs
        )
        # self.chains_samples = chains_samples
        self.samples=np.asarray(samples)
        self.diagnostics=diagnostics_df
        self.logs=logs

        # Clear MCMC object
        mcmc._last_state = None
        mcmc.post_warmup_state = None
        mcmc._states = None
        mcmc._states_flat = None
        if hasattr(mcmc, "_cache"):
            mcmc._cache.clear()
        del mcmc, kernel
        
        return samples, diagnostics_df, logs


class ImportanceSamplingEstimator:
    
    """
    Self-normalized importance sampling estimator of posterior means under
    a list of alternative priors, reusing one set of previously-drawn MCMC
    samples.
    """

    @classmethod
    def __psislw(
        cls,
        lw,
        Reff=1.0,
        overwrite_lw=False
    ):
        """
        From Aki Vehtari's PSIS GitHub repo: https://github.com/avehtari/PSIS. 

        Pareto smoothed importance sampling (PSIS).

        Parameters
        ----------
        lw : ndarray
            Array of size n x m containing m sets of n log weights. It is also
            possible to provide one dimensional array of length n.

        Reff : scalar, optional
            relative MCMC efficiency ``N_eff / N``

        overwrite_lw : bool, optional
            If True, the input array `lw` is smoothed in-place, assuming the array
            is F-contiguous. By default, a new array is allocated.

        Returns
        -------
        lw_out : ndarray
            smoothed log weights
        kss : ndarray
            Pareto tail indices

        """
        if lw.ndim == 2:
            n, m = lw.shape
        elif lw.ndim == 1:
            n = len(lw)
            m = 1
        else:
            raise ValueError("Argument `lw` must be 1 or 2 dimensional.")
        if n <= 1:
            raise ValueError("More than one log-weight needed.")

        if overwrite_lw and lw.flags.f_contiguous:
            # in-place operation
            lw_out = lw
        else:
            # allocate new array for output
            lw_out = np.copy(lw, order='F')

        # allocate output array for kss
        kss = np.empty(m)

        # precalculate constants
        cutoff_ind = - int(np.ceil(min(0.2 * n, 3 * np.sqrt(n / Reff)))) - 1
        cutoffmin = np.log(np.finfo(float).tiny)
        logn = np.log(n)
        k_min = 1/3

        # loop over sets of log weights
        for i, x in enumerate(lw_out.T if lw_out.ndim == 2 else lw_out[None, :]):
            # improve numerical accuracy
            x -= np.max(x)
            # sort the array
            x_sort_ind = np.argsort(x)
            # divide log weights into body and right tail
            xcutoff = max(
                x[x_sort_ind[cutoff_ind]],
                cutoffmin
            )
            expxcutoff = np.exp(xcutoff)
            tailinds, = np.where(x > xcutoff)
            x2 = x[tailinds]
            n2 = len(x2)
            if n2 <= 4:
                # not enough tail samples for gpdfitnew
                k = np.inf
            else:
                # order of tail samples
                x2si = np.argsort(x2)
                # fit generalized Pareto distribution to the right tail samples
                np.exp(x2, out=x2)
                x2 -= expxcutoff
                k, sigma = cls.__gpdfitnew(x2, sort=x2si)
            if k >= k_min and not np.isinf(k):
                # no smoothing if short tail or GPD fit failed
                # compute ordered statistic for the fit
                sti = np.arange(0.5, n2)
                sti /= n2
                qq = cls.__gpinv(sti, k, sigma)
                qq += expxcutoff
                np.log(qq, out=qq)
                # place the smoothed tail into the output array
                x[tailinds[x2si]] = qq
                # truncate smoothed values to the largest raw weight 0
                x[x > 0] = 0
            # renormalize weights
            x -= cls.__sumlogs(x)
            # store tail index k
            kss[i] = k

        # If the provided input array is one dimensional, return kss as scalar.
        if lw_out.ndim == 1:
            kss = kss[0]

        return lw_out, kss

    @classmethod
    def __gpdfitnew(
        cls,
        x,
        sort=True,
        sort_in_place=False,
        return_quadrature=False
    ):
        """
        From Aki Vehtari's PSIS GitHub repo: https://github.com/avehtari/PSIS.

        Estimate the paramaters for the Generalized Pareto Distribution (GPD)

        Returns empirical Bayes estimate for the parameters of the two-parameter
        generalized Parato distribution given the data.

        Parameters
        ----------
        x : ndarray
            One dimensional data array

        sort : bool or ndarray, optional
            If known in advance, one can provide an array of indices that would
            sort the input array `x`. If the input array is already sorted, provide
            False. If True (default behaviour), the array is sorted internally.

        sort_in_place : bool, optional
            If `sort` is True and `sort_in_place` is True, the array is sorted
            in-place (False by default).

        return_quadrature : bool, optional
            If True, quadrature points and weight `ks` and `w` of the marginal posterior distribution of k are also calculated and returned. False by
            default.

        Returns
        -------
        k, sigma : float
            estimated parameter values

        ks, w : ndarray
            Quadrature points and weights of the marginal posterior distribution
            of `k`. Returned only if `return_quadrature` is True.

        Notes
        -----
        This function returns a negative of Zhang and Stephens's k, because it is
        more common parameterisation.

        """
        if x.ndim != 1 or len(x) <= 1:
            raise ValueError("Invalid input array.")

        # check if x should be sorted
        if sort is True:
            if sort_in_place:
                x.sort()
                xsorted = True
            else:
                sort = np.argsort(x)
                xsorted = False
        elif sort is False:
            xsorted = True
        else:
            xsorted = False

        n = len(x)
        PRIOR = 3
        m = 30 + int(np.sqrt(n))

        bs = np.arange(1, m + 1, dtype=float)
        bs -= 0.5
        np.divide(m, bs, out=bs)
        np.sqrt(bs, out=bs)
        np.subtract(1, bs, out=bs)
        if xsorted:
            bs /= PRIOR * x[int(n/4 + 0.5) - 1]
            bs += 1 / x[-1]
        else:
            bs /= PRIOR * x[sort[int(n/4 + 0.5) - 1]]
            bs += 1 / x[sort[-1]]

        ks = np.negative(bs)
        temp = ks[:,None] * x
        np.log1p(temp, out=temp)
        np.mean(temp, axis=1, out=ks)

        L = bs / ks
        np.negative(L, out=L)
        np.log(L, out=L)
        L -= ks
        L -= 1
        L *= n

        temp = L - L[:,None]
        np.exp(temp, out=temp)
        w = np.sum(temp, axis=1)
        np.divide(1, w, out=w)

        # remove negligible weights
        dii = w >= 10 * np.finfo(float).eps
        if not np.all(dii):
            w = w[dii]
            bs = bs[dii]
        # normalise w
        w /= w.sum()

        # posterior mean for b
        b = np.sum(bs * w)
        # Estimate for k, note that we return a negative of Zhang and
        # Stephens's k, because it is more common parameterisation.
        temp = (-b) * x
        np.log1p(temp, out=temp)
        k = np.mean(temp)
        if return_quadrature:
            np.negative(x, out=temp)
            temp = bs[:, None] * temp
            np.log1p(temp, out=temp)
            ks = np.mean(temp, axis=1)
        # estimate for sigma
        sigma = -k / b * n / (n - 0)
        # weakly informative prior for k
        a = 10
        k = k * n / (n+a) + a * 0.5 / (n+a)
        if return_quadrature:
            ks *= n / (n+a)
            ks += a * 0.5 / (n+a)

        if return_quadrature:
            return k, sigma, ks, w
        else:
            return k, sigma

    @classmethod
    def __gpinv(
        cls,
        p,
        k,
        sigma
    ):
        """
        
        From Aki Vehtari's PSIS GitHub repo: https://github.com/avehtari/PSIS.
        
        Inverse Generalised Pareto distribution function.
        
        """
        x = np.empty(p.shape)
        x.fill(np.nan)
        if sigma <= 0:
            return x
        ok = (p > 0) & (p < 1)
        if np.all(ok):
            if np.abs(k) < np.finfo(float).eps:
                np.negative(p, out=x)
                np.log1p(x, out=x)
                np.negative(x, out=x)
            else:
                np.negative(p, out=x)
                np.log1p(x, out=x)
                x *= -k
                np.expm1(x, out=x)
                x /= k
            x *= sigma
        else:
            if np.abs(k) < np.finfo(float).eps:
                # x[ok] = - np.log1p(-p[ok])
                temp = p[ok]
                np.negative(temp, out=temp)
                np.log1p(temp, out=temp)
                np.negative(temp, out=temp)
                x[ok] = temp
            else:
                # x[ok] = np.expm1(-k * np.log1p(-p[ok])) / k
                temp = p[ok]
                np.negative(temp, out=temp)
                np.log1p(temp, out=temp)
                temp *= -k
                np.expm1(temp, out=temp)
                temp /= k
                x[ok] = temp
            x *= sigma
            x[p == 0] = 0
            if k >= 0:
                x[p == 1] = np.inf
            else:
                x[p == 1] = -sigma / k
        return x

    @classmethod
    def __sumlogs(
        cls,
        x,
        axis=None,
        out=None
    ):
        """
        
        From Aki Vehtari's PSIS GitHub repo: https://github.com/avehtari/PSIS.

        Sum of vector where numbers are represented by their logarithms.

        Calculates ``np.log(np.sum(np.exp(x), axis=axis))`` in such a fashion that
        it works even when elements have large magnitude.

        """
        maxx = x.max(axis=axis, keepdims=True)
        xnorm = x - maxx
        np.exp(xnorm, out=xnorm)
        out = np.sum(xnorm, axis=axis, out=out)
        if isinstance(out, np.ndarray):
            np.log(out, out=out)
        else:
            out = np.log(out)
        out += np.squeeze(maxx)
        return out

    @classmethod
    def __resolve_experiment(
        cls,
        baseline_prior,
        target_priors,
        prior_control_params,
        baseline_source_model,
        target_source_models,
        source_model_control_params
    ):
        impsamp_configs = {}

        # Error: nothing specified
        if (
            (baseline_prior is None) and 
            (target_priors is None) and 
            (baseline_source_model is None) and
            (target_source_models is None)
        ):
            raise ValueError('Neither priors or source models are specified.')
        
        # Only prior variations specified
        if (
            (baseline_prior is not None) and 
            (target_priors is not None) and 
            (baseline_source_model is not None) and
            (target_source_models is None)
        ):
            impsamp_configs['prior_variation'] = {
                'targets': [p.get() for p in target_priors], 
                'baseline': baseline_prior.get(),
                'params': prior_control_params
            }

        # Only source model variations specified
        if (
            (baseline_prior is not None) and 
            (target_priors is None) and 
            (baseline_source_model is not None) and
            (target_source_models is not None)
        ):
            impsamp_configs['source_model_variation'] = {
                'targets': [
                    PosteriorUtilities.get_log_posterior_fn(
                        source_pdf_fn=sm,
                        prior_pdf_fn=lambda B: 1
                    ) for sm in target_source_models
                ],
                'baseline': PosteriorUtilities.get_log_posterior_fn(
                    source_pdf_fn=baseline_source_model.get(),
                    prior_pdf_fn=lambda B: 1
                ),
                'params': source_model_control_params
            }

        # Both prior and source model variations specified
        if (
            (baseline_prior is not None) and 
            (target_priors is not None) and 
            (baseline_source_model is not None) and
            (target_source_models is not None)
        ):
            impsamp_configs['prior_source_model_variation'] = {
                'targets': [
                    PosteriorUtilities.get_log_posterior_fn(
                        source_pdf_fn=sm.get(),
                        prior_pdf_fn=p.get()
                    ) for p, sm in itertools.product(
                        target_priors,
                        target_source_models
                    )
                ],
                'baseline': PosteriorUtilities.get_log_posterior_fn(
                    source_pdf_fn=baseline_source_model.get(),
                    prior_pdf_fn=baseline_prior.get()
                ),
                'params': itertools.product(
                    prior_control_params,
                    source_model_control_params
                )
            }

        cls.impsamp_configs = impsamp_configs
        

    @classmethod
    def __get_log_weights(
        cls,
        samples: np.ndarray,
        diagnostics,
        x,
        run_psis
    ) -> np.ndarray:
        
        def __get_one_prior(
            samples,
            baseline_prior,
            target_prior
        ):
            return np.asarray(
                target_prior(B=samples)/baseline_prior(B=samples)
            )
            
        def __get_one_posterior(
            samples,
            baseline_post,
            x,
            post
        ):
            w_p=[]
            for B_k in samples:
                weight = post(
                    x=x,
                    B=B_k
                ) - baseline_post(
                    x=x,
                    B=B_k
                )
                w_p.append(weight)

            return np.exp(w_p)
            
        
        impsamp_results = cls.impsamp_configs
        # Ge weights for prior variation analysis
        if 'prior_variation' in impsamp_results.keys():
            impsamp_results['prior_variation']['raw_weights'] = []
            impsamp_results['prior_variation']['weights'] = []
            if run_psis:
                impsamp_results['prior_variation']['k_hat'] = []
            for target_prior in impsamp_results['prior_variation']['targets']:
                # Get raw weights
                raw_weights = __get_one_prior(
                    samples=samples,
                    baseline_prior=impsamp_results['prior_variation']['baseline'],
                    target_prior=target_prior
                )

                # Get final weights
                if not run_psis:
                    weights = raw_weights
                else:
                    # Retrieve minimum ESS among scalar parameters (conservative summary)
                    ess_cols = [c for c in diagnostics.columns if 'ess' in c]
                    ess = np.min(
                        diagnostics.iloc[-1,:][ess_cols]
                    )
                    
                    # Perform PSIS
                    weights, k_hat = cls.__psislw(
                        lw=np.log(raw_weights),
                        overwrite_lw=False,
                        Reff=ess/len(samples)
                    )

                # Log
                impsamp_results['prior_variation']['raw_weights'].append(raw_weights)
                impsamp_results['prior_variation']['weights'].append(np.exp(weights))
                if run_psis:
                    impsamp_results['prior_variation']['k_hat'].append(k_hat)

        cls.impsamp_results = impsamp_results

    @classmethod
    def __get_estimates(
        cls,
        samples
    ):
        def __get_moments(
            samples,
            weights
        ):
            # Get sum total of weights
            sum_weights = np.sum(weights)
            
            # Get mean
            B_mean = np.sum([
                s*w for s, w in zip(samples, weights)
            ], axis=0)/sum_weights

            # Differences wrt mean
            mean_diffs = np.array([
                np.subtract(
                    s,
                    B_mean
                ) for s in samples
            ])

            # Get variance 
            B_var = np.sum([
                w*np.power(d, 2) for d, w in zip(mean_diffs, weights)
            ], axis=0)/sum_weights

            # Get skew 
            B_skew = np.sum([
                w*np.power(d, 3) for d, w in zip(mean_diffs, weights)
            ], axis=0)/sum_weights

            # Get kurtosis 
            B_kurt = np.sum([
                w*np.power(d, 4) for d, w in zip(mean_diffs, weights)
            ], axis=0)/sum_weights

            return {
                'mean': B_mean,
                'var': B_var,
                'skew': B_skew,
                'kurt': B_kurt
            }
        
        for analysis_type, analysis_params in cls.impsamp_results.items():
            analysis_params['means'] = []
            analysis_params['vars'] = []
            analysis_params['skews'] = []
            analysis_params['kurts'] = []
            for weights in analysis_params['weights']:
                r = __get_moments(
                    samples,
                    weights
                )
                analysis_params['means'].append(r['mean'])
                analysis_params['vars'].append(r['var'])
                analysis_params['skews'].append(r['skew'])
                analysis_params['kurts'].append(r['kurt'])

            cls.impsamp_results[analysis_type].update(analysis_params)

    @classmethod
    def run(
        cls,
        samples,
        diagnostics,
        baseline_prior,
        target_priors,
        prior_control_params,
        baseline_source_model=None,
        target_source_models=None,
        source_model_control_params=None,
        x=None,
        run_psis=True,
        ess_ratio_thresh: float = 0.1,
        eps: float = 1e-300,
        show_progress: bool = False,
    ):
        
        # Resolve importance sampling experiment configurations
        cls.__resolve_experiment(
            baseline_prior=baseline_prior,
            target_priors=target_priors,
            prior_control_params=prior_control_params,
            baseline_source_model=baseline_source_model,
            target_source_models=target_source_models,
            source_model_control_params=source_model_control_params
        )
        
        # Get importance sampling weights
        cls.__get_log_weights(
            samples=samples,
            diagnostics=diagnostics,
            x=x,
            run_psis=run_psis
        )

        # Get estimates of first 4 moments
        cls.__get_estimates(
            samples=samples
        )

        return cls.impsamp_results

    
    
        