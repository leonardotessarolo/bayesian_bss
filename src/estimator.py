import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathos.pools import ProcessPool
import functools

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
        burn_in_mcmc,
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
            'burn_in': burn_in_mcmc
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
        initial_B,
        n_jobs=1
    ):
        
        # Initialize MH estimator
        mmse_estimator = MMSEMetropolisHastingsEstimator(
            n_samples=mmse_configs['n_samples'],
            source_pdf_fn=mmse_configs['source_pdf_fn'],
            prior_pdf_fn=mmse_configs['prior_pdf_fn'],
            exploration_var=mmse_configs['exploration_var'],
            burn_in=mmse_configs['burn_in']
        )
        
        # Execute MCMC estimation
        mmse_estimator.fit(
            s,
            x,
            initial_condition=initial_B,
            n_jobs=n_jobs
        )
    
        # Initialize MAP estimator
        map_estimator = MAPGradientAscentEstimator(
            learning_rate=map_configs['learning_rate'],
            thresh=map_configs['thresh'],
            max_it=map_configs['max_it'],
            stopping_criterion_persistance_its=map_configs['stopping_criterion_persistance_its'],
            source_pdf=map_configs['source_pdf'],
            source_pdf_derivative=map_configs['source_pdf_derivative'],
            prior_pdf=map_configs['prior_pdf'],
            prior_pdf_derivative=map_configs['prior_pdf_derivative'],
            normalize_posterior=map_configs['normalize_posterior'],
            is_natural_gradient=map_configs['is_natural_gradient']
        )
    
        # Run optimization
        map_estimator.fit(
            s=s,
            x=x,
            initial_condition=initial_B,
            n_jobs=n_jobs
        )

        return mmse_estimator, map_estimator


class MAPGradientAscentEstimator:
    
    def __init__(
        self,
        learning_rate: float,
        thresh: float,
        max_it: int,
        stopping_criterion_persistance_its: int,
        source_pdf,
        source_pdf_derivative,
        prior_pdf,
        prior_pdf_derivative,
        normalize_posterior,
        is_natural_gradient: bool=False   
    ):
        self.learning_rate=learning_rate
        self.thresh=thresh
        self.max_it=max_it
        self.stopping_criterion_persistance_its=stopping_criterion_persistance_its
        self.is_natural_gradient=is_natural_gradient
        self.source_pdf=source_pdf
        self.source_pdf_derivative=source_pdf_derivative
        self.prior_pdf=prior_pdf
        self.normalize_posterior=normalize_posterior
        self.prior_pdf_derivative=prior_pdf_derivative
    
    
    
    def fit(
        self,
        s,
        x,
        initial_condition='random',
        n_initializations:int=1,
        n_jobs:int=1
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
            # print('-'*100)
            while continue_opt:
                
                # Determinação de gradiente
                deltaB = NOBS*np.transpose(np.linalg.inv(B))
                deltaB = deltaB + (prior_pdf_derivative(B=B)/prior_pdf(B=B))
                for t in range(NOBS):
                    x_t = x[:,t]
                    y_t = B@x_t
                    g_y = np.array([
                        source_pdf_derivative(y_t_i)/source_pdf(y_t_i) for y_t_i in y_t
                    ]).reshape((NSOURCES,1))
                    deltaB = deltaB + (g_y@x_t.reshape((1,NSOURCES)))

                # # Natural gradient
                # if is_natural_gradient:
                #     deltaB = deltaB@B.T@B
                
                # Normalize posterior update, if so specified
                if self.normalize_posterior:
                    deltaB = deltaB/NOBS

                # Atualização de matriz de separação B
                B = B + learning_rate*deltaB

                # Cálculo de posteriori para registros
                posteriori = NOBS*np.log(np.abs(np.linalg.det(B)))
                y=B@x
                for i, j in np.ndindex(x.shape):
                    posteriori += np.log(source_pdf(y[i,j]))
                posteriori += np.log(prior_pdf(B=B))

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
            self.source_pdf,
            self.source_pdf_derivative,
            self.prior_pdf,
            self.prior_pdf_derivative,
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
        # with ProcessPool(n_jobs) as p:
        #     results=p.map(
        #         exec_fn, 
        #         [
        #             initial_B[:,:,i] for i in range(initial_B.shape[-1])
        #         ]
        #     )
        # For debugging
        results = [
            exec_fn(initial_B[:,:,i]) for i in range(initial_B.shape[-1])
        ]

        __parse_GradientAscent_results(
            gradient_ascent_results=results
        )



class MMSEMetropolisHastingsEstimator:
    
    def __init__(
        self,
        n_samples: int,
        # log_posterior_fn,
        # Q,
        source_pdf_fn,
        prior_pdf_fn,
        exploration_var,
        burn_in: float=0.5   
    ):
        self.n_samples=n_samples
        self.log_posterior_fn=self.__get_log_posterior_fn(
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn
        )
        self.Q=self.__get_Q(
            exploration_var=exploration_var
        )
        self.burn_in=burn_in
        self.burn_in_start=int(burn_in*n_samples)


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

    def __get_log_posterior_fn(
        self,
        source_pdf_fn,
        prior_pdf_fn
    ):
        """
            This method returns a method for calculating log-posterior inside MCMC.
        """
        def __log_posterior_fn(
            x,
            B,
            source_pdf_fn,
            prior_pdf_fn
        ):
            NOBS=x.shape[-1]
            
            # Cálculo de posteriori para registros
            posteriori = NOBS*np.log(np.abs(np.linalg.det(B)))
            y=B@x
            for i, j in np.ndindex(x.shape):
                posteriori += np.log(source_pdf_fn(y[i,j]))
            posteriori += np.log(prior_pdf_fn(B))
        
            return posteriori

        return lambda x, B: __log_posterior_fn(
            x=x,
            B=B,
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn
        )
        
    
    def fit(
        self,
        s,
        x,
        initial_condition='random',
        n_jobs:int=1
    ):
        
        def __get_samples(
            n_samples: int,
            log_posterior_fn,
            log_proposal_distribution,
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

            max_posterior = -np.inf
            n=1
            MH_samples = []
            last_B = initial_value
            new_B = None
            logs = pd.DataFrame()
            for i in range(n_samples):
                # import pdb; pdb.set_trace()
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
                                'iteration': [i+1],
                                'alpha_proposal': [alpha_proposal],
                                'alpha': [alpha],
                                'save_new_sample': [save_new_sample],
                                'log_posterior': [log_posterior_fn(x, last_B)]
                            }
                        )
                    ]
                )
            
            return np.array(MH_samples), logs


        def __parse_MCMC_results(
            mcmc_results
        ):

            parsed_results = []
            for i in range(len(mcmc_results)):
                # Get samples and logs
                samples=mcmc_results[i][0]
                logs=mcmc_results[i][-1]
                
                # Get maximum posterior in realization
                max_posterior = logs.log_posterior.max()

                # Get MMSE estimate
                valid_samples=samples[self.burn_in_start:, :, :]
                B_est = np.sum(
                    valid_samples,
                    axis=0
                )/len(valid_samples)

                parsed_results.append(
                    {
                        'result_number': i,
                        'samples': samples,
                        'logs': logs,
                        'valid_samples': valid_samples,
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
            self.mcmc_results = parsed_results

        
        # Pin static arguments
        exec_fn = functools.partial(
            __get_samples,
            self.n_samples,
            self.log_posterior_fn,
            self.Q,
            s,
            x
        )
        
        # Initial conditions
        initial_B=initial_condition

        # Execute MCMC realizations with different starting points
        # with ProcessPool(n_jobs) as p:
        #     results=p.map(
        #         exec_fn, 
        #         [
        #             initial_B[:,:,i] for i in range(initial_B.shape[-1])
        #         ]
        #     )
        # For debugging
        results = [
            exec_fn(initial_B[:,:,i]) for i in range(initial_B.shape[-1])
        ]

        # Parse results
        __parse_MCMC_results(
            mcmc_results=results
        )
        