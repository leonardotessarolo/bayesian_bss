import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathos.pools import ProcessPool
import functools

class MAPGradientEstimator:
    
    def __init__(
        self,
        learning_rate: float,
        thresh: float,
        max_it: int,
        source_pdf,
        source_pdf_derivative,
        prior_pdf,
        prior_pdf_derivative,
        is_natural_gradient: bool=False   
    ):
        self.learning_rate=learning_rate
        self.thresh=thresh
        self.max_it=max_it
        self.is_natural_gradient=is_natural_gradient
        self.source_pdf=source_pdf
        self.source_pdf_derivative=source_pdf_derivative
        self.prior_pdf=prior_pdf
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
            # print('-'*100)
            while continue_opt:
                # Determinação de gradiente
                deltaB = np.transpose(np.linalg.inv(B))
                deltaB = deltaB + (1/NOBS)*(prior_pdf_derivative(X=B)/prior_pdf(X=B))
                for t in range(NOBS):
                    x_t = x[:,t]
                    y_t = B@x_t
                    g_y = np.array([
                        source_pdf_derivative(y_t_i)/source_pdf(y_t_i) for y_t_i in y_t
                    ]).reshape((NSOURCES,1))
                    deltaB = deltaB + (1/NOBS)*(g_y@x_t.reshape((1,NSOURCES)))

                # Natural gradient
                if is_natural_gradient:
                    deltaB = deltaB@B.T@B

                # Atualização de matriz de separação B
                B = B + learning_rate*deltaB

                # Cálculo de posteriori para registros
                posteriori = NOBS*np.log(np.abs(np.linalg.det(B)))
                y=B@x
                for i, j in np.ndindex(x.shape):
                    posteriori += np.log(source_pdf(y[i,j]))
                posteriori += np.log(prior_pdf(B))

                logs = pd.concat(
                    [
                        logs,
                        pd.DataFrame(
                            index=[n],
                            data={
                                'iteration': [n+1],
                                'detB': [np.linalg.det(B)],
                                'log_posteriori': [posteriori],
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
                    continue_opt = (n<max_it) and (np.abs(posteriori-last_posteriori) > thresh)
                    last_posteriori=posteriori

                # if n%100==0:
                #     print('Iteração: {}'.format(n))
                #     print('Log-Posteriori: {}'.format(posteriori))
                #     print('-'*100)

            return {
                'unmixing_matrix': B,
                'logs': logs
            }
        
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
            self.is_natural_gradient,
            s,
            x
        )
        
        # Initial conditions
        if initial_condition=='random':
            initial_B = np.random.normal(
                0,1,
                (
                    s.shape[0],
                    s.shape[0],
                    n_initializations
                )
            )
        else:
            initial_B=initial_condition
        # initial_B=initial_condition
        
        # with ProcessPool(n_jobs) as p:
        #     results=p.map(exec_fn, [initial_B[:,:,i] for i in range(initial_B.shape[-1])])
        results = [
            exec_fn(initial_B[:,:,i]) for i in range(initial_B.shape[-1])
        ]
        
        
        self.fit_results=results



class MMSEMetropolisHastingsEstimator:
    
    def __init__(
        self,
        n_samples: int,
        log_posterior_fn,
        Q,
        burn_in: float=0.5   
    ):
        self.n_samples=n_samples
        self.log_posterior_fn=log_posterior_fn
        self.Q=Q
        self.burn_in=burn_in
        self.burn_in_start=int(burn_in*n_samples)
    
    
    def fit(
        self,
        s,
        x,
        initial_condition='random',
        print_every=None,
        n_jobs:int=1
    ):
        
        def __get_samples(
            n_samples: int,
            log_posterior_fn,
            log_proposal_distribution,
            s,
            x,
            print_every,
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
                alpha_proposal = self.log_posterior_fn(new_B) - self.log_posterior_fn(last_B)
                
                # Check whether it is greater than 0 and anti-transform alpha to yield a probability
                alpha = 1 if alpha_proposal > 0 else np.exp(alpha_proposal)
            
                # Get random uniform sample and check whether it is smaller than alpha. This will
                # say if new sample is accepted
                choice_sample = np.random.uniform(0,1)
                save_new_sample = choice_sample < alpha

                if print_every is not None:
                    if i%print_every==0:
                        print('-'*100)
                        print('iteration {}'.format(i))
                        print('last_B:\n{}'.format(last_B))
                        print('new_B:\n{}'.format(new_B))
                        print('alpha_proposal: {}'.format(alpha_proposal))
                        print('alpha: {}'.format(alpha))
                        print('save_new_sample: {}'.format(save_new_sample))
                        print('-'*100)
                
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
                                'log_posterior': [self.log_posterior_fn(last_B)]
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
            x,
            print_every
        )
        
        # Initial conditions
        initial_B=initial_condition

        # Execute MCMC realizations with different starting points
        with ProcessPool(n_jobs) as p:
            results=p.map(
                exec_fn, 
                [
                    initial_B[:,:,i] for i in range(initial_B.shape[-1])
                ]
            )
        # For debugging
        # results = [
        #     exec_fn(initial_B[:,:,i]) for i in range(initial_B.shape[-1])
        # ]

        # Parse results
        __parse_MCMC_results(
            mcmc_results=results
        )
        