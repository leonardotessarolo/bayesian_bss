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
        
        # with ProcessPool(n_jobs) as p:
        #     results=p.map(exec_fn, [initial_B[:,:,i] for i in range(initial_B.shape[-1])])
            results = [
                exec_fn(initial_B[:,:,i]) for i in range(initial_B.shape[-1])
            ]
        
        
        self.fit_results=results
        
        
        