import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathos.pools import ProcessPool
import functools

class LinearInstantaneousPosterior:
    
    @staticmethod
    def get_log_posterior_fn(
        source_pdf_fn,
        prior_pdf_fn
    ):
        """
            This method returns a method for calculating log-posterior.
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
            iterator = np.ndindex(x.shape)
            posteriori += np.sum(np.log(
                np.array([
                    source_pdf_fn(y[i,j]) for i,j in iterator
                ])
            ))
            posteriori += np.log(prior_pdf_fn(B))
        
            return posteriori
        
        return lambda x, B: __log_posterior_fn(
            x=x,
            B=B,
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn
        )