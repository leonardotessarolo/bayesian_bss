import numpy as np
import itertools
from functools import partial
import pathos


class PosteriorContourLines:
    
    def __init__(
        self,
        u_lims,
        v_lims,
        n_points,
        central_point,
        source_pdf_fn,
        prior_pdf_fn,
        normalize_posterior=False
    ):

        # Get grid points at which posterior will be evaluated
        self.__get_evaluation_grid(
            u_lims=u_lims,
            v_lims=v_lims,
            n_points=n_points,
            central_point=central_point
        )

        self.normalize_posterior=normalize_posterior
        self.log_posterior_fn = self.__get_log_posterior_fn(
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn,
            normalize_posterior=normalize_posterior
        )

        
        

    def __get_log_posterior_fn(
        self,
        source_pdf_fn,
        prior_pdf_fn,
        normalize_posterior
    ):
        """
            This method returns a method for calculating log-posterior inside MCMC.
        """
        def __log_posterior_fn(
            x,
            B,
            source_pdf_fn,
            prior_pdf_fn,
            normalize_posterior
        ):
            NOBS=x.shape[-1]
            
            # Cálculo de posteriori para registros
            posteriori = NOBS*np.log(np.abs(np.linalg.det(B)))
            y=B@x
            for i, j in np.ndindex(x.shape):
                posteriori += np.log(source_pdf_fn(y[i,j]))
            posteriori += np.log(prior_pdf_fn(B))

            if normalize_posterior:
                posteriori = posteriori/NOBS
        
            return posteriori

        return lambda x, B: __log_posterior_fn(
            x=x,
            B=B,
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn,
            normalize_posterior=normalize_posterior
        )
        
    def __get_evaluation_grid(
        self,
        u_lims,
        v_lims,
        n_points,
        central_point
    ):
        # Get step
        u_step = (u_lims[-1] - u_lims[0])/(n_points-1)
        v_step = (v_lims[-1] - v_lims[0])/(n_points-1)
        
        # Step multipliers
        step_multipliers = range(-(n_points-1)//2, (n_points-1)//2)
        
        # Get u_vec and v_vec
        self.u_vec = np.array([central_point[0] + s*u_step for s in step_multipliers])
        self.v_vec = np.array([central_point[-1] + s*v_step for s in step_multipliers])


    def get_posteriori_neighborhood(
        self,
        A,
        x,
        N,
        njobs=1
    ):
        
        def __get_posteriori(
            A,
            x,
            N,
            idx_info
        ):
            # Parse index info
            i, u = idx_info[0]
            j, v = idx_info[-1]

            # Get shifted value of A which will be evaluated and corresponding value of B
            # A_shifted = A + u*symmetric_basis + v*skew_symmetric_basis
            u_matrix = np.array([
                [np.cosh(u), np.sinh(u)],
                [np.sinh(u), np.cosh(u)]
            ])

            v_matrix = np.array([
                [np.cos(v), -np.sin(v)],
                [np.sin(v), np.cos(v)]
            ])
            shift_matrix = u_matrix@v_matrix
            # import pdb;pdb.set_trace()
            # A_shifted = A*u_matrix*v_matrix
            
            # B = np.linalg.inv(A)@(np.eye(2) + u*symmetric_basis + v*skew_symmetric_basis)
            B = np.linalg.inv(A) + u*symmetric_basis + v*skew_symmetric_basis

            # # Initialize likelihood with part that does not depend on observation
            # likelihood = np.log(np.abs(np.linalg.det(B)))

            # # Iterator over sample and observation position
            # likelihood_iterator = [
            #     (t,i) for t,i in itertools.product(
            #         range(x.shape[-1]),
            #         range(x.shape[0]))
            # ]

            # # Get source estimate from value of B
            # s_est = B@x

            # # Get part of likelihood that depends on observation
            # likelihood += (1/N)*np.sum([
            #     np.log(source_pdf_fn(s_est[i,t])) for t,i in likelihood_iterator
            # ])

            # # Evaluate prior
            # prior = np.log(prior_pdf_fn(B))/N

            # Get posterior value at B
            posteriori = self.log_posterior_fn(
                x=x,
                B=B
            )
            
            return {
                'i': i,
                'j': j,
                'log_posteriori': posteriori
            }
            

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
        

        # Get grid positions
        iterator = itertools.product(
            enumerate(self.u_vec),
            enumerate(self.v_vec)
        )

        # Wrapper function which will be used
        exec_fn = partial(
            __get_posteriori,
            A,
            x,
            N
        )
        
        # Execute posteriori calculations
        # with pathos.multiprocessing.ProcessingPool(njobs) as p:
        #     results = p.map(exec_fn, iterator)
        results = []
        for idx in iterator:
            results.append(exec_fn(idx))

        # Parse results
        z = np.empty(
            shape=(
                self.u_vec.shape[0],
                self.v_vec.shape[0]
            )
        )
        for r in results:
            i=r['i']
            j=r['j']
            z[i, j]=r['log_posteriori']

        # Save posterior grid
        self.posterior_grid = z

    
    def get_max_point(
        self
    ):
        # Point of MAP
        self.max_post_point = np.unravel_index(
            self.posterior_grid.argmax(), 
            self.posterior_grid.shape
        )
        
        # Posterior at max point
        self.max_post = self.posterior_grid[self.max_post_point]

        # Maximum u and v
        self.u_max = self.u_vec[self.max_post_point[0]]
        self.v_max = self.v_vec[self.max_post_point[1]]
        
        # print('-'*100)
        # print('Ponto de máximo: u={}, v={}'.format(self.u_max, self.v_max))
        # print('-'*100)