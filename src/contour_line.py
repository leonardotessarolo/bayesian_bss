import numpy as np
import itertools
from functools import partial
import pathos
from .utilities import PosteriorUtilities
import jax
import jax.numpy as jnp


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
        self.log_posterior_fn = PosteriorUtilities.get_log_posterior_fn(
            source_pdf_fn=source_pdf_fn,
            prior_pdf_fn=prior_pdf_fn,
            use_jax=True
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
        def __get_grid_matrices(
            B_true,
            idx_info
        ):
            # Parse index info
            i, u = idx_info[0]
            j, v = idx_info[-1]

            # Get shifted value of A which will be evaluated and corresponding value of B
            B = B_true + u*symmetric_basis + v*skew_symmetric_basis

            return {
                'i': i,
                'j': j,
                'B': B
            }
            
        
        def __get_posteriori_fn(
            x,
        ):  
            return lambda B: self.log_posterior_fn(
                        x=x,
                        B=B
                    )
            

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

        # Get grid matrices
        matrices_dicts = [
            __get_grid_matrices(
                B_true = np.linalg.inv(A),
                idx_info=idx_info
            ) for idx_info in iterator
        ]
        matrix_idxs = jnp.asarray([
            (d['i'], d['j']) for d in matrices_dicts
        ])
        B_matrices = jnp.asarray([
            d['B'] for d in matrices_dicts
        ])

        # Wrapper function which will be used on matrices
        # exec_fn = partial(
        #     self.log_posterior_fn,
        #     x
        # )

        
        
        # Execute posteriori calculations
        # with pathos.multiprocessing.ProcessingPool(njobs) as p:
        #     results = p.map(exec_fn, iterator)
        # results = []
        # for idx in iterator:
        #     results.append(exec_fn(idx))

        # Evaluate posteriors on grid using vectorized jax.vmap call
        exec_map = jax.vmap(
            partial(
                self.log_posterior_fn,
                x
            ),
            in_axes=0
        )
        results = exec_map(B_matrices)
        
        # Parse results
        z = np.empty(
            shape=(
                self.u_vec.shape[0],
                self.v_vec.shape[0]
            )
        )
        for idxs, r in zip(
            matrix_idxs,
            results
        ):
            i=idxs[0]
            j=idxs[-1]
            z[i, j]=r

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