import numpy as np
import jax.numpy as jnp
import math
import scipy.stats as st
from jax.scipy.special import gammaln

class ExponentialPrior:
    def __init__(
        self,
        center,
        std,
        use_jax=False
    ):
        # Save attributes
        self.center=center
        self.std=std
        # Backend toggle: when True, densities are built with jax.numpy so that
        # they are differentiable / jit-traceable inside a JAX-based sampler.
        self.use_jax=use_jax

    def get(
        self
    ):
        if not self.use_jax:
        # if True:
            def __exponential_prior(
                B
            ):
                # Get norm of difference vector
                diff = np.linalg.norm(
                    x=np.subtract(B, self.center),
                    ord='fro'
                )
                norm = diff*diff
                # Get prior value
                prior = math.exp(
                    -0.5*norm/self.std/self.std
                )
                return prior
        else:
            def __exponential_prior(
                B
            ):
                # Get norm of difference vector (Frobenius) with jax.numpy
                diff = jnp.linalg.norm(
                    jnp.subtract(B, self.center),
                    ord='fro'
                )
                norm = diff*diff
                # Get prior value
                prior = jnp.exp(
                    -0.5*norm/self.std/self.std
                )
                return prior

        return __exponential_prior

    def get_derivative(
        self
    ):
        if not self.use_jax:
            def __get_derivative_adj_factor(
                B
            ):
                return -1/(self.std*self.std)*np.subtract(B, self.center)
        else:
            def __get_derivative_adj_factor(
                B
            ):
                return -1/(self.std*self.std)*jnp.subtract(B, self.center)

        # Get prior derivative
        prior_derivative = lambda B: self.get()(B=B)*__get_derivative_adj_factor(B=B)
        return prior_derivative



class MultivariateTPrior:
    def __init__(
        self,
        center,
        scale,
        df,
        use_jax=False
    ):
        # Save attributes
        self.center=center
        self.scale=scale
        self.df=df
        self.use_jax=use_jax
    def __get_shape_matrix(
        self,
        p
    ):
        """
            Builds the (p x p) shape (scale) matrix Sigma of the multivariate-t from
            the `scale` attribute. A scalar `scale` is treated as an isotropic std,
            so Sigma = scale^2 * I (analogous to `std` in ExponentialPrior); a (p x p)
            `scale` is used directly as the shape matrix.
        """
        if np.ndim(self.scale)==0:
            Sigma = (self.scale*self.scale)*np.eye(p)
        else:
            Sigma = np.asarray(self.scale)

        if self.use_jax:
            Sigma = jnp.asarray(Sigma)
        return Sigma
    
    def __prepare_batch_numpy(
        self,
        B,
        center_shape,
        p
    ):
        """
            Reshapes a raw `B` (NumPy) into shape (..., p), where the trailing
            dimensions match `center`'s own shape and any remaining leading
            dimensions are treated as a batch. Returns (b, batch_shape) so
            that results can be reshaped back to B's own shape after
            evaluation. Accepts either a single `B` matching `center`'s shape,
            or a batch of `B`'s with `center`'s shape as trailing dimensions
            (any number of leading batch dimensions is supported).
        """
        B_arr = np.asarray(B)
        ndim_center = len(center_shape)
        if B_arr.shape==center_shape:
            return B_arr.reshape(p), ()
        trailing = B_arr.shape[-ndim_center:] if ndim_center>0 else ()
        if B_arr.ndim>ndim_center and trailing==center_shape:
            batch_shape = B_arr.shape[:B_arr.ndim-ndim_center]
            return B_arr.reshape(batch_shape+(p,)), batch_shape
        raise ValueError(
            f"`B` has shape {B_arr.shape}, which is incompatible with "
            f"`center`'s shape {center_shape}. `B` must either match "
            f"`center`'s shape exactly (single evaluation) or have "
            f"`center`'s shape as its trailing dimensions, with one or more "
            f"leading batch dimensions."
        )
    def __prepare_batch_jax(
        self,
        B,
        center_shape,
        p
    ):
        """
            JAX counterpart of __prepare_batch_numpy. Reshapes a raw `B`
            (jax.numpy) into shape (..., p), where the trailing dimensions
            match `center`'s own shape and any remaining leading dimensions
            are treated as a batch. Shape checks are performed on static
            shapes only, so this remains jit/vmap-traceable.
        """
        B_arr = jnp.asarray(B)
        ndim_center = len(center_shape)
        if B_arr.shape==center_shape:
            return jnp.reshape(B_arr, (p,)), ()
        trailing = B_arr.shape[-ndim_center:] if ndim_center>0 else ()
        if B_arr.ndim>ndim_center and trailing==center_shape:
            batch_shape = B_arr.shape[:B_arr.ndim-ndim_center]
            return jnp.reshape(B_arr, batch_shape+(p,)), batch_shape
        raise ValueError(
            f"`B` has shape {B_arr.shape}, which is incompatible with "
            f"`center`'s shape {center_shape}. `B` must either match "
            f"`center`'s shape exactly (single evaluation) or have "
            f"`center`'s shape as its trailing dimensions, with one or more "
            f"leading batch dimensions."
        )
    def get(
        self
    ):
        if not self.use_jax:
            # Flatten center and pre-build shape matrix / frozen rv (once)
            center_shape = np.asarray(self.center).shape
            mu = np.ravel(np.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=p)
            rv = st.multivariate_t(
                loc=mu,
                shape=Sigma,
                df=self.df
            )
            def __multivariate_t_prior(
                B
            ):
                """
                    Evaluates the multivariate-t density at `B`. Accepts a
                    single `B` matching `center`'s shape (returns a scalar),
                    or a batch of `B`'s with `center`'s shape as trailing
                    dimensions and any leading batch shape (returns an array
                    of pdf values with that batch shape).
                """
                # Reshape B to (..., p), keeping any leading batch dims
                b, batch_shape = self.__prepare_batch_numpy(
                    B=B,
                    center_shape=center_shape,
                    p=p
                )
                # Get prior value(s) from multivariate-t density
                prior = rv.pdf(b)
                return prior
            return __multivariate_t_prior
        else:
            # Flatten center and pre-build shape matrix (jax.numpy)
            center_shape = jnp.asarray(self.center).shape
            mu = jnp.ravel(jnp.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=int(p))
            Sigma = jnp.asarray(Sigma)
            Sigma_inv = jnp.linalg.inv(Sigma)
            sign, logdet = jnp.linalg.slogdet(Sigma)
            # Normalizing constant (log) of the multivariate-t density
            log_const = (
                gammaln((self.df+p)/2)
                - gammaln(self.df/2)
                - 0.5*p*jnp.log(self.df*jnp.pi)
                - 0.5*logdet
            )
            def __multivariate_t_prior(
                B
            ):
                """
                    Evaluates the multivariate-t density at `B` using JAX.
                    Accepts a single `B` matching `center`'s shape (returns a
                    scalar), or a batch of `B`'s with `center`'s shape as
                    trailing dimensions and any leading batch shape (returns
                    an array of pdf values with that batch shape). Fully
                    jit/vmap-traceable.
                """
                # Reshape B to vector, keeping leading dims
                b, batch_shape = self.__prepare_batch_jax(
                    B=B,
                    center_shape=center_shape,
                    p=p
                )
                # Mahalanobis-type quadratic form, batched: contracts only
                # the trailing p axis, leaving any leading batch axes intact.
                diff = b - mu
                quad = jnp.einsum(
                    '...i,ij,...j->...',
                    diff,
                    Sigma_inv,
                    diff
                )
                # Get prior value(s) from multivariate-t density
                prior = jnp.exp(
                    log_const
                )*jnp.power(
                    1 + quad/self.df,
                    -(self.df+p)/2
                )
                return prior
            return __multivariate_t_prior


    def get_derivative(
        self
    ):
        if not self.use_jax:
            # Flatten center and pre-build shape matrix / frozen rv (once)
            center_shape = np.asarray(self.center).shape
            mu = np.ravel(np.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=p)
            Sigma_inv = np.linalg.inv(Sigma)
            rv = st.multivariate_t(
                loc=mu,
                shape=Sigma,
                df=self.df
            )
            def __multivariate_t_prior_derivative(
                B
            ):
                """
                    Evaluates the (vectorized) gradient of the
                    multivariate-t density at `B`. Accepts a single `B`
                    matching `center`'s shape, or a batch of `B`'s with
                    `center`'s shape as trailing dimensions and any leading
                    batch shape. Returns an array of gradients with `B`'s own
                    shape.
                """
                # Reshape B to (..., p), keeping any leading batch dims
                b, batch_shape = self.__prepare_batch_numpy(
                    B=B,
                    center_shape=center_shape,
                    p=p
                )
                # Mahalanobis-type quadratic form, batched
                diff = np.subtract(b, mu)
                quad = np.einsum(
                    '...i,ij,...j->...',
                    diff,
                    Sigma_inv,
                    diff
                )
                # Get prior value(s) from multivariate-t density
                prior = rv.pdf(b)
                # Multiplicative factor from differentiating the density
                factor = -(self.df+p)/(self.df+quad)
                # Batched matrix-vector product, contracting only the
                # trailing p axis; leaves any leading batch axis untouched.
                Sigma_inv_diff = np.einsum(
                    'ij,...j->...i',
                    Sigma_inv,
                    diff
                )
                # Gradient vector(s), then reshape back to B's own shape
                grad_vector = (
                    np.expand_dims(np.asarray(prior), axis=-1)
                    *np.expand_dims(np.asarray(factor), axis=-1)
                    *Sigma_inv_diff
                )
                return np.reshape(grad_vector, np.asarray(B).shape)
            return __multivariate_t_prior_derivative
        else:
            # Flatten center and pre-build shape matrix (jax.numpy)
            center_shape = jnp.asarray(self.center).shape
            mu = jnp.ravel(jnp.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=int(p))
            Sigma = jnp.asarray(Sigma)
            Sigma_inv = jnp.linalg.inv(Sigma)
            sign, logdet = jnp.linalg.slogdet(Sigma)
            # Normalizing constant (log) of the multivariate-t density, reused
            # here so the derivative does not need to re-invoke get() (and
            # therefore does not rebuild Sigma/Sigma_inv/log_const per call).
            log_const = (
                gammaln((self.df+p)/2)
                - gammaln(self.df/2)
                - 0.5*p*jnp.log(self.df*jnp.pi)
                - 0.5*logdet
            )
            def __multivariate_t_prior_derivative(
                B
            ):
                """
                    Evaluates the (vectorized) analytical gradient of the
                    multivariate-t density at `B` using JAX. Accepts a single
                    `B` matching `center`'s shape, or a batch of `B`'s with
                    `center`'s shape as trailing dimensions and any leading
                    batch shape. Returns an array of gradients with `B`'s own
                    shape. Fully jit/vmap-traceable.
                """
                # Reshape B to (..., p), keeping any leading batch dims
                b, batch_shape = self.__prepare_batch_jax(
                    B=B,
                    center_shape=center_shape,
                    p=p
                )
                # Mahalanobis-type quadratic form, batched
                diff = jnp.subtract(b, mu)
                quad = jnp.einsum(
                    '...i,ij,...j->...',
                    diff,
                    Sigma_inv,
                    diff
                )
                # Get prior value(s) from multivariate-t density
                prior = jnp.exp(
                    log_const
                )*jnp.power(
                    1 + quad/self.df,
                    -(self.df+p)/2
                )
                # Multiplicative factor from differentiating the density
                factor = -(self.df+p)/(self.df+quad)
                # Batched matrix-vector product, contracting only the
                # trailing p axis; leaves any leading batch axis untouched.
                Sigma_inv_diff = jnp.einsum(
                    'ij,...j->...i',
                    Sigma_inv,
                    diff
                )
                # Gradient vector(s), then reshape back to B's own shape
                grad_vector = (
                    jnp.expand_dims(prior, axis=-1)
                    *jnp.expand_dims(factor, axis=-1)
                    *Sigma_inv_diff
                )
                return jnp.reshape(grad_vector, jnp.asarray(B).shape)
            return __multivariate_t_prior_derivative

