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
                return (-1/self.std*self.std)*np.subtract(B, self.center)
        else:
            def __get_derivative_adj_factor(
                B
            ):
                return (-1/self.std*self.std)*jnp.subtract(B, self.center)

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
        # Backend toggle: when True, the density is built with jax.numpy so that
        # it is differentiable / jit-traceable inside a JAX-based sampler.
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
        return Sigma

    def get(
        self
    ):
        if not self.use_jax:
            # Flatten center and pre-build shape matrix
            mu = np.matrix.flatten(np.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=p)

            def __multivariate_t_prior(
                B
            ):
                # Flatten B to a vector, since multivariate-t works on vectors
                b = np.matrix.flatten(np.asarray(B))
                # Get prior value from multivariate-t density
                prior = st.multivariate_t(
                    loc=mu,
                    shape=Sigma,
                    df=self.df
                ).pdf(b)
                return prior

            return __multivariate_t_prior
        else:
            # Flatten center and pre-build shape matrix (jax.numpy)
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
                # Flatten B to a vector, since multivariate-t works on vectors
                b = jnp.ravel(jnp.asarray(B))
                # Mahalanobis-type quadratic form
                diff = b - mu
                quad = diff @ (Sigma_inv @ diff)
                # Get prior value from multivariate-t density
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
        # NOTE: with use_jax=True you do not need this analytic derivative at all,
        # since autodiff differentiates `get()` directly. It is kept for the
        # standard-Python path and for backward compatibility.
        if not self.use_jax:
            # Flatten center and pre-build shape matrix
            mu = np.matrix.flatten(np.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=p)
            Sigma_inv = np.linalg.inv(Sigma)

            def __multivariate_t_prior_derivative(
                B
            ):
                # Flatten B to a vector
                b = np.matrix.flatten(np.asarray(B))
                # Mahalanobis-type quadratic form
                diff = np.subtract(b, mu)
                quad = diff @ (Sigma_inv @ diff)
                # Multiplicative factor from differentiating the density
                factor = -(self.df+p)/(self.df+quad)
                # Gradient vector, then reshape back to B's matrix shape
                grad_vector = self.get()(B=B)*factor*(Sigma_inv @ diff)
                return np.reshape(grad_vector, np.asarray(B).shape)

            return __multivariate_t_prior_derivative
        else:
            # Flatten center and pre-build shape matrix (jax.numpy)
            mu = jnp.ravel(jnp.asarray(self.center))
            p = mu.size
            Sigma = self.__get_shape_matrix(p=int(p))
            Sigma = jnp.asarray(Sigma)
            Sigma_inv = jnp.linalg.inv(Sigma)

            def __multivariate_t_prior_derivative(
                B
            ):
                # Flatten B to a vector
                b = jnp.ravel(jnp.asarray(B))
                # Mahalanobis-type quadratic form
                diff = jnp.subtract(b, mu)
                quad = diff @ (Sigma_inv @ diff)
                # Multiplicative factor from differentiating the density
                factor = -(self.df+p)/(self.df+quad)
                # Gradient vector, then reshape back to B's matrix shape
                grad_vector = self.get()(B=B)*factor*(Sigma_inv @ diff)
                return jnp.reshape(grad_vector, jnp.asarray(B).shape)

            return __multivariate_t_prior_derivative

