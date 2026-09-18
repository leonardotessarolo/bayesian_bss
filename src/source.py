import numpy as np
import math
import scipy.stats as st
from scipy.special import gamma
import jax.numpy as jnp

class LogisticSource:
    def __init__(
        self,
        mu,
        sigma,
        use_jax=False
    ):
        # Save attributes
        self.mu=mu
        self.sigma=sigma
        # Backend toggle. Standard-Python path keeps the original SCALAR pdf
        # (evaluated element-by-element). JAX path returns an ELEMENTWISE pdf that
        # can be applied to the whole (nsources, nobs) array at once and is
        # differentiable / jit-traceable.
        self.use_jax=use_jax

    def get(
        self
    ):
        if not self.use_jax:
            def __logistic_distribution(
                x
            ):
                """
                    This method takes value x and return SCALAR source pdf evaluated at x.
                """
                arg = math.exp(
                    -(x-self.mu)/self.sigma
                )
                return arg/(
                    self.sigma*((1+arg)*(1+arg))
                )
            return lambda x: __logistic_distribution(x=x)
        else:
            def __logistic_distribution(
                x
            ):
                """
                    This method takes x and returns the ELEMENTWISE source pdf (jax.numpy).
                """
                arg = jnp.exp(
                    -(x-self.mu)/self.sigma
                )
                return arg/(
                    self.sigma*((1+arg)*(1+arg))
                )
            return lambda x: __logistic_distribution(x=x)

    def get_derivative(
        self
    ):
        if not self.use_jax:
            def __logistic_distribution_derivative(
                x
            ):
                """
                    This method takes  value x and return SCALAR source pdf derivative evaluated at x.
                """
                # Calculate repeated subexpressions
                arg1 = math.exp(
                    -(x-self.mu)/self.sigma
                )
                arg2 = 1 + arg1
                return (
                    (2*(arg1*arg1)*arg2) - (arg1*arg2*arg2)
                )/(
                    (self.sigma*self.sigma)*pow(arg2, 4)
                )
            return lambda x: __logistic_distribution_derivative(x=x)
        else:
            def __logistic_distribution_derivative(
                x
            ):
                """
                    This method takes value x and return SCALAR source pdf derivative evaluated at x.
                """
                # Calculate repeated subexpressions
                import pdb; pdb.set_trace()
                arg1 = jnp.exp(
                    -(x-self.mu)/self.sigma
                )
                arg2 = 1 + arg1
                return (
                    (2*(arg1*arg1)*arg2) - (arg1*arg2*arg2)
                )/(
                    (self.sigma*self.sigma)*(arg2*arg2*arg2*arg2)
                )
            print('----')
            return lambda x: jnp.vectorize(__logistic_distribution_derivative(x=x))
        

    def get_cumulative(
        self
    ):
        if not self.use_jax:
            def __logistic_distribution_cumulative(
                x
            ):
                """
                    This method takes SCALAR value x and return SCALAR source cdf evaluated at x.
                """
                return 1/(
                    1 + math.exp(-(x-self.mu)/self.sigma)
                )
            return lambda x: __logistic_distribution_cumulative(x=x)
        else:
            def __logistic_distribution_cumulative(
                x
            ):
                """
                    This method takes an ARRAY x and returns the ELEMENTWISE source cdf (jax.numpy).
                """
                return 1/(
                    1 + jnp.exp(-(x-self.mu)/self.sigma)
                )
            return lambda x: __logistic_distribution_cumulative(x=x)

    def get_realization(
        self,
        nsources,
        nobs,
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
        return np.random.logistic(
            loc=self.mu,
            scale=self.sigma,
            size=(nsources, nobs)
        )


class StandardLogisticSource:
    def __init__(
        self,
        use_jax=False
    ):
        self.use_jax=use_jax

    def get(
        self
    ):
        if not self.use_jax:
            def __logistic_distribution(
                x
            ):
                """
                    This method takes SCALAR value x and return SCALAR source pdf evaluated at x.
                """
                arg = math.exp(-x)
                return arg/((1+arg)*(1+arg))
            return lambda x: __logistic_distribution(x=x)
        else:
            def __logistic_distribution(
                x
            ):
                """
                    This method takes an ARRAY x and returns the ELEMENTWISE source pdf (jax.numpy).
                """
                arg = jnp.exp(-x)
                return arg/((1+arg)*(1+arg))
            return lambda x: __logistic_distribution(x=x)

    def get_derivative(
        self
    ):
        if not self.use_jax:
            def __logistic_distribution_derivative(
                x
            ):
                """
                    This method takes value x and return SCALAR source pdf derivative evaluated at x.
                """
                arg = 1+math.exp(-x)
                return (
                    2*math.exp(-2*x)*(arg) - math.exp(-x)*(arg*arg)
                )/pow(arg, 4)
            return lambda x: __logistic_distribution_derivative(x=x)
        else:
            def __logistic_distribution_derivative(
                x
            ):
                """
                    This method takes value x and return SCALAR source pdf derivative evaluated at x.
                """
                arg = 1+jnp.exp(-x)
                return (
                    2*jnp.exp(-2*x)*(arg) - jnp.exp(-x)*(arg*arg)
                )/pow(arg, 4)
            return lambda x: __logistic_distribution_derivative(x=x)

    def get_cumulative(
        self
    ):
        if not self.use_jax:
            def __logistic_distribution_cumulative(
                x
            ):
                """
                    This method takes SCALAR value x and return SCALAR source cdf evaluated at x.
                """
                return 1/(1 + math.exp(-x))
            return lambda x: __logistic_distribution_cumulative(x=x)
        else:
            def __logistic_distribution_cumulative(
                x
            ):
                """
                    This method takes an ARRAY x and returns the ELEMENTWISE source cdf (jax.numpy).
                """
                return 1/(1 + jnp.exp(-x))
            return lambda x: __logistic_distribution_cumulative(x=x)

    def get_realization(
        self,
        nsources,
        nobs,
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
        return np.random.logistic(
            loc=0,
            scale=1,
            size=(nsources, nobs)
        )

    
class StudentTSource:
    def __init__(
        self,
        mu,
        sigma,
        df
    ):
        # Save attributes
        self.mu=mu
        self.sigma=sigma
        self.df=df

    def get(
        self
    ):
            
        return lambda x: st.t.pdf(
            x=x,
            df=self.df,
            loc=self.mu,
            scale=self.sigma
        )


    def get_derivative(
        self
    ):
        def __t_distribution_derivative(
            x
        ):
            """
                This method takes SCALAR value x and return SCALAR source pdf derivative evaluated at x.
            """

            # Calculate repeated subexpression
            arg1 = (x-self.mu)/(self.df*self.sigma)
            
            # Calculate multiplicative parameter
            mult = gamma((self.df+1)/2)/(gamma(self.df/2)*math.sqrt(math.pi*self.df*self.sigma*self.sigma))
            mult = mult*0.5*(-self.df-1)

            # Calculate power parameter
            p = math.pow(
                1+arg1*arg1*self.df,
                (-self.df-3)/2
            )*2*arg1/self.sigma

            return mult * p

        return lambda x: __t_distribution_derivative(x=x)

    def get_cumulative(
        self
    ):
            
        return lambda x: st.t.cdf(
            x=x,
            df=self.df,
            loc=self.mu,
            scale=self.sigma
        )
    
    
    def get_realization(
        self,
        nsources,
        nobs,
        seed=None
    ):
        return st.t.rvs(
            df=self.df,
            loc=self.mu,
            scale=self.sigma,
            size=(nsources, nobs),
            random_state=seed
        )
    

class TriangularSource:
    def __init__(
        self,
        lower,
        upper,
        mode,
        use_jax=False
    ):
        # Save attributes
        self.lower=lower
        self.upper=upper
        self.mode=mode
        self.use_jax = use_jax

    def get(
        self
    ):
        pass

    def get_derivative(
        self
    ):
        pass

    def get_cumulative(
        self
    ):
        pass
    
    def get_realization(
        self,
        nsources,
        nobs,
        seed
    ):
        if seed is not None:
            np.random.seed(seed)
            
        return np.random.triangular(
            left=self.lower,
            mode=self.mode,
            right=self.upper,
            size=(nsources, nobs)
        )

    



