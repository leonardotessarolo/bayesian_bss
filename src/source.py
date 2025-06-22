import numpy as np

class LogisticSource:
    def __init__(
        self,
        mu,
        sigma
    ):
        # Save attributes
        self.mu=mu
        self.sigma=sigma

    def get(
        self
    ):
        def __logistic_distribution(
            x
        ):
            """
                This method takes SCALAR value x and return SCALAR source pdf evaluated at x.
            """
            return np.exp(
                -(x-self.mu)/self.sigma
            )/(
                self.sigma*np.square(
                    1 + np.exp(-(x-self.mu)/self.sigma)
                )
            )
            
        return lambda x: __logistic_distribution(x=x)


    def get_derivative(
        self
    ):
        def __logistic_distribution_derivative(
            x
        ):
            """
                This method takes SCALAR value x and return SCALAR source pdf derivative evaluated at x.
            """
            return (
                2*np.exp(-2*x)*(1+np.exp(-x)) - np.exp(-x)*np.square(1+np.exp(-x))
            )/np.power(
                [1+np.exp(-x)],
                [4]
            )

        return lambda x: __logistic_distribution_derivative(x=x)

    def get_cumulative(
        self
    ):
        def __logistic_distribution_cumulative(
            x
        ):
            """
                This method takes SCALAR value x and return SCALAR source cdf evaluated at x.
            """
            return 1/(
                1 + np.exp(-(x-self.mu)/self.sigma)
            )
            
        return lambda x: __logistic_distribution_cumulative(x=x)
    
    
    def get_realization(
        self,
        nsources,
        nobs
    ):
        return np.random.logistic(
            loc=self.mu,
            scale=self.sigma,
            size=(nsources, nobs)
        )


class TriangularSource:
    def __init__(
        self,
        lower,
        upper,
        mode
    ):
        # Save attributes
        self.lower=lower
        self.upper=upper
        self.mode=mode

    def get(
        self
    ):
        pass
        # def __logistic_distribution(
        #     x
        # ):
        #     """
        #         This method takes SCALAR value x and return SCALAR source pdf evaluated at x.
        #     """
        #     return np.exp(
        #         -(x-self.mu)/self.sigma
        #     )/(
        #         self.sigma*np.square(
        #             1 + np.exp(-(x-self.mu)/self.sigma)
        #         )
        #     )
            
        # return lambda x: __logistic_distribution(x=x)

    def get_derivative(
        self
    ):
        pass
        # def __logistic_distribution_derivative(
        #     x
        # ):
        #     """
        #         This method takes SCALAR value x and return SCALAR source pdf derivative evaluated at x.
        #     """
        #     return (
        #         2*np.exp(-2*x)*(1+np.exp(-x)) - np.exp(-x)*np.square(1+np.exp(-x))
        #     )/np.power(
        #         [1+np.exp(-x)],
        #         [4]
        #     )

        # return lambda x: __logistic_distribution_derivative(x=x)

    def get_cumulative(
        self
    ):
        pass
        # def __logistic_distribution_cumulative(
        #     x
        # ):
        #     """
        #         This method takes SCALAR value x and return SCALAR source cdf evaluated at x.
        #     """
        #     return 1/(
        #         1 + np.exp(-(x-self.mu)/self.sigma)
        #     )
            
        # return lambda x: __logistic_distribution_cumulative(x=x)
    
    def get_realization(
        self,
        nsources,
        nobs
    ):
        return np.random.triangular(
            left=self.lower,
            mode=self.mode,
            right=self.upper,
            size=(nsources, nobs)
        )



