import numpy as np

class ExponentialPrior:
    def __init__(
        self,
        center,
        std
    ):
        # Save attributes
        self.center=center
        self.std=std

    def get(
        self
    ):
        def __exponential_prior(
            B            
        ):
            # Get norm of difference vector
            norm = np.square(
                np.linalg.norm(
                    x=np.subtract(B, self.center),
                    ord='fro'
                )
            )
            
            # Get prior value
            prior = np.exp(
                -0.5*np.square(1/self.std)*norm
            )
            
            return prior
            
        return __exponential_prior

    def get_derivative(
        self
    ):
        def __get_derivative_adj_factor(
            B
        ):
            return (-1/self.std)*np.subtract(B, self.center)

        # Get prior derivative
        prior_derivative = lambda B: self.get()(B=B)*__get_derivative_adj_factor(B=B)

        return prior_derivative



