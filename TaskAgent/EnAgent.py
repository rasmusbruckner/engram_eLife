""" Agent: Implementation of the forgetting model """

import numpy as np
from scipy.stats import beta
from random import choices
from en_utilities import safe_div


class Agent:
    """ This class definition specifies the properties of the object that implements the forgetting model """

    def __init__(self, agent_vars):
        """ This function creates an agent object of class Agent based on the agent initialization input """

        # Set variable task properties based on input
        self.beta = agent_vars.beta
        self.mu = agent_vars.mu
        self.kappa = agent_vars.kappa
        self.a = agent_vars.a
        self.b = agent_vars.b
        self.alpha_pos = agent_vars.alpha_pos
        self.alpha_neg = agent_vars.alpha_neg
        self.e = agent_vars.e
        self.delta = agent_vars.delta

    def compute_delta(self, outcome):
        """ This function computes the prediction error

        :param outcome: Presented object
        :return: None
        """

        self.delta = outcome - self.e

    def learn(self):
        """ This function implements the engram-updating computations of the forgetting model """

        # Determine learning rate depending on sign of prediction error
        if self.delta >= 0:
            alpha = self.alpha_pos
        else:
            alpha = self.alpha_neg

        # Update engram relevancy
        self.e = self.e + alpha * self.delta
    
    def softmax(self):
        """ This function implements the softmax modeling exploration probability """

        self.mu = safe_div(np.exp(np.dot([0, self.e], self.beta)), np.sum(np.exp(np.dot([0, self.e], self.beta))))

    def shape_params(self):
        """ This function computes the shape parameters a and b of the beta distribution

            We translate the mu parameter reflecting the mean exploration probability into a and b to make
            use of the beta distribution for modeling behavioral variability. The higher the
            kappa parameter, the lower the variability of the distribution.

            α = μν, β = (1 − μ)ν

            For more background, see:
                - https://en.wikipedia.org/wiki/Beta_distribution  alternative parametrizations
                - https://nyu-cdsc.github.io/learningr/assets/kruschke_bayesian_in_R.pdf (page 129)
        """

        self.a = self.mu[1] * self.kappa
        self.b = (1-self.mu[1]) * self.kappa

    def sample_exploration_prob(self):
        """ This function is used in simulations to sample exploration behavior based on
            the a and b parameters and the beta PDF

        :return: disc_choices: Sampled exploration probabilities
        """

        # Grid from low to high probability
        x_prob = np.linspace(0.001, 0.999, 100)

        # Normalized beta distribution
        probs = beta.pdf(x_prob, self.a, self.b)/np.nansum(beta.pdf(x_prob, self.a, self.b))

        # Return sample from the distribution
        return choices(x_prob, probs)[0]
