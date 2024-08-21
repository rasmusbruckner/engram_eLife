""" AgentVars: Initialization of the forgetting model """


import numpy as np


class AgentVars:
    """ This class specifies the task parameters of the retroactive interference task """

    def __init__(self):
        """ this function defines the instance variables unique to each instance """

        self.beta = -0.5  # slope of the softmax function
        self.mu = [0.5, 0.5]  # average exploration probability
        self.kappa = 20  # concentration of the beta distribution
        self.a = 1  # alpha shape parameter of beta distribution
        self.b = 1  # beta shape parameter of beta distribution
        self.alpha_pos = 1  # learning rate positive PEs
        self.alpha_neg = 1  # learning rate negative PEs
        self.e = 0  # engram value
        self.delta = np.nan  # prediction error
