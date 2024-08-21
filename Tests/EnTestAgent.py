""" EnTestAgent: This class runs the unit tests for the agent """

import numpy as np
import unittest
from TaskAgent.EnAgentVars import AgentVars
from TaskAgent.EnAgent import Agent
import random


class TestAgent(unittest.TestCase):
    """ This class definition implements the Agent unit tests
        in order to test the critical learning functions

        We have the following test functions

            test_agent_init: Agent initialization parameters
            test_learn: Agent learning function
            test_compute_delta: Prediction-error function
            test_softmax: Softmax function used for modeling mean exploration behavior
            test_shape_params: Function that translates mean exploration probability
                to a and b parameter to model behavioral variability
            test_sample_exploration_prob: Sampling exploration probability based on a and b parameters
    """

    def test_agent_init(self):
        """ This function tests the agent initialization based on AgentVars """

        agent_vars = AgentVars()
        agent = Agent(agent_vars)

        self.assertEqual(agent.beta, -0.5)
        self.assertEqual(agent.mu, [0.5, 0.5])
        self.assertEqual(agent.kappa, 20)
        self.assertEqual(agent.a, 1)
        self.assertEqual(agent.b, 1)
        self.assertEqual(agent.alpha_pos, 1)
        self.assertEqual(agent.alpha_neg, 1)
        self.assertEqual(agent.e, 0)
        self.assertTrue(np.isnan(agent.delta))

    def test_learn(self):
        """ This function tests the learning function of the agent model

            We test both positive and negative PEs to test the different LRs.
        """

        # Agent-initialization-object and agent-object instances
        agent_vars = AgentVars()
        agent = Agent(agent_vars)

        # Set test parameters for negative-PE test
        agent.e = 0.5  # engram relevancy
        agent.alpha_neg = 0.1  # learning rate on negative PEs
        agent.delta = -0.5  # PE

        # Run learning function and test updated engram relevancy
        agent.learn()
        self.assertEqual(agent.e, 0.45)

        # Set test parameters for positive-PE test
        agent.e = 0.5
        agent.alpha_pos = 0.2
        agent.delta = 0.5

        # Run learning function and test updated engram relevancy
        agent.learn()
        self.assertEqual(agent.e, 0.6)

    def test_compute_delta(self):
        """ This function tests the PE function """

        # Agent-initialization-object and agent-object instances
        agent_vars = AgentVars()
        agent = Agent(agent_vars)

        # Set test parameters for positive-PE test
        agent.e = 0.5
        outcome = 1

        # Run PE function and test result
        agent.compute_delta(outcome)
        self.assertEqual(agent.delta, 0.5)

        # Set test parameters for negative-PE test
        agent.e = 0.5
        outcome = 0

        # Run PE function and test result
        agent.compute_delta(outcome)
        self.assertEqual(agent.delta, -0.5)

    def test_softmax(self):
        """ This function tests the softmax function modeling mean exploration probability """

        # Agent-initialization-object and agent-object instances
        agent_vars = AgentVars()
        agent = Agent(agent_vars)

        # Set test parameters
        agent.beta = -0.7
        agent.e = 0.5

        # Run softmax function and test result
        agent.softmax()
        self.assertAlmostEqual(agent.mu[0], 0.58661758, 6)
        self.assertAlmostEqual(agent.mu[1], 0.41338242, 6)

        # Set test parameters for flat probabilites
        agent.beta = 0
        agent.e = 0.5

        # Run softmax function and test result
        agent.softmax()
        self.assertEqual(agent.mu[0], 0.5)
        self.assertEqual(agent.mu[1], 0.5)

    def test_shape_params(self):
        """ This function tests the shape-parameter function """

        # Agent-initialization-object and agent-object instances
        agent_vars = AgentVars()
        agent = Agent(agent_vars)

        # Set test parameters
        agent.mu = [0.4, 0.6]
        agent.kappa = 20

        # Run shape-parameter function and test a, b, and resulting mean
        agent.shape_params()
        self.assertEqual(agent.a, 12)
        self.assertEqual(agent.b, 8)
        self.assertEqual(agent.a/(agent.a + agent.b), 0.6)

    def test_sample_exploration_prob(self):
        """ This function tests the sampling function that generates exploration behavior """

        # Agent-initialization-object and agent-object instances
        agent_vars = AgentVars()
        agent = Agent(agent_vars)

        # Control random-number generator for reproducible results
        random.seed(123)

        # Set test parameters
        agent.a = 12
        agent.b = 8

        # Run function and test sampled exploration probability
        disc_choices = agent.sample_exploration_prob()
        self.assertAlmostEqual(disc_choices, 0.4243939393939394, 6)


# Run unit test
if __name__ == '__main__':
    unittest.main()