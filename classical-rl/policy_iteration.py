import gym
import numpy as np


class PolicyIteration(object):
    def __init__(self,
                 ENV_NAME: str,
                 GAMMA: float,
                 iterations=1000):
        """
        Value iteration class

        Args:
            ENV_NAME (string): Gym env name
            GAMMA (int): discount factor
            iterations (int, optional): number of iteration to optimize value table. Defaults to 1000.
        """
        self.env = gym.make(ENV_NAME)
        self.env_name = ENV_NAME

        # discount factor
        self.gamma = GAMMA

        # to compare value tables
        self.cuttoff = 1e-10
        # number of iterations used to optimize value table
        self.n_iterations = iterations

        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        # initialize value table and policy randomly
        self.initial_value_table = np.zeros(self.state_space)

        # initialize policy with zeros, means 1st action from action space
        # it could be any action from the action state
        self.initial_policy = np.zeros(self.state_space)

        # at start, initial and final policy is same, the policy is gradually optimized
        self.final_policy = np.copy(self.initial_policy)

        # reset the environment on initialization
        self.env.reset()

    def value_table_calculation(self, policy):
        """Optimize value table
        Unlike Value Iteration, here we need to provide a policy which will be 
        used to optimize the value table for the given policy

        Args:
            policy (arrary): policy using which value table will be optimized

        Returns:
            array: optimal value table
        """

        for iteration in range(self.n_iterations):
            # with every iteration, the final value table will initiated with best values from last iteration
            # this way with every iteration, the value table get optimized
            self.final_value_table = np.copy(self.initial_value_table)

            # remember, we need the q_values for each state for every action,
            # that's why no for loop for action here but within the list comprehension
            for state in range(self.state_space):

                # action will be fetched using provided policy
                action = policy[state]

                q_values = [sum([prob * (reward + self.gamma * self.final_value_table[next_state])
                                for prob, next_state, reward, info in self.env.P[state][action]])]

                # store the maximum q_value for next iteration
                self.initial_value_table[state] = max(q_values)

            # compare values for each state, if they are almost same, no need to iterate further
            # np.fabs is used to compare element wise comparison
            if np.sum(np.fabs(self.final_value_table - self.initial_value_table)) <= self.cuttoff:
                break
        return self.final_value_table

    def policy_extraction(self, value_table):
        """Extration optimal policy

        Args:
            value_table (array): optimized value table

        Returns:
            array: optimal policy
        """
        for state in range(self.state_space):
            self.final_q_table = [sum([prob * (reward + self.gamma * value_table[next_state])
                                       for prob, next_state, reward, info in self.env.P[state][action]])
                                  for action in range(self.action_space)]
            # retrieve the optimal action for each state,
            # and this is nothing but optimal policy
            self.final_policy[state] = np.argmax(np.array(self.final_q_table))
        return

    def policy_optimization(self):
        """Main function for policy iteration method
        which returns the optimal policy

        Returns:
            _type_: _description_
        """
        for iteration in range(self.n_iterations):
            self.initial_policy = np.copy(self.final_policy)
            optimal_value_table = self.value_table_calculation(
                self.initial_policy)

            self.policy_extraction(optimal_value_table)

            if np.sum(np.fabs(self.final_policy - self.initial_policy)) == 0:
                break
        print(f'Final policy for the {self.env_name}:\n {self.final_policy}')
        return self.final_policy

    def understand_agent_transition(self, policy):
        """Illustrate how the action state transition happens using optimal policy

        Args:
            policy (arrary): optimal policy
        """
        for state, action in enumerate(policy):
            print(f'State: {state}, Action: {action}')
            print(self.env.P[state][action])
            print()


if __name__ == "__main__":

    # any model based, stochastic environment can be provided here
    GYM_ENV = 'FrozenLake8x8-v1'

    # discount factor, try different values, 0.9, 0.8
    GAMMA = 0.9

    policy_iteration = PolicyIteration(GYM_ENV, GAMMA)
    optimal_policy = policy_iteration.policy_optimization()
    policy_iteration.understand_agent_transition(optimal_policy)
