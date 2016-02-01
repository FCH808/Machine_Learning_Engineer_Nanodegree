import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import defaultdict
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_vals = defaultdict(lambda: 10)
        self.state_visits = defaultdict(lambda: 0)
        self.state_action_visits = defaultdict(lambda: 0)
        self.alpha = 0.1 # Learning rate
        self.discount = 0.9 # Future discount factor
        self.total_reward = 0.0 # Cumulative reward
        self.diagnostics = {}
        self.n_trials = 0 # Counter for n_trials
        self.epsilon = 0.05 # Random action chance


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs

        if t == 0:
            self.n_trials += 1
            self.diagnostics[self.n_trials] = 'Fail!'

        # Update state, defined below
        state = self.getState()


        # Select action according to policy
        # action = random.choice(self.env.valid_actions)
        action = self.computeActionFromQValues(state, stochastic=True)

        # Track how many times states were visited to weight unvisited states
        #  more heavily
        self.state_visits[state] += 1
        self.state_action_visits[(state, action)] += 1

        # Get current Q(s,a) value
        current_Q_sa = self.getQValue(state, action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if not reward >= 10:
            # If the destination was reached, don't update the Q-values since
            #  the nexy_waypoint is invalid and None for the next_state.
            # Q-values table would just have extra invalid 'terminal' states
            #  with default scores added (which can never be updated under this
            #  current MDP implementation).

            # After the Action/Update (self.env.act()) acts on the environment
            next_state = self.getState()

            # ~ Q-VALUE UPDATE ~
            # Update Q(s, a) with a discounted future max(Q(s') values)
            # Add an exploration bonus to unvisited Q-values, computeValueFromQValues(.., exploration_bonus=True)
            # Note: Unneeded in our formulation since we started with
            #  optimistic Q values to encourage exploration.
            new_Q_sa = (1 - self.alpha) * current_Q_sa + self.alpha * (reward + self.discount * self.computeValueFromQValues(next_state, exploration_bonus=False))

            # Update our Q-value table
            self.Q_vals[(state, action)] = new_Q_sa

        self.total_reward += reward

        if reward < 0:
            print ""
            print "NEGATIVE!"
            print reward
            print ""

        if reward >= 10:
            # Print out current trial record each time destination is reached.
            # This is presumably when the target has been reached.
            self.diagnostics[self.n_trials] = reward
            print "-"*30 + ' Target Found! ' + "-"*30

            #print self.Q_vals
        if self.n_trials == 100:
            # Print out statistics after last trial
            print self.Q_vals
            print "*"*80
            print self.diagnostics
            print "*"*80
            print 'Total Cumulative Rewards {}'.format(self.total_reward)
            #print self.state_visits
            print len(self.state_action_visits), len(self.state_visits)

    def getState(self):
        ''' Returns tuple of tuples describing the current state of the system.

        Returns:
            state (tuple): Tuple of tuples describing the current state
                of the system.

        '''
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        # deadline = self.env.get_deadline(self)
        # green_light = 1 if inputs['light'] == 'green' else 0

        # Light status and next_waypoint seem sufficient to learn very fast
        #  (in 1-2 trials) under certain initial learning conditions.
        state = (('light', inputs['light']),
                 ('next_waypoint', self.next_waypoint))
        if self.next_waypoint == None:
            print self.n_trials, "*" * 80

        return state

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return a default value if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        return self.Q_vals[(state, action)]

    def computeValueFromQValues(self, state, exploration_bonus=False):
        ''' Returns the max value of the Q-values of the valid actions in a
        particular state.


        Args:
            state (tuple): Tuple of tuples describing the current state
                of the system.
            exploration_bonus (Boolean): Whether to use an exploration bonus to
             encourage unexplored actions.

        Returns:
            max_Q (float): Max Q-value of valid actions in current state.

        '''
        max_Q = float('-inf')
        all_actions = self.env.valid_actions
        random.shuffle(all_actions)

        for action in all_actions:
            current_Q = self.getQValue(state, action)

            # Artificially inflate Q-values if they have not been visited many times
            if exploration_bonus:
                state_action_visits = self.state_visits[(state, action)]
                state_action_visits = 1 if state_action_visits == 0 else state_action_visits

                current_Q += float(0.01)/state_action_visits
                # print 'Current: ', current_Q, ' | max: ', max_Q

            if current_Q > max_Q:
                max_Q = current_Q

        return max_Q

    def computeActionFromQValues(self, state, stochastic=False):
        ''' Returns the argmax action for the valid actions based on the stored
        Q-values for the state that is passed in.

        Can also add a random action choice chance with the 'stochastic' flag.

        Args:
            state (tuple): Tuple of tuples describing the current state
                of the system.
            stochastic (Boolean): Whether to add a small chance (self.epsilon)
             to take a random action instead of the optimal action from the
             stored actions' Q-values for the state passed in.

        Returns:
            best_action (string): A valid action to take from the current state.

        '''
        best_action = ""
        max_Q = float('-inf')
        # Shuffle actions randomly in case no Qvalue is above the
        #  default returned value when nothing is found in our Q-values default
        #  dict. This effectively makes it a random choice if we haven't
        #  encountered any of these state/action pairs yet.
        all_actions = self.env.valid_actions
        random.shuffle(all_actions)

        for action in all_actions:
            current_Q = self.getQValue(state, action)
            if current_Q > max_Q:
                max_Q = current_Q
                best_action = action
        if best_action == "":
            # Won't reach here currently since defaultdict returns a value
            #  which is greater than -inf
            best_action = random.choice(all_actions)

        # Add random choice of actions with probability epsilon
        if stochastic == True:
            r = random.random()
            if r < self.epsilon:
                best_action = random.choice(all_actions)
        return best_action



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    #with open('goals_reached.csv', 'w') as f:
    #    f.write('{0},{1}\n'.format('Trial', 'Goal_Reached'))
    #    [f.write('{0},{1}\n'.format(key, value)) for key, value in sim.goals_reached.items()]

if __name__ == '__main__':
    run()
