import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import defaultdict
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, Q_val_starting_val, replay_memory_episodes):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_vals = defaultdict(lambda: Q_val_starting_val)
        self.state_visits = defaultdict(lambda: 0)
        self.state_action_visits = defaultdict(lambda: 0)
        self.alpha = 0.1 # Learning rate
        self.discount = 0.9 # Future discount factor
        self.total_reward = 0.0 # Cumulative reward
        self.diagnostics = {}
        self.n_trials = 0 # Counter for n_trials
        self.epsilon = 0.05 # Random action chance

        self.replay_memory = defaultdict()
        self.replay_memory_episodes = replay_memory_episodes


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        if t == 0:
            self.n_trials += 1
            self.diagnostics[self.n_trials] = '*FAILED*'


        # Update state, defined below
        state = self.getState()
        self.state = state

        # Select action according to policy
        # action = random.choice(self.env.valid_actions)
        action = self.computeArgmaxAction(state, stochastic=True)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # After self.env.act() acts on the environment, retrieve the next state
        next_state = self.getState()

        # For each state/action pair, record the deterministic predicted
        #  next_state/reward pair.
        self.replay_memory[(state, action)] = (next_state, reward)

        # Track how many times states were visited to weight unvisited states
        #  more heavily
        self.state_visits[state] += 1
        self.state_action_visits[(state, action)] += 1

        if not reward >= 10:
            # If the destination was reached, don't update the Q-values this is
            # a terminal state. Also, this means that the nexy_waypoint is
            #  invalid and 'None' for the next_state.
            # Q-values table would just have extra invalid 'terminal' states
            #  with default scores added (which can never be updated
            #  in this implementation since they are terminal states).

            # ~ Q-VALUE UPDATE ~
            # Update Q(s, a) with a discounted future max(Q(s') values)
            # Add an exploration bonus to unvisited Q-values, computeValueFromQValues(.., exploration_bonus=True)
            # Note: Unneeded in our formulation since we started with
            #  optimistic Q values to encourage exploration.
            new_Q_sa = self.update_Q_sa(self.state, action, reward, next_state)

            # Update our Q-value table
            self.Q_vals[(self.state, action)] = new_Q_sa


            # Do some Dyna-Q/experience replay learning!
            try:
                # Try/except works here b/c it will only fail briefly until at
                #  least 'self.planning_steps' many state actions pairs are visited

                # Sample K different random state/actions;
                memory_states = random.sample(self.replay_memory.items(), self.replay_memory_episodes)
                self.experience_replay_updates(memory_states)

            except ValueError:
                memory_states = self.replay_memory.items()
                self.experience_replay_updates(memory_states)


        self.total_reward += reward

        if reward < 0:
            print "*"*80
            print "NEGATIVE!"
            print reward
            print "*"*80

        if reward >= 10:
            # Print out current trial record each time destination is reached.
            # This is presumably when the target has been reached.
            self.diagnostics[self.n_trials] = 'yay'

        if self.n_trials == 100:
            # Print out statistics up to the last trial
            print self.Q_vals
            print "*"*80
            print self.diagnostics
            print "*"*80
            print 'Total Cumulative Rewards {}'.format(self.total_reward)
            #print self.state_visits
            print 'Total State/Action visits: ', sum(self.state_action_visits.values())
            print 'Total State Visits: ', sum(self.state_visits.values())

            print 't:', t, "!!!!"

    def experience_replay_updates(self, planning_states):
        """ Takes a collection of <s, a, s' r> tuples of the agent's experiences


        Args:
            planning_states (tuple):

        Returns:

        """

        for (random_state, random_action), (predicted_next_state, predicted_reward) in planning_states:
            new_Q_sa = self.update_Q_sa(random_state, random_action, predicted_reward, predicted_next_state)
            self.Q_vals[random_state, random_action] = new_Q_sa

            self.state_visits[random_state] += 1
            self.state_action_visits[(random_state, random_action)] += 1

    def update_Q_sa(self, state, action, reward, next_state):
        """ Updates Q(s,a) for <s, a, r, s'> the state, action, reward, and
             next state that are passed in.

        Args:
            state (tuple): Tuple of tuples representing a particular state s of
             the system.
            action (string): A valid action to take.
            reward (float): A reward for being in state s
            next_state (tuple): Tuple of tuples representing a particular state
             s' of the system.

        Returns:

        """
        # Get current Q(s,a) value
        current_Q_sa = self.getQValue(state, action)

        #new_Q_sa = (1 - self.alpha) * current_Q_sa + self.alpha * (reward + self.discount * self.computeMaxQ_value(next_state, exploration_bonus=False))
        # Algebraic manipulation: rewritten more compactly.
        new_Q_sa = current_Q_sa + self.alpha * (reward + self.discount * self.computeMaxQ_value(next_state) - current_Q_sa)
        return new_Q_sa

    def getState(self):
        ''' Returns tuple of tuples describing the current state of the system.

        Returns:
            state (tuple): Tuple of tuples describing the current state
                of the system.

        '''
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        state = (('light', inputs['light']),
                 ('next_waypoint', self.next_waypoint))

        return state

    def getQValue(self, state, action):
        """ Returns QValue of state/action pair that are passed in.

        Args:
            state (tuple): Tuple of tuples representing a particular state of
             the system.

          Returns:
               Q-node value (int): Default value if we have never seen a state
                or the Q node value otherwise
        """

        return self.Q_vals[(state, action)]

    def computeMaxQ_value(self, state, exploration_bonus=False):
        ''' Returns the max Q-value over all valid actions that can be taken
        from the state that is passed in.

        Args:
            state (tuple): Tuple of tuples representing a particular state of
             the system.
            exploration_bonus (Boolean): Whether to use an exploration bonus to
             encourage unexplored actions.

        Returns:
            max_Q (float): Max Q-value of valid actions from the state that
             is being evaluated.

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

                current_Q + 0.01/state_action_visits

            if current_Q > max_Q:
                max_Q = current_Q

        return max_Q

    def computeArgmaxAction(self, state, stochastic=False):
        ''' Returns the argmax Q-value action over all valid actions for the
        state that is passed in.

        Can also add a random action choice chance with the 'stochastic' flag.

        Args:
            state (tuple): Tuple of tuples representing a particular state of
             the system.
            stochastic (Boolean): Whether to add a small chance (self.epsilon)
             to take a random action instead of the optimal action from the
             stored actions' Q-values for the state passed in.

        Returns:
            best_action (string): A valid action to take from the state that
             is being evaluated.

        '''
        best_action = ""
        max_Q = float('-inf')
        # Shuffle actions randomly in case no Qvalue is above the
        #  default returned value when nothing is found in our Q-values default
        #  dict. This effectively makes it a random choice if we haven't
        #  encountered any of these state/action pairs yet.
        all_actions = self.env.valid_actions
        random.shuffle(all_actions)

        # Add random choice of actions with probability epsilon
        if stochastic == True:
            r = random.random()
            if r < self.epsilon:
                best_action = random.choice(all_actions)
                return best_action # No need to eval rest if randomly chosen

        for action in all_actions:
            current_Q = self.getQValue(state, action)
            if current_Q > max_Q:
                max_Q = current_Q
                best_action = action

        return best_action



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, Q_val_starting_val=0, replay_memory_episodes=20)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.002)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


    print e.primary_agent.total_reward

    print 'Goal reached {} times'.format(sum(sim.goals_reached.values()))
    #with open('goals_reached.csv', 'w') as f:
    #    f.write('{0},{1}\n'.format('Trial', 'Goal_Reached'))
    #    [f.write('{0},{1}\n'.format(key, value)) for key, value in sim.goals_reached.items()]

if __name__ == '__main__':
    run()
