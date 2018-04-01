import gym
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt

# Adapted from https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/README.md

class Q_agent:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {} # Q-learning table, indexed with tuples
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    # Gets the expected reward from a given action in a given state
    def get_Q(self, state, action):
        return self.q.get((state, action), 0.0)

    def learn(self, prev_state, prev_action, reward, state):
        # Get the best possible reward expected reward for this time step
        max_q = max([self.get_Q(state, action) for action in self.actions])


        q_prev = self.q.get((prev_state, prev_action), None)
        if q_prev is None:
            self.q[(prev_state, prev_action)] = reward
        else:
            self.q[(prev_state, prev_action)] = q_prev + self.alpha * ((reward + self.gamma * max_q) - q_prev)


    def select_action(self, state):
        q = np.array([self.get_Q(state, action) for action in self.actions], dtype=np.float64)
        max_q = np.max(q)

        if random.random() < self.epsilon: # Explore!
            min_q = np.min(q)
            noise_magnitude = max(abs(min_q), abs(max_q))
            q += np.random.uniform(noise_magnitude / 2.0, -noise_magnitude / 2.0, q.shape)
            max_q = np.max(q)

        action = 0
        if np.sum(q == max_q) > 1:
            action = random.choice([action for action in xrange(len(self.actions)) if q[action] == max_q])
        else:
            action = q.argmax()

        return self.actions[action]
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    ### Stolen ###
    max_number_of_steps = 201
    n_bins = 8
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)
    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    def make_state(observation):
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)])
        return state

    qlearn = Q_agent(actions=xrange(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)

   ### End stolen ###

    scores = []

    for i_episode in range(3000):
        observation = env.reset()
        state = make_state(observation)

        for t in range(max_number_of_steps):
            action = qlearn.select_action(state)
            observation, reward, done, info = env.step(action)
            next_state = make_state(observation)
            if not(done):
                qlearn.learn(state, action, reward, next_state)
                state = next_state
            else: # learn that dying is bad!
                reward = -200
                qlearn.learn(state, action, reward, next_state)
                print "iteration %i score: %i"%(i_episode, t)
                scores.append(t)
                break



            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

plt.plot(scores)
plt.show()