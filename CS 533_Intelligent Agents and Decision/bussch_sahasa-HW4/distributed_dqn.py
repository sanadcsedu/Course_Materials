from memory_remote import ReplayBuffer_remote
from dqn_model import DQNModel
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv
import ray

import os
import time
from random import randint, choice, uniform
FloatTensor = torch.FloatTensor
import matplotlib.pyplot as plt

def save_obj(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f)

import pickle

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)#, temp_dir = '~/tmp/ray/')

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing floder
result_folder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
torch.set_num_threads(12)

ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}

env_CartPole = CartPoleEnv()

@ray.remote
class DQLmodel_server(object):
    def __init__(self, env, memory_server, hyper_params, action_space = len(ACTION_DICT)):

        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.memory_server = memory_server
        self.results = []
        self.eval_id = 0

        self.training_episodes = 0
        self.test_interval = 0
        self.evals_ready = 0

        self.evaluator_done = False
        self.learning_done = False

        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon.
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps',
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True 
        self.action_space = action_space

        """
            input_len The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
            use_target_model: Trigger for turn 'target_model' on/off
        """
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)

        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

    def setTrainingEpisodes(self, training_episodes):
        self.training_episodes = training_episodes

    def setTestInterval(self, test_interval):
        self.test_interval = test_interval

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon,
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)

        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state)

    def greedy_policy(self, state):
        return self.eval_model.predict(state)
        
    def get_epsilon(self):
        return self.linear_decrease(self.initial_epsilon,
                           self.final_epsilon,
                           self.steps,
                           self.epsilon_decay_steps)

    def update_batch(self, collector_steps):
        self.steps = self.steps + collector_steps

        if self.steps < self.batch_size:
            return

        batch = ray.get(self.memory_server.sample.remote(self.batch_size))
        
        (states, actions, reward, next_states,
         is_terminal) = batch

        states = states
        next_states = next_states
        terminal = FloatTensor([0 if t else 1 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)

        # If is_terminal == 1, q_target = reward + discounted factor * Q_max, otherwise, q_target = reward
        q_max, max_action = torch.max(q_next, dim = 1)
        q_target = FloatTensor(reward + self.beta*q_max*terminal)

        # update model
        self.eval_model.fit(q_values, q_target)

        if self.steps % self.model_replace_freq == 0:
            self.target_model.replace(self.eval_model)

    def increment_episode(self):
        self.episode = self.episode + 1
        print(self.episode)
        if self.episode % self.test_interval == 0:
            self.evals_ready = self.evals_ready + 1

        if self.episode >= self.training_episodes:
            self.learning_done = True

        return self.learning_done

    def get_results(self):
        return [result[0] for result in sorted(self.results, key=lambda tup: tup[1])]
    
    def add_result(self, result, num):
        self.results.append((result,num))

    # Should pass self to worker so they can call the storeResults method
    def ask_evaluation(self):
        if 0 < self.evals_ready:
            self.evals_ready = self.evals_ready - 1
            return self.eval_model, self.evaluator_done, len(self.results)
        else:
            if self.episode >= self.training_episodes:
                self.evaluator_done = True
            return None, self.evaluator_done, None

@ray.remote
def collecting_worker(memory_server, model_server, env, epsilon, update_steps,
                max_episode_steps, action_space = len(ACTION_DICT)):
    def explore_or_exploit_policy(state):
        p = uniform(0, 1)
        if p < epsilon:
            #return action
            return randint(0, action_space - 1)
        else:
            #return action
            return ray.get(model_server.greedy_policy.remote(state))

    steps = 0
    learn_done = False
    while True:
        if learn_done:
            break
        done = False
        steps = steps % update_steps
        state = env.reset()
        while steps < max_episode_steps and not done:
                steps += 1
                # add experience from explore-exploit policy to memory_server
                action = explore_or_exploit_policy(state)
                state_prime, reward, done, _ = env.step(action)
                memory_server.add.remote(state, action, reward, state_prime, done)
                state = state_prime

                # update the model every 'update_steps' of experience
                if steps % update_steps == 0:
                    model_server.update_batch.remote(steps)
                    epsilon = ray.get(model_server.get_epsilon.remote())

        learn_done = ray.get(model_server.increment_episode.remote())

@ray.remote
def evaluation_worker(model_server, env, max_episode_steps, trials = 30):
    def greedy_policy(state):
        return model.predict(state)

    while True:
        model, done, num = ray.get(model_server.ask_evaluation.remote())
        if done:
            break
        if model == None:
            continue
        total_reward = 0
        for trial in range(trials):
            state = env.reset()
            done = False
            steps = 0
            
            while steps < max_episode_steps and not done:
                steps += 1
                action = greedy_policy(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print(avg_reward)
        model_server.add_result.remote(avg_reward, num)

class distributed_DQL_agent():
    def __init__(self, env, hyper_params):

        # Create servers
        self.memory_server = ReplayBuffer_remote.remote(hyper_params['memory_size'])
        self.model_server = DQLmodel_server.remote(env, self.memory_server, hyper_params)

        self.cw_num = hyper_params['cw_num']
        self.ew_num = hyper_params['ew_num']
        self.initial_epsilon = 1
        self.update_steps = hyper_params['update_steps']
        self.agent_name = "Distributed DQ-learning"
        self.env = env

    def learn_and_evaluate(self, training_episodes, test_interval):
        self.model_server.setTrainingEpisodes.remote(training_episodes)
        self.model_server.setTestInterval.remote(test_interval)

        workers_id = []
        
        max_episode_steps = self.env._max_episode_steps
        for i in range(self.cw_num):
            workers_id.append(collecting_worker.remote(self.memory_server, self.model_server, self.env,
                                                       self.initial_epsilon, self.update_steps,
                                                       max_episode_steps))

        for i in range(self.ew_num):
            workers_id.append(evaluation_worker.remote(self.model_server, self.env, max_episode_steps))

        #while len(workers_id) > 0:
        _, workers_id = ray.wait(workers_id, len(workers_id))
        
        return ray.get(self.model_server.get_results.remote())

hyperparams_CartPole = {
    'epsilon_decay_steps' : 70000,
    'final_epsilon' : 0.1,
    'batch_size' : 32,
    'update_steps' : 10,
    'memory_size' : 2000,
    'beta' : 0.99,
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True,
    'cw_num': 4,
    'ew_num': 4
}

def plot_result(total_rewards, learning_num):
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('Performance')
    #plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig('4_4_plot_result.png')

start_time = time.time()
#training_episodes, test_interval = 10000, 50 ### Total episodes/episodes before evaluate
training_episodes, test_interval = 7000, 50
#training_episodes, test_interval = 100, 10
agent = distributed_DQL_agent(env_CartPole, hyperparams_CartPole)
result = agent.learn_and_evaluate(training_episodes, test_interval)
print(result)

plot_result(result, test_interval)
run_time = time.time() - start_time
print("Learning time:")
print(run_time)

save_obj(run_time, 'runtime_4_4')
