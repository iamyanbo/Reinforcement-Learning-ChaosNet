from decimal import *
from math import inf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import gymnasium as gym
import numpy as np
import time


class ChaosNet:
    def __init__(self, epsilon, b, time_series_length, q, skew_map_type):
        """
        avg_internal_representation:
            index 0: action 0, reward 0 - 5
            index 1: action 1, reward 0 - 5
            index 2: action 2, reward 0 - 5
            index 3: action 3, reward 0 - 5
            index 4: action 0, reward 5 and above
            index 5: action 1, reward 5 and above
            index 6: action 2, reward 5 and above
            index 7: action 3, reward 5 and above
            index 8: action 0, reward -5 - 0
            index 9: action 1, reward -5 - 0
            index 10: action 2, reward -5 - 0
            index 11: action 3, reward -5 - 0
            index 12: action 0, reward -5 and below
            index 13: action 1, reward -5 and below
            index 14: action 2, reward -5 and below
            index 15: action 3, reward -5 and below
        """
        self.num_neurons = 8
        self.feature_extraction_method = "TT-SS"
        self.num_classes = 4 * 4 # 4 actions, 4 possible rewards
        self.avg_internal_representation = np.zeros((self.num_classes, self.num_neurons))
        self.internal_rep_avg_count = np.zeros(self.num_classes)
        self.epsilon = epsilon
        self.b = b
        self.time_series_length = time_series_length
        self.q = q
        self.a = 0
        self.c = 1
        self.skew_map_type = skew_map_type
        self.timeseries = self.iterations()
        
    def get_internal_representation(self):
        return self.avg_internal_representation
    
    def skew_tent(self, x):
        """
        Returns the value after applying the skew tent/skew binary map
        """
        if self.skew_map_type == 'Sk-T':
            if x < self.b:
                xn = ((self.c - self.a)*(x-self.a))/(self.b - self.a)
            else:
                xn = ((-(self.c-self.a)*(x-self.b))/(self.c - self.b)) + (self.c - self.a)
            return xn
        if self.skew_map_type == "Sk-B":
            if x < self.b:
                xn = x / self.b
            else:
                xn = (x - self.b)/(1 - self.b)
            return xn
        
    def iterations(self):
        """
        Returns the sequence of values after applying the skew tent/skew binary map
        along with its index values
        """
        timeseries = (np.zeros((self.time_series_length, 2)))
        timeseries[0, 0] = self.q
        for i in range(1, self.time_series_length):
            timeseries[i, 0] = self.skew_tent(timeseries[i-1, 0])
            timeseries[i, 1] = i
        return timeseries
    
    def firingtime_calculation(self, X_train, timeseries):
        """
        Returns the firing time for each neuron, finds index of the first value in the timeseries
        such that the absolute difference between the value and the neuron is less than epsilon
        """
        M = X_train.shape[0]
        N = X_train.shape[1]

        firingtime = np.zeros((M,N))
        for i in range(0,M):
            for j in range(0,N):
                A = (np.abs((X_train[i,j]) - timeseries[:,0]) < self.epsilon)
                firingtime[i,j] = timeseries[A.tolist().index(True),1]
        return firingtime
        
    def probability_calculation(self, X_train, timeseries):
        """
        This function computes probabilities for each element in `X_train` based on 
        a time-based threshold method. For each element in `X_train`, it identifies 
        where the absolute difference with the first column of `timeseries` is less 
        than `epsilon` and calculates the probability as the frequency of values 
        greater than `b` in the corresponding firing times.
        """
        M = X_train.shape[0]

        probability = np.zeros(M)
        
        for i in range(M):
            A = np.abs(X_train[i] - timeseries[:, 0]) < self.epsilon
            
            # Check if there are any `True` values in `A`
            if np.any(A):
                # Get the first index where A is True
                first_true_index = A.tolist().index(True)
                # Extract firing times up to the index specified
                freq = (timeseries[:int(timeseries[first_true_index, 1]), 0] - self.b < 0)
                
                if len(freq) == 0:
                    probability[i] = 0
                else: 
                    probability[i] = freq.tolist().count(False) / float(len(freq))
            else:
                # If no `True` values are found, set probability to 0 for that position
                probability[i] = 0

        return probability
    
    def method(self, X_train, timeseries):
        if self.feature_extraction_method == 'TT':
            return self.firingtime_calculation(X_train, timeseries)
        if self.feature_extraction_method == 'TT-SS':
            return self.probability_calculation(X_train, timeseries)
        
    def train(self, X_train, label):
        firingtime = self.method(X_train, self.timeseries)
        # Update average internal representation
        self.avg_internal_representation[label] = (self.avg_internal_representation[label] * self.internal_rep_avg_count[label] + firingtime) / (self.internal_rep_avg_count[label] + 1)
        self.internal_rep_avg_count[label] += 1
        
    def greedy(self, firingtime):
        label = 0
        cosine_similarities = []
        for i in range(self.num_classes):
            cosine_similarities.append(cosine_similarity(firingtime.reshape(1,-1), self.avg_internal_representation[i].reshape(1,-1)))
        diff = []
        for i in range(4):
            # find 'best' action, positive reward - negative reward
            diff.append(0.7 * (cosine_similarities[i] + cosine_similarities[i + 4]) - 0.3 *(cosine_similarities[i + 8] + cosine_similarities[i + 12]))
            
        label = np.argmax(diff)
        return label
    
    def predict(self, X_test):
        firingtime = self.method(X_test, self.timeseries)
        # Find the cosine similarity between the firing time of the test data and the average internal representation
        label = self.greedy(firingtime)
        return label
        
def normalize(x):
    x[0] = (x[0] + 2.5) / 5
    x[1] = (x[1] + 2.5) / 5
    x[2] = (x[2] + 10) / 20
    x[3] = (x[3] + 10) / 20
    x[4] = (x[4] + 6.2831855) / (6.2831855 * 2)
    x[5] = (x[5] + 10) / 20
    return x

def test_train(b, q):
    start_time = time.time()
    model = ChaosNet(epsilon=0.01, b=b, time_series_length=100000, q=q, skew_map_type='Sk-B')
    env = gym.make("LunarLander-v3", continuous=False)
    for i in range(10):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            obs = normalize(obs)
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            score += reward
            if reward > 0 and reward < 5:
                label = action
            elif reward >= 5:
                label = action + 4
            elif reward >= -5 and reward < 0:
                label = action + 8
            else:
                label = action + 12
            model.train(obs, label)
            obs = next_obs
    internal_representations = model.get_internal_representation()
    # print(internal_representations)
    num_episodes = 990
    scores = []
    env.close()
    env = gym.make("LunarLander-v3", continuous=False)
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        moves = 0
        while not done:
            obs = normalize(obs)
            action = model.predict(obs)
            next_obs, reward, done, _, _ = env.step(action)
            moves += 1
            score += reward
            if reward > 0 and reward < 5:
                label = action
            elif reward >= 5:
                label = action + 4
            elif reward >= -5 and reward < 0:
                label = action + 8
            else:
                label = action + 12
            model.train(obs, label)
            obs = next_obs
        scores.append(score)
    avg_score = np.mean(scores)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time taken: {elapsed_time}')
    return avg_score, scores, internal_representations

def tune_hyperparameters():
    # Hyperparameter tuning setup
    b_start = 0.01
    q_start = 0.01
    b_increment = 0.1
    q_increment = 0.1
    num_b_iterations = int(1 / b_increment)
    num_q_iterations = int(1 / q_increment)

    # To store the results
    b_values = []
    q_values = []
    avg_scores = []

    # Track the best score and parameters
    best_score = float('-inf')
    best_b = None
    best_q = None

    # Run through the values of b and q
    for i in range(num_b_iterations):
        for j in range(num_q_iterations):
            b = b_start + i * b_increment
            q = q_start + j * q_increment
            print(f'Iteration: b = {b}, q = {q}')
            best_avg_score, scores, _ = test_train(b, q)
            print(f'Best average Score: {best_avg_score}')
            b_values.append(b)
            q_values.append(q)
            avg_scores.append(best_avg_score)

            # Check if this is the best score so far
            if best_avg_score > best_score:
                best_score = best_avg_score
                best_b = b
                best_q = q

    # Convert to arrays for plotting
    b_values = np.array(b_values)
    q_values = np.array(q_values)
    avg_scores = np.array(avg_scores)

    # Print the best parameters and score
    print(f'\nBest Score: {best_score} with b = {best_b}, q = {best_q}')

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(b_values, q_values, avg_scores, cmap='viridis')
    ax.set_xlabel('b')
    ax.set_ylabel('q')
    ax.set_zlabel('Average Score')
    plt.title('Average Score as b and q Vary')
    plt.show()
    

if __name__ == "__main__":
    # tune_hyperparameters()
    # 0.01, 0.21 with custom norm
    avg_score, scores, internal_representations = test_train(0.01, 0.21)
    print(f'Average Score: {avg_score}')
    print(f'Scores: {scores}')

