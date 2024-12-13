import numpy as np
from decimal import *
from math import inf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import gym


class ChaosNet:
    def __init__(self, num_neurons: int, skew_map_type: str, feature_extraction_method: str, num_labels: int, epsilon: float, b: float, time_series_length: int,
                a: float, c: float, q: float):
        """
        Initialize the ChaosNet model with the specified hyperparameters.
        num_neurons: Number of neurons in the model
        skew_map_type: Type of skew map to use (Sk-T or Sk-B)
        feature_extraction_method: Method to use for feature extraction (TT or TT-SS)
        num_labels: Number of labels to classify
        epsilon: Threshold value for feature extraction
        b: Threshold value for feature extraction
        time_series_length: Length of the time series
        a: Threshold value for feature extraction
        c: Threshold value for feature extraction
        q: Initial value for the time series
        ---
        self.num_classes: number of output classes, 2 * num_labels, for each label, there is a positive and negative rewards class.
            index 0: positive reward for label 0
            index 1: positive reward for label 1
            ...
            index num_labels: negative reward for label 0
            index num_labels + 1: negative reward for label 1
            ...
        """
        self.num_neurons = num_neurons
        self.skew_map_type = skew_map_type # Sk-T or Sk-B for tend and binary
        self.feature_extraction_method = feature_extraction_method # TT or TT-SS
        self.num_classes = num_labels * 2
        self.avg_internal_representation = np.zeros((self.num_classes, self.num_neurons, 1)) # Average internal representation for each class
        self.internal_rep_avg_count = [1, 1, 1, 1] # How many times each class has been averaged
        # Hyperparameters
        self.epsilon = epsilon
        self.b = b
        self.time_series_length = time_series_length
        self.a = a
        self.c = c
        self.q = q
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
        for i in range(M):
            for j in range(N):
                # Create the boolean array
                A = np.abs(X_train[i, j] - timeseries[:, 0]) < self.epsilon
                # Use np.argmax to find the index of the first True value
                idx = np.argmax(A)
                if A[idx]:  # Check if there is at least one True value
                    firingtime[i, j] = timeseries[idx, 1]
                else:
                    firingtime[i, j] = np.nan  # Or any other default value if no True value is found
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
        N = X_train.shape[1]

        probability = np.zeros((M, N))
        
        for i in range(M):
            for j in range(N):
                A = np.abs(X_train[i, j] - timeseries[:, 0]) < self.epsilon
                
                # Check if there are any True values in A
                if np.any(A):
                    # Find the first index where A is True
                    first_true_index = np.argmax(A)  # First True index
                    
                    # Extract firing times up to the index specified
                    valid_times = timeseries[:int(timeseries[first_true_index, 1]), 0] - self.b < 0
                
                    if valid_times.size == 0:
                        probability[i, j] = 0
                    else: 
                        # Count how many False values are in valid_times
                        probability[i, j] = np.sum(~valid_times) / float(valid_times.size)
                else:
                    # If no True values are found, set probability to 0 for that position
                    probability[i, j] = 0

        return probability

    
    def method(self, X_train, timeseries):
        if self.feature_extraction_method == 'TT':
            return self.firingtime_calculation(X_train, timeseries)
        if self.feature_extraction_method == 'TT-SS':
            return self.probability_calculation(X_train, timeseries)
        
    def class_avg_distance(self, DistMat, y_train, label):
        """
        Calculate the average distance for a specified class label.
        """
        samples = y_train.shape[0]
        P = y_train.tolist().count([label])
        Q = DistMat.shape[1]
        class_dist = np.zeros((P,Q))
        k = 0
        for i in range(0, samples):
            if (y_train[i,0] == label):
                class_dist[k,:]=DistMat[i,:]
                k = k+1
        return np.sum(class_dist, axis = 0)/class_dist.shape[0]
    
    def cosine_similarity(self, test_firingtime, y_test, avg_class_dist):
        """
        This function calculates cosine similarity between each sample in `test_firingtime` 
        and the average class distances in `avg_class_dist` for classification purposes. 
        The predicted class is determined as the one with the highest cosine similarity for 
        each test sample.
        """
        samples = y_test.shape[0]
        cosine_sim = np.zeros((samples, avg_class_dist.shape[0]))
        for i in range(0, samples):
            for j in range(0, avg_class_dist.shape[0]):
                cosine_sim[i,j] = cosine_similarity(test_firingtime[i,:].reshape(1,-1), avg_class_dist[j,:].reshape(1,-1))
        return cosine_sim
    
    def find_max_diff(self, cosine_similarities, num_labels):
        """
        This function finds the maximum difference between cosine similarities for each class label.
        """
        diffs = []
        label = 0
        for i in range(0, num_labels):
            diffs.append(cosine_similarities[i] * 0.7 - cosine_similarities[i + num_labels] * 0.3) # Weight the positive reward more than the negative reward
        
        max_diff = max(diffs)
        label = diffs.index(max_diff)
        return max_diff, label
    
    def greedy(self, test_firingtime):
        """
        This function calculates the cosine similarity between each sample in `test_firingtime`
        """
        label = 0
        cosine_similarities = []
        for i in range(0, len(self.avg_internal_representation)):
            cosine_sim = cosine_similarity(test_firingtime.reshape(1,-1), self.avg_internal_representation[i].reshape(1,-1))
            cosine_similarities.append(cosine_sim)
        # print(cosine_similarities)
        max_diff, label = self.find_max_diff(cosine_similarities, len(self.avg_internal_representation) // 2)
        return max_diff, label
    
    def train(self, X_train, label):
        """
        Used for training the model on data. Will update the average internal representation for each class label.
        """
        firingtime = self.method(X_train, self.timeseries)
        # Calculate the average internal representation for the specified class label
        old_avg_internal_representation = self.avg_internal_representation[label]
        self.avg_internal_representation[label] = (self.internal_rep_avg_count[label] * old_avg_internal_representation + firingtime) / (self.internal_rep_avg_count[label] + 1)
        self.internal_rep_avg_count[label] += 1
        
    def predict(self, state):
        """
        Predicts the best action to take given the current state.
        """
        firingtime = self.method(state, self.timeseries)
        # Find the class with the highest cosine similarity
        _, label = self.greedy(firingtime)
        return label

def normalize(x):
    x[0] = x[0] / 32
    x[1] = x[1] / 11
    x[2]= x[2] * 0.9
    return x
    
def test_blackjack(b, q):
    model = ChaosNet(num_neurons=3, skew_map_type='Sk-B', feature_extraction_method='TT-SS', num_labels=2, epsilon=0.01, b=b, time_series_length=10000, a=0, c=1, q=q)
    env = gym.make('Blackjack-v1')
    # random training
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, (3, 1))
            next_state = normalize(next_state)
            if not done: # reward if we havent busted
                model.train(next_state, action)
            state = next_state 
        if reward > 0: 
            model.train(state, action)
        elif reward < 0: 
            label = env.action_space.n + action
            model.train(state, label) 
        
    avg_internal_representation = model.get_internal_representation()
    num_episodes = 10000
    scores = []
    for i in range(num_episodes):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            state = np.reshape(state, (3, 1))
            state = normalize(state)
            action = model.predict(state)
            next_state, reward, done, _, _ = env.step(action)
            score += reward
            state = next_state
            next_state = np.reshape(next_state, (3, 1))
            next_state = normalize(next_state)
            if not done:
                model.train(next_state, action) 
        if reward < 0: 
            label = env.action_space.n + action
            model.train(next_state, label) 
        elif reward > 0: 
            model.train(next_state, action) 
        scores.append(reward)
        avg_score = np.mean(scores)
    return avg_score, scores, avg_internal_representation

def tune_hyperparameters():
    # Hyperparameter tuning setup
    # Adjust precision as needed
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
            best_avg_score, scores, _ = test_blackjack(b, q)
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

    
if __name__ == '__main__':
    # tune_hyperparameters()
    # 0.41, 0.21 for blackjack
    avg_score, scores, avg_internal_representation = test_blackjack(0.41, 0.21)
    print(avg_score)
