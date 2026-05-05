"""
Perceptron implementation in Python
Author: [Your Name]
This code defines a Perceptron class that can be trained on a dataset to perform binary classification. The perceptron uses a simple linear model and updates its weights based on the error between predicted and target labels. The training process continues until convergence or until a specified number of epochs is reached.
Usage:
1. Create an instance of the Perceptron class with the desired input dimension and learning
   rate.
2. Call the train method with your training data (features and target labels).
Example:
p = Perceptron(input_dim=2, learning_rate=0.1)
p.train(X_train, y_train, max_epochs=100)

"""

import numpy as np

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.1): 
        """
        Initialize the perceptron with random weights and threshold.
        
        Parameters:
        input_dim (int): Number of features in the input data.
        learning_rate (float): Step size for weight updates.
        
        """
        self.m = input_dim                            # stores number of features
        self.eta = learning_rate                      # stores the learning rate
        
        # Initialize weights and threshold randomly in [-1, 1] to break symmetry
        self.w = np.random.uniform(low=-1, high=1, size=self.m) # weights initialized randomly in range [-1, 1] corresponding to number of features, m ie. w1​,w2​,...,wm​
        self.theta = np.random.uniform(low=-1, high=1) # threshold initialized randomly in range [-1, 1]
    
    def predict(self, x):
        """
        Compute perceptron output:
        y = 1 if w·x > θ else 0

        Parameters:
                    x (numpy array): Input features for a single data point.
        Returns:
                int: Predicted label (0 or 1) based on the linear combination of inputs and weights compared to the threshold.
        
        """
        h = np.dot(self.w, x)
        return 1 if h > self.theta else 0
    
    def compute_loss(self, X, T):
        """
        Compute the total loss for the dataset using the formula:
        L = sum |t - y|
        where t is the target label and y is the predicted label

        Parameters:
                    X (numpy array): Input features for the dataset.
                    T (numpy array): Target labels for the dataset.
        Returns:
                float: Total loss computed as the sum of absolute differences between target and predicted labels.

        """
        loss = 0
        for x, t in zip(X, T):                  # Iterate through each input and target pair
            y = self.predict(x)                 # Get the predicted label for the current input
            loss += abs(t - y)                  # Increment loss by the absolute difference between target and predicted label
        return loss
    

    
    def train(self, X, T, max_epochs=100):
        """
        Perceptron Training Algorithm
        1. For each training example (x, t):
            a. Compute the output y = predict(x)
            b. Update weights and threshold:
            w = w + η * (t - y) * x
            θ = θ - η * (t - y)
        2. Repeat for a specified number of epochs or until convergence (no updates or zero loss)

        Parameters:
                    X (numpy array): Input features for the training dataset.
                    T (numpy array): Target labels for the training dataset.
                    max_epochs (int): Maximum number of epochs to train the perceptron.
        Returns:
                None: The function updates the perceptron's weights and threshold in place.

        """
        for epoch in range(max_epochs):   # Loop over epochs

            # Shuffle data each epoch to ensure better convergence and avoid cycles
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            T_shuffled = T[indices]
            
            updates = 0    # Counter to track number of updates in this epoch
            
            for x, t in zip(X_shuffled, T_shuffled):              # Loop over each training example

                y = self.predict(x)                               # Compute the predicted label for the current input

                
                if y != t:                                        # If the prediction is incorrect, update weights and threshold

                    self.w = self.w + self.eta * (t - y) * x      # Update weights based on the error (t - y) and the input features x
                    self.theta = self.theta - self.eta * (t - y)  # Update threshold based on the error (t - y)
                    updates += 1
            
            # Compute loss
            loss = self.compute_loss(X, T)
            
            print(f"Epoch {epoch+1}: Loss = {loss}, Updates = {updates}")
            
            # Stopping condition: no updates OR perfect classification
            if updates == 0 or loss == 0: 
                print("Training converged.")
                break



if __name__ == "__main__":
    # Example usage
    # Dataset
    X = np.array([
        [3, 1],
        [2, 2.5],
        [2, 1.5],
        [4, 3],
        [3, 3]
    ])

    T = np.array([1, 0, 1, 1, 0])
    p = Perceptron(input_dim=2, learning_rate=0.1)
    p.train(X= X, T= T, max_epochs=50)

    print("================================================================================")
    print("\nFinal weights:", p.w)
    print("================================================================================")
    print("Final threshold:", p.theta)
    print("================================================================================")
    print("\nPredictions on training data:")
    test_points = np.array([
    [3, 1],
    [2, 2.5],
    [3, 2]   # unseen point
    ])

    for x in test_points:
        print(f"Input {x} -> Prediction: {p.predict(x)}")
print("================================================================================")    
