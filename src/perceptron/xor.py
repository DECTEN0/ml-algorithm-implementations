import numpy as np

# Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self):
        # Architecture: 2 → 2 → 1 (XOR)
        
        # Weights (random [-1,1])
        self.W1 = np.random.uniform(-1, 1, (2, 2))  # input → hidden
        self.b1 = np.random.uniform(-1, 1, (1, 2))
        
        self.W2 = np.random.uniform(-1, 1, (2, 1))  # hidden → output
        self.b2 = np.random.uniform(-1, 1, (1, 1))
        
        self.lr = 0.1

    def forward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2

    def compute_loss(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def backward(self, X, T):
        # Output layer error
        dL_da2 = (self.a2 - T)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)
        
        # Gradients for W2, b2
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        # Hidden layer error
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)
        
        # Gradients for W1, b1
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # Update weights (Gradient Descent)
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def train(self, X, T, epochs=10000):
        for epoch in range(epochs):
            y = self.forward(X)
            loss = self.compute_loss(y, T)
            
            self.backward(X, T)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
