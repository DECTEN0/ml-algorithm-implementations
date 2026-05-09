import numpy as np

# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def sigmoid(z):
    """
    Sigmoid activation function.

    Formula:
        σ(z) = 1 / (1 + e^(-z))

    Compresses values between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """
    Derivative of sigmoid.

    If:
        a = sigmoid(z)

    Then:
        σ'(z) = a(1 - a)

    IMPORTANT:
    We pass the ACTIVATION VALUE (a),
    not the original z.
    """
    return a * (1 - a)


# ============================================================
# INPUT DATA
# ============================================================

# Input vector
x = np.array([[1, 0]])   # shape = (1, 2)

# Target output
t = np.array([[1]])      # shape = (1, 1)


# ============================================================
# INITIAL WEIGHTS AND BIASES
# ============================================================

# Input -> Hidden weights
W1 = np.array([
    [0.5, -0.5],
    [0.3,  0.8]
])

# Hidden layer bias
b1 = np.array([[0.1, -0.1]])

# Hidden -> Output weights
W2 = np.array([
    [0.7],
    [-1.2]
])

# Output bias
b2 = np.array([[0.05]])

# Learning rate
eta = 0.1


# ============================================================
# FORWARD PROPAGATION
# ============================================================

print("\n================ FORWARD PASS ================\n")

# ------------------------------------------------------------
# HIDDEN LAYER
# ------------------------------------------------------------

# Weighted sum into hidden layer
z1 = np.dot(x, W1) + b1

print("z1 (hidden weighted sums):")
print(z1)

# Hidden layer activations
a1 = sigmoid(z1)

print("\na1 (hidden activations):")
print(a1)


# ------------------------------------------------------------
# OUTPUT LAYER
# ------------------------------------------------------------

# Weighted sum into output neuron
z2 = np.dot(a1, W2) + b2

print("\nz2 (output weighted sum):")
print(z2)

# Final prediction
y = sigmoid(z2)

print("\ny (network prediction):")
print(y)


# ============================================================
# LOSS CALCULATION
# ============================================================

print("\n================ LOSS ================\n")

# Squared error loss
loss = 0.5 * np.sum((y - t) ** 2)

print("Loss:")
print(loss)


# ============================================================
# BACKPROPAGATION
# ============================================================

print("\n================ BACKPROPAGATION ================\n")

# ------------------------------------------------------------
# OUTPUT LAYER ERROR
# ------------------------------------------------------------

# Error at output
# dL/dy
output_error = y - t

print("Output error (y - t):")
print(output_error)

# Sigmoid derivative at output
output_delta = output_error * sigmoid_derivative(y)

print("\nOutput delta:")
print(output_delta)


# ------------------------------------------------------------
# GRADIENTS FOR W2 AND b2
# ------------------------------------------------------------

# dL/dW2
dW2 = np.dot(a1.T, output_delta)

print("\ndW2:")
print(dW2)

# dL/db2
db2 = output_delta

print("\ndb2:")
print(db2)


# ------------------------------------------------------------
# HIDDEN LAYER ERROR
# ------------------------------------------------------------

# Propagate error backwards
hidden_error = np.dot(output_delta, W2.T)

print("\nHidden error:")
print(hidden_error)

# Hidden delta
hidden_delta = hidden_error * sigmoid_derivative(a1)

print("\nHidden delta:")
print(hidden_delta)


# ------------------------------------------------------------
# GRADIENTS FOR W1 AND b1
# ------------------------------------------------------------

# dL/dW1
dW1 = np.dot(x.T, hidden_delta)

print("\ndW1:")
print(dW1)

# dL/db1
db1 = hidden_delta

print("\ndb1:")
print(db1)


# ============================================================
# GRADIENT DESCENT UPDATES
# ============================================================

print("\n================ WEIGHT UPDATES ================\n")

# Update W2
W2 = W2 - eta * dW2

# Update b2
b2 = b2 - eta * db2

# Update W1
W1 = W1 - eta * dW1

# Update b1
b1 = b1 - eta * db1


# ============================================================
# UPDATED PARAMETERS
# ============================================================

print("Updated W2:")
print(W2)

print("\nUpdated b2:")
print(b2)

print("\nUpdated W1:")
print(W1)

print("\nUpdated b1:")
print(b1)


# ============================================================
# SECOND FORWARD PASS (AFTER LEARNING)
# ============================================================

print("\n================ AFTER UPDATE ================\n")

# Forward pass again to see improvement

z1_new = np.dot(x, W1) + b1
a1_new = sigmoid(z1_new)

z2_new = np.dot(a1_new, W2) + b2
y_new = sigmoid(z2_new)

new_loss = 0.5 * np.sum((y_new - t) ** 2)

print("New prediction:")
print(y_new)

print("\nNew loss:")
print(new_loss)
