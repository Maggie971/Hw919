import numpy as np

# Activation class
class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

# Neuron class
class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation_function = activation_function
        self.output = None  # 保存每个神经元的输出

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation_function(z)
        return self.output

# Layer class
class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, activation_function):
        self.neurons = [Neuron(num_inputs_per_neuron, activation_function) for _ in range(num_neurons)]
    
    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

# Loss function class
class LossFunction:
    def mse(self, predicted, actual):
        return np.mean(np.square(predicted - actual))
    
    def mse_derivative(self, predicted, actual):
        return 2 * (predicted - actual) / actual.size

# Forward propagation class
class ForwardProp:
    def __init__(self, model):
        self.model = model
    
    def forward(self, inputs):
        output = inputs
        for layer in self.model.layers:
            output = layer.forward(output)
        return output

# Backpropagation class
class BackProp:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function

    def backward(self, predicted, actual, inputs):
        deltas = []
        # 计算输出层的误差
        error = self.loss_function.mse_derivative(predicted, actual)
        deltas.append(error)

        # 反向传播隐藏层
        for i in reversed(range(len(self.model.layers) - 1)):
            layer = self.model.layers[i]
            next_layer = self.model.layers[i + 1]
            error = np.dot(deltas[-1], [neuron.weights for neuron in next_layer.neurons]) * Activation().relu_derivative(layer.forward(inputs))
            deltas.append(error)

        deltas.reverse()
        return deltas

# Gradient descent optimizer class
class GradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, layers, deltas, inputs):
        for i, layer in enumerate(layers):
            inputs_for_layer = inputs if i == 0 else [neuron.output for neuron in layers[i-1].neurons]  # 当前层的输入
            for j, neuron in enumerate(layer.neurons):
                # 更新每个神经元的权重
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= self.learning_rate * deltas[i][j] * inputs_for_layer[k]
                # 更新偏置
                neuron.bias -= self.learning_rate * deltas[i][j]

# Model class
class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            Layer(hidden_size, input_size, Activation().relu),
            Layer(output_size, hidden_size, Activation().sigmoid)
        ]
    
    def predict(self, inputs):
        return ForwardProp(self).forward(inputs)

# Training class
class Training:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(X.shape[0]):
                inputs = X[i]
                target = y[i]
                
                # Forward propagation
                predictions = ForwardProp(self.model).forward(inputs)
                
                # Compute loss
                loss = self.loss_function.mse(predictions, target)
                epoch_loss += loss
                
                # Backpropagation and update weights
                deltas = BackProp(self.model, self.loss_function).backward(predictions, target, inputs)
                self.optimizer.update(self.model.layers, deltas, inputs)
            
            print(f"Epoch {epoch}, Loss: {epoch_loss / X.shape[0]}")

# Main execution block
if __name__ == "__main__":
    model = Model(input_size=2, hidden_size=3, output_size=1)

    # Loss function and optimizer
    loss_function = LossFunction()
    optimizer = GradDescent(learning_rate=0.01)

    # Trainer
    trainer = Training(model, loss_function, optimizer)

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train
    trainer.train(X, y, epochs=1000)
