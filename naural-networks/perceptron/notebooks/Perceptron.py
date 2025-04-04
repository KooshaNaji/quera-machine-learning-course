import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(input, self.weights)

    def activation(self, weighted_input):
        return 1 if weighted_input >= 0 else -1  

    def predict(self, inputs):
        bias = np.ones((inputs.shape[0], 1))
        new_inputs = np.hstack((bias, inputs)) 
        predictions = []
        for input_vector in new_inputs:
            weighted_input = self.weighting(input_vector)
            prediction = self.activation(weighted_input)
            predictions.append(prediction)
        return np.array(predictions)

    def fit(self, inputs, outputs, learning_rate=0.001, epochs=64):
        bias = np.ones((inputs.shape[0], 1))
        new_inputs = np.hstack((bias, inputs)) 
        self.weights = np.random.rand(new_inputs.shape[1])
        for epoch in range(epochs):
            for sample, target in zip(new_inputs, outputs):
                weighted_input = self.weighting(sample) 
                diff = self.activation(weighted_input) - target 
                self.weights = self.weights - (learning_rate * diff * sample)

