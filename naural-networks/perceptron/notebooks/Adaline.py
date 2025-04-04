import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        self.weights = None

    def weighting(self, inputs):
        return np.dot(inputs, self.weights)

    def activation(self, weighted_input):
        return weighted_input

    def predict(self, inputs):
        bias = np.ones((inputs.shape[0], 1))
        new_inputs = np.hstack((bias, inputs))
        predictions = []
        for input_vector in new_inputs:
            weighted_input = np.dot(input_vector, self.weights) 
            prediction = 1 if self.activation(weighted_input) >= 0 else -1
            
            predictions.append(prediction)
        return np.array(predictions) 

    def fit(self, inputs, outputs, learning_rate=0.1, epochs=64):
        bias = np.ones((inputs.shape[0], 1)) 
        new_inputs = np.hstack((bias, inputs))
        self.weights = np.random.rand(new_inputs.shape[1])
        # self.errors = []

        for epoch in range(epochs):
            # total_error = 0
            weighted_inputs = self.weighting(new_inputs)
                    
            diff = self.activation(weighted_inputs) - outputs
            self.weights = self.weights - learning_rate * np.dot(new_inputs.T, diff)
            # total_error = np.mean(diff**2)
            # self.errors.append(total_error)

