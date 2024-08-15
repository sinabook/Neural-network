import numpy as np
class Model:
    def __init__(self):
        self.w1 = np.random.normal(loc=0, scale=0.01, size=(1000, 9))
        self.w2 = np.random.normal(loc=0, scale=0.01, size=(1, 1000))

    def predict(self, inputs):
        Z_1 =(self.w1)@inputs
        A_1 = np.maximum(0, Z_1)  
        Z_2 =(self.w2)@A_1
        A_2 = 1 / (1 + np.exp(-Z_2))
        return A_1, A_2

    def update_weights_for_one_epoch(self, inputs, outputs, learning_rate):
        x, y_true = inputs, outputs
        A_1, A_2 = self.predict(inputs)
        n =x.shape[0]
        relu_gradient = np.where(A_1 > 0, 1, 0)
        shared_coefficient = (-2*learning_rate/n) * (y_true - A_2) * A_2 * (1 - A_2)
        self.w1 = self.w1 - (((shared_coefficient.T).dot(self.w2)).T * relu_gradient).dot(x.T)
        self.w2 = self.w2 - (shared_coefficient.dot(A_1.T))

    def fit(self, inputs, outputs, learning_rate, epochs=64):
        for _ in range(epochs):
            self.update_weights_for_one_epoch(inputs, outputs, learning_rate)

