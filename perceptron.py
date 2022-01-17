import numpy as np
import numpy.random


class Perceptron:
    weights = []
    bias = None

    def __init__(self, learning_digit, learning_rate=0.01, steps=10000, input_length=35):
        self.learning_rate = learning_rate
        self.steps = steps
        self.input_length = input_length
        self.draw_initial_weights()
        self.pocket = None
        self.lifespan = 0
        self.learning_digit = learning_digit

    def draw_initial_weights(self):
        self.weights = [np.random.rand() * 2 - 1 for i in range(self.input_length)]
        self.bias = np.random.rand() * 2 - 1

    def fit(self, training_data, labels):
        lifespan = 0
        for i in range(self.steps):
            index = np.random.randint(0, len(training_data))
            feature = training_data[index]
            # linear_output = np.dot(feature, self.weights) + self.bias
            O, probability = self.predict(feature)
            label = 1 if self.learning_digit == labels[index] else -1
            error = self.__calc_error(O, label)
            if error == 0:
                lifespan += 1
                if lifespan > self.lifespan:
                    self.pocket = np.copy(self.weights) #deep copy
                    #bias
            else:
                # self.weights += np.dot(self.weights, feature)
                for j, weight in enumerate(self.weights):
                    self.weights[j] = weight + self.learning_rate * error * feature[j]
                self.bias = self.bias - self.learning_rate * error
                self.lifespan = 0
            # self.__update_weights(feature, error)

        # bias from pocket
        self.weights = self.pocket

    def predict(self, feature):
        linear_output = np.dot(feature, self.weights) + self.bias
        y_predicted = self.__activation_function(linear_output)
        probability = (linear_output + 1) / 2

        return y_predicted, probability

    def predict_with_debug_info(self, feature):
        y_predicted, probability = self.predict(feature)
        print("I\'m ", self.learning_digit, " perceptron")
        print("This is definitely ", self.learning_digit)if y_predicted == 1 else print("This is not ", self.learning_digit)
        print("\n")

    def print_learning_result(self):
        pass

    def __update_weights(self, feature, error):
        update = self.learning_rate * error

        self.weights += update * feature
        self.bias += update

    def __activation_function(self, linear_output):
        return 1 if linear_output > 0 else -1

    def __calc_error(self, linear_output, label):
        return label - linear_output

    def debug(self):
        print(self.weights)

