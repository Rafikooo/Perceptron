import random
from http.server import BaseHTTPRequestHandler
import json

import numpy as np

from perceptron import Perceptron
from training_data import data, labels

class Server(BaseHTTPRequestHandler):
    body = {"default": "Json response"}

    def __init__(self, perceptrons, *args, **kwargs):
        self.perceptrons = perceptrons
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        return

    def do_OPTIONS(self):
        self.respond({})

    def do_GET(self):
        self.launch_perceptrons()
        accuracy_on_learning_data = self.get_accuracy(data)
        augmented_data = self.get_augmented_data(data)
        accuracy_on_augmented_data = self.get_accuracy(augmented_data)
        self.respond({"accuracy_on_learning_data": accuracy_on_learning_data,
                      "accuracy_on_augmented_data": accuracy_on_augmented_data})

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        content_binary = self.rfile.read(content_length)  # <--- Gets the data itself

        content = content_binary.decode()
        input_data = json.loads(content)
        guesses = self.guess_digit(input_data['data'])
        body = {'data': guesses}
        self.respond(body)
        return

    def do_PUT(self):
        self.respond({"put": "works"})

    def handle_http(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Expose-Headers', '*')
        self.end_headers()

        return bytes(json.dumps(body), 'UTF-8')

    def respond(self, body):
        content = self.handle_http(200, 'application/json', body)
        self.wfile.write(content)

    def guess_digit(self, input_data):
        guesses = []
        for perceptron in self.perceptrons:
            guesses.append(perceptron.predict(input_data))

        return guesses

    def launch_perceptrons(self):
        perceptrons = []
        for i in range(10):
            perceptrons.append(Perceptron(i))
            perceptrons[i].fit(data, labels)
        self.perceptrons = perceptrons

    def get_accuracy(self, input_data):
        sum = 0
        accuracies = []
        for perceptron in self.perceptrons:
            for i, feature in enumerate(input_data):
                y_predicted, _ = perceptron.predict(feature)
                if y_predicted == 1 and labels[i] == perceptron.learning_digit or y_predicted == -1 and labels[i] != perceptron.learning_digit:
                    sum += 1
            accuracies.append(sum/len(data))
            sum = 0

        return accuracies

    def get_augmented_data(self, input_data):
        output = []
        replacement_count = 2
        for feature in input_data:
            indexes = random.sample(range(35), replacement_count)
            for index in indexes:
                feature[index] = 1 if feature[index] == -1 else -1
            output.append(np.copy(feature))

        return output
