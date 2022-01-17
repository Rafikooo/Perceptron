from functools import partial

from perceptron import Perceptron
from training_data import *
import time
from http.server import HTTPServer
from server import Server

HOST_NAME = 'localhost'
PORT_NUMBER = 7000

perceptrons = []
for i in range(10):
    perceptrons.append(Perceptron(i))
    perceptrons[i].fit(data, labels)

if __name__ == '__main__':
    handler = partial(Server, perceptrons)
    httpd = HTTPServer((HOST_NAME, PORT_NUMBER), handler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
