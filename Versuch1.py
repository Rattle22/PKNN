import KTimage as kt
import look

import numpy
import math
from multiprocessing import Pool

def learnXOR(rate, tries):
    input_count = 2
    inputs = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_count = 1
    outputs = numpy.array([0, 1, 1, 0])

    hidden_layer_neuron_count = 2
    initial_weight_range = [-0.1, 0.1]
    learning_rate = rate

    w_hid = numpy.random.uniform(*initial_weight_range, (hidden_layer_neuron_count, input_count))
    w_out = numpy.random.uniform(*initial_weight_range, (output_count, hidden_layer_neuron_count))
    b_hid = numpy.random.uniform(0.0, -1, hidden_layer_neuron_count)
    b_out = numpy.random.uniform(0.0, 0.0, output_count)

    lowestErr = 1

    for x in range(tries):
        for i, o in zip(inputs, outputs):
            #Sum weigthed inputs, then apply sigmoid function
            h_hid = numpy.dot(w_hid, i)+b_hid
            s_hid = 1/(1+numpy.exp(-h_hid))
            #Calculate derivative of sigmoid function
            ablsig = s_hid * (1-s_hid)
            #Sum weighted hidden to get the output
            h_out = numpy.dot(w_out, s_hid)+b_out
            s_out = h_out
            #Calculate Error
            err = o - s_out
            delta = ablsig * numpy.dot(numpy.transpose(w_out), err)
            #Backpropagate
            w_out += learning_rate * numpy.outer(err, s_hid)
            w_hid += learning_rate * numpy.outer(delta, i)
            b_out += learning_rate * err
            b_hid += learning_rate * delta
            if x % 1000:
                if abs(*err) < lowestErr:
                    lowestErr = abs(*err)
                    kt.exporttiles(w_hid, 2, 2)
                #print(err)

    return lowestErr

def out():
    print("Ding!")

def go():
    workers = Pool()

    results = [workers.apply_async(learnXOR, [0.01, 10000]) for y in range(10)]
    errors = [r.get() for r in results]
    avgErr = numpy.mean(errors)
    print("0.01 best Err: ", avgErr)

    results = [workers.apply_async(learnXOR, [0.1, 1000]) for y in range(10)]
    errors = [r.get() for r in results]
    avgErr = numpy.mean(errors)
    print("0.1 best Err: ", avgErr)
