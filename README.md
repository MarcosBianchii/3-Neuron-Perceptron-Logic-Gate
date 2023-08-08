# Multi-Layer Perceptron Logic Gate
The model uses 3 neurons each activated by a sigmoid function to learn any logical gate.

## Compile and Run
```
$ ./build.sh
```
## Use
The truth table in `model.c` can be changed to any logic gate. The model will learn the behavior of the gate and print the results to the console.

## XOR Problem
The model has the ability to learn the behavior of an XOR gate. The XOR gate is defined as:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

It is not possible to learn the XOR gate using a single neuron. However, it is possible to learn it using at least 3.
