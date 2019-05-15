# Neural-Network-From-Scratch
This is a neural network module that can learn from training data using nesterov momentum as optimization. It has a lot of functionalities; one can for example define the width and hight of the network, add dropout to layers, and customize the activation function in each layer.

### Usage
Using the module's basic functionality, solving the XOR problem is as simple as this:
```
from neural_network import NeuralNetwork

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

nn = NeuralNetwork([2, 10, 10, 1])
nn.train(X, Y, 1000)

pred = nn.predict(X)
print(pred)
```
We import the module and define the input data and its corresponding labels:
```
from neural_network import NeuralNetwork

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
```
We build the layout of the network:

In this example the network has 4 layers. The numbers represent the number of neurons in each layer.
```
nn = NeuralNetwork([2, 10, 10, 1])
```
Then we train the network on the data 1000 times with a learning rate of 0.01:
```
nn.train(X, Y, 1000, 0.01)
```
After we've trained our model, we can use it to predict our input data and see how well it performs:
```
pred = nn.predict(X)
print(pred)
```
When it's done we print out its predictions:
```
[[ 0.02524398]
 [ 0.94654331]
 [ 0.93204751]
 [ 0.08222075]]
```
Quite close to the solution, isn't it?
