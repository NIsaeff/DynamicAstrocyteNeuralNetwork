import numpy as np
import matplotlib as plt

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Xavier/He weights initialization
        self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(sizes[:-1], sizes[1:])]
            
   
    # def forward_prop(self, X):
    #     for b, w in zip(self.biases[:-1], self.weights[:-1]):
    #         X = self.ReLU(np.dot(w, X) + b)
    #     return self.softmax(np.dot(self.weights[-1], X) + self.biases[-1])
    
    def forward_prop(self, X):
        # Ensure X is 2D: (input_size, num_examples)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        activation = X
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            activation = self.ReLU(z)
        return self.softmax(np.dot(self.weights[-1], activation) + self.biases[-1])

    def backward_prop(self, X, Y):
        m = X.shape[1]  # number of examples
        # forward prop to save activations
        Z, A = [X], [X]  # start with input layer
        for b, w in zip(self.biases, self.weights):
            Z.append(np.dot(w, A[-1]) + b)
            A.append(self.ReLU(Z[-1]) if b is not self.biases[-1] else self.softmax(Z[-1]))
        
        # back prop
        dZ = [A[-1] - self.one_hot(Y)]
        dW = [(1 / m) * np.dot(dZ[0], A[-2].T)]
        dB = [(1 / m) * np.sum(dZ[0], axis=1, keepdims=True)]
        
        for l in range(2, self.num_layers):
            dZ.insert(0, np.dot(self.weights[-l+1].T, dZ[0]) * self.ReLU_deriv(Z[-l]))
            dW.insert(0, (1 / m) * np.dot(dZ[0], A[-l-1].T))
            dB.insert(0, (1 / m) * np.sum(dZ[0], axis=1, keepdims=True))
    
        return dW, dB
    
    def update_params(self, alpha, dW, dB):
        for i, (w, dw) in enumerate(zip(self.weights, dW)):
            self.weights[i] = w - alpha * dw
        for i, (b, db) in enumerate(zip(self.biases, dB)):
            self.biases[i] = b - alpha * db

    def gradient_descent(self, X, Y, alpha, iterations):
        for i in range(iterations):
            dW, dB = self.backward_prop(X, Y)
            
            self.update_params(alpha, dW, dB)

            if i % 10 == 0:
                print("Iteration: ", i)
                # ic(X.shape)
                predictions = self.get_predictions(X)
                print(self.get_accuracy(predictions, Y))
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, self.sizes[-1]))
        # one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        # ic(one_hot_Y.shape, one_hot_Y)
        # ic(Y)
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    # def make_predictions(self, X):
    #     A = self.forward_prop(X)
    #     predictions = self.get_predictions(A)
    #     return predictions

    # def test_prediction(self, X, Y, index):
    #     current_image = X[:, index]
    #     prediction = self.make_predictions(current_image)
    #     label = Y[index]
    #     print("Prediction: ", prediction)
    #     print("Label: ", label)
    
    #     current_image = current_image.reshape((28, 28)) * 255
    #     plt.gray()
    #     plt.imshow(current_image, interpolation='nearest')
    #     plt.show()
            
    def ReLU(self, Z):
        return np.maximum(Z, 0)
        
    def ReLU_deriv(self, Z):
        return Z > 0
    
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def get_predictions(self, X):
        return np.argmax(self.forward_prop(X), 0)
    
    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return(np.sum(predictions == Y) / Y.size)
    

# astrocyte remembers past n inputs
# if sum of n is greater than theshhold, 
# mult current output of activation function by mod factor
# then replace output w modified output

# for random evolution training, randomize theshold and adjacent mod factors simutaneously
# check if improvement is made, if improvement is made, keep new values, else revert