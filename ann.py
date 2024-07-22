# AstroNet based on average activation of layer
import numpy as np
import matplotlib.pyplot as plt
from network import Network

class AstrocyteNetwork(Network):
    def __init__(self, sizes, astrocyte_density=1, initial_threshold=0.5, initial_effect=0.1):
        super().__init__(sizes)
        self.astrocyte_density = astrocyte_density
        self.astrocytes = self.initialize_astrocytes()
        
        # Initialize trainable astrocyte parameters
        self.astrocyte_thresholds = [np.full((size, 1), initial_threshold) for size in sizes[1:]]
        self.astrocyte_effects = [np.full((size, 1), initial_effect) for size in sizes[1:]]

    def initialize_astrocytes(self):
            return [np.random.rand(size, 1) < self.astrocyte_density for size in self.sizes[1:]]

    # TODO implement list of activations, thresholds
    # activation is determined by the average acrivation of current layer
    def astrocyte_activation(self, activation, threshold):
        # ic(np.mean(activation, axis=1, keepdims=True))
        # ic(threshold)
        # ic((np.mean(activation, axis=1, keepdims=True) > threshold).astype(float))
        return (np.mean(activation, axis=1, keepdims=True) > threshold).astype(float)

    def modify_weights(self, weights, astrocyte_active, effect):
        ic(astrocyte_active)
        ic(effect.shape)
        ic(weights.shape)
        return weights * (1 + astrocyte_active * effect)

    def forward_prop(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        activation = X
        # note threshold, effect are now lists
        for b, w, astrocytes, threshold, effect in zip(self.biases[:-1], self.weights[:-1], self.astrocytes,
                                                       self.astrocyte_thresholds[:-1], self.astrocyte_effects):
            z = np.dot(w, activation) + b
            activation = self.ReLU(z)
            
            astrocyte_active = self.astrocyte_activation(activation, threshold)
            w = self.modify_weights(w, astrocyte_active, effect)
            
        return self.softmax(np.dot(self.weights[-1], activation) + self.biases[-1])

    def backward_prop(self, X, Y):
        # ic(X.shape[1])
        m = X.shape[1]  # number of examples
        Z, A = [X], [X]  # start with input layer
        astrocyte_activations = []
        
        # Forward pass
        for b, w, astrocytes, threshold, effect in zip(self.biases, self.weights, self.astrocytes, 
                                                       self.astrocyte_thresholds, self.astrocyte_effects):
            Z.append(np.dot(w, A[-1]) + b)
            activation = self.ReLU(Z[-1]) if b is not self.biases[-1] else self.softmax(Z[-1])
            A.append(activation)
            
            astrocyte_active = self.astrocyte_activation(activation, threshold)
            astrocyte_activations.append(astrocyte_active)
            w = self.modify_weights(w, astrocyte_active, effect)

        # Backward pass
        dZ = [A[-1] - self.one_hot(Y)]
        dW = [(1 / m) * np.dot(dZ[0], A[-2].T)]
        dB = [(1 / m) * np.sum(dZ[0], axis=1, keepdims=True)]
        dThreshold = []
        dEffect = []
        
        for l in range(2, self.num_layers):
            dZ_current = np.dot(self.weights[-l+1].T, dZ[0]) * self.ReLU_deriv(Z[-l])
            dW_current = (1 / m) * np.dot(dZ_current, A[-l-1].T)
            dB_current = (1 / m) * np.sum(dZ_current, axis=1, keepdims=True)

            # Compute gradients for astrocyte parameters 
            dThreshold_current = -np.sum(dZ_current * self.astrocyte_effects[-l] * 
                                         (A[-l] == self.astrocyte_thresholds[-l]), axis=1, keepdims=True)
            dEffect_current = np.sum(dZ_current * A[-l] * astrocyte_activations[-l], axis=1, keepdims=True)

            dZ.insert(0, dZ_current)
            dW.insert(0, dW_current)
            dB.insert(0, dB_current)
            dThreshold.insert(0, dThreshold_current)
            dEffect.insert(0, dEffect_current)

        return dW, dB, dThreshold, dEffect

    def gradient_descent(self, X, Y, alpha, iterations, batch_size=32):
        for i in range(iterations):
            for j in range(0, m, batch_size):
                X_batch = X[:, j:j+batch_size]
                Y_batch = Y[j:j+batch_size]
            
                # Skip empty batches
                if X_batch.shape[1] == 0:
                    continue
                
                dW, dB, dThreshold, dEffect = self.backward_prop(X_batch, Y_batch)
                self.update_params(alpha, dW, dB, dThreshold, dEffect)

            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(X)
                print(self.get_accuracy(predictions, Y))

    def update_params(self, alpha, dW, dB, dThreshold, dEffect):
        super().update_params(alpha, dW, dB)
        # TODO figure out how to apply loss to theshold matrix (list of lists) instead of list
        for i in range(len(self.astrocyte_thresholds)-1):
            self.astrocyte_thresholds[i] -= alpha * dThreshold[i]
            self.astrocyte_effects[i] -= alpha * dEffect[i]

    def plot_astrocyte_params(self):
        fig, axs = plt.subplots(2, len(self.astrocytes), figsize=(15, 6))
        for i in range(len(self.astrocytes)):
            axs[0, i].hist(self.astrocyte_thresholds[i], bins=20)
            axs[0, i].set_title(f'Layer {i+1} Thresholds')
            axs[1, i].hist(self.astrocyte_effects[i], bins=20)
            axs[1, i].set_title(f'Layer {i+1} Effects')
        fig.suptitle('Astrocyte Parameters Distribution')
        plt.tight_layout()
        plt.show()