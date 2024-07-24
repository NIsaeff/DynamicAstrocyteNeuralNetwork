import numpy as np
import matplotlib.pyplot as plt
import time


class Network:
    def __init__(self, sizes, output_discrete:bool):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.output_discrete = output_discrete
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Xavier/He weights initialization
        self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def forward_prop(self, X):
        # Ensure X is 2D: (input_size, num_examples)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        activation = X
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            activation = self.ReLU(z)

        # Use sigmoid for the output layer when output is not discrete
        if not self.output_discrete:
            return self.sigmoid(np.dot(self.weights[-1], activation) + self.biases[-1])
        else:
            return self.softmax(np.dot(self.weights[-1], activation) + self.biases[-1])

    def backward_prop(self, X, Y):
        m = X.shape[1]  # number of examples
        # forward prop to save activations
        Z, A = [X], [X]  # start with input layer
        for b, w in zip(self.biases, self.weights):
            Z.append(np.dot(w, A[-1]) + b)
            if b is self.biases[-1]:
                A.append(self.sigmoid(Z[-1]) if not self.output_discrete else self.softmax(Z[-1]))
            else:
                A.append(self.ReLU(Z[-1]))
        
        # back prop
        # Compute the output layer error
        if not self.output_discrete:
            dZ = [A[-1] - Y]  # Binary cross-entropy error
        else:
            dZ = [A[-1] - self.one_hot(Y)]  # Softmax cross-entropy error
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
        if self.output_discrete:
            one_hot_Y = np.zeros((Y.size, self.sizes[-1]))
            one_hot_Y[np.arange(Y.size), Y] = 1
            one_hot_Y = one_hot_Y.T
            return one_hot_Y
        else:
            return Y
            
    def ReLU(self, Z):
        return np.maximum(Z, 0)
        
    def ReLU_deriv(self, Z):
        return Z > 0
    
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def get_predictions(self, X):
        if self.output_discrete:
            return np.argmax(self.forward_prop(X), 0)
        else:
            return (self.forward_prop(X) > 0.5).astype(int).flatten()
    
    def get_accuracy(self, predictions, Y):
        return np.mean(predictions == Y)
    
    def evaluate(self, X_test, Y_test):
        predictions = self.get_predictions(X_test)
        
        # Calculate accuracy
        accuracy = self.get_accuracy(predictions, Y_test)
        
        if self.output_discrete:
            # Multi-class metrics
            num_classes = self.sizes[-1]
            precisions = []
            recalls = []
            f1_scores = []
            
            for class_idx in range(num_classes):
                true_positives = np.sum((predictions == class_idx) & (Y_test == class_idx))
                false_positives = np.sum((predictions == class_idx) & (Y_test != class_idx))
                false_negatives = np.sum((predictions != class_idx) & (Y_test == class_idx))
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                
                # the harmonic mean of precision and recall
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
        
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1_score = np.mean(f1_scores)
        
        else:
            # Binary classification metrics
            true_positives = np.sum((predictions == 1) & (Y_test == 1))
            false_positives = np.sum((predictions == 1) & (Y_test == 0))
            false_negatives = np.sum((predictions == 0) & (Y_test == 1))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def train_with_visualization(self, X, Y, alpha, iterations, batch_size=32, eval_interval=10):
        accuracies = []
        iterations_list = []
        times = []
        start_time = time.time()

        m = m.shape[1]

        for i in range(iterations):
            for j in range(0, m, batch_size):
                X_batch = X[:, j:j+batch_size]
                Y_batch = Y[j:j+batch_size]

                # Skip empty batches
                if X_batch.shape[1] == 0:
                    continue

                dW, dB = self.backward_prop(X_batch, Y_batch)
                self.update_params(alpha, dW, dB)
            
            if i % eval_interval == 0 or i == iterations - 1:
                predictions = self.get_predictions(X)
                accuracy = self.get_accuracy(predictions, Y)
                current_time = time.time() - start_time
                
                accuracies.append(accuracy)
                iterations_list.append(i)
                times.append(current_time)
                
                print(f"Iteration: {i}, Accuracy: {accuracy:.4f}, Time: {current_time:.2f}s")

        self.plot_learning_curves(accuracies, iterations_list, times)

    def plot_learning_curves(self, accuracies, iterations, times):
        # Accuracy over time
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(times, accuracies)
        plt.title('Accuracy vs. Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Accuracy')
        plt.grid(True)

        # Accuracy per iteration
        plt.subplot(1, 2, 2)
        plt.plot(iterations, accuracies)
        plt.title('Accuracy vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    