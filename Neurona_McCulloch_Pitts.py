import numpy as np

# Implementar MPNeuron
class MPNeuron:
    def __init__(self):
        self.threshold = None

    def model(self, x):
        # Input: [1, 0, 1, 0] [x1, x2, .., xn]
        z = sum(x) # Función agregación
        return (z >= self.threshold) # función activación

    def predict(self, X):
        # Input: [[1, 0, 1, 0], [1, 0, 1, 1]]
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)


# Instanciar la neurona
mp_neuron = MPNeuron()

# Establecer un threshold
mp_neuron.threshold = 2

# Evaluar diferentes casos de uso
print(mp_neuron.predict([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]]))