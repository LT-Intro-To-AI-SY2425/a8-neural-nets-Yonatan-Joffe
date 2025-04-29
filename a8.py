from neural import *



print("\n\nTraining XOR\n\n")
xor_training_data = [([
    [0.9, 0.6, 0.8, 0.3, 0.1],
    [0.8, 0.8, 0.4, 0.4, 0.4],
    [0.7, 0.2, 0.4, 0.2, 0.3],
    [0.5, 0.5, 0.6, 0.8, 0.8],
    [0.3, 0.1, 0.6, 0.8, 0.8],
    [0.6, 0.3, 0.4, 0.3, 0.6]
])]

xorn = NeuralNet(1, 1, 1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))


