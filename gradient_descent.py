import numpy as np

def hypothesis(weights, X):
    return np.dot(weights, X)

def meanSquaredError(X, y, weights):
    cost = 0
    for Xrow, yValue in zip(X, y):
        cost += 0.5 * (hypothesis(weights, Xrow) - yValue) ** 2
    return cost / len(X)

def meanSquaredErrorDerived(X, y, weights, derivationIndex):
    cost = 0
    for Xrow, yValue in zip(X, y):
        cost += (hypothesis(weights, Xrow) - yValue) * Xrow[derivationIndex]
    return cost / len(X)

def updateWeights(X, y, weights, learningRate):
    updated_weights = np.copy(weights)
    for i in range(len(weights)):
        updated_weights[i] -= learningRate * meanSquaredErrorDerived(X, y, weights, i)
    return updated_weights

def trainModel(X, y, learningRate, iterations):
    weights = np.random.rand(len(X[0])) # random weights
    for itr in range(iterations):
        weights = updateWeights(X, y, weights, learningRate)
    return weights

def extractX(points):
    X = points[:,0]
    X = np.dstack((np.zeros(len(X)), X)) #adding a column of ones for the bias term
    return X[0]

def extractY(points):
    y = points[:,1]
    return y

def run():
    points = np.array(np.genfromtxt("data.csv", delimiter=","))
    learning_rate = 0.0005
    iterations = 1000

    X = extractX(points)
    y = extractY(points)

    weights = trainModel(X, y, learning_rate, iterations)
    print(weights)
    print('Error: {}'.format(meanSquaredError(X, y, weights)))

if __name__ == "__main__":
    run()
