class Perceptron:
    def __init__(self, n_features,learning_rate=0.1,epochs=50):
        self.w = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs

    def activate(self,z):
        return 1 if z>=0 else 0

    def predict(self, x):
        z = sum(w_i*x_i for w_i, x_i in zip(self.w, x)) + self.bias
        return self.activate(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for x, target in zip(X, y):
                error = target - self.predict(x)
                for i in range(len(self.w)):
                    self.w[i] += self.lr * error * x[i]
                self.bias += self.lr * error

    def score(self, X, y):
        correct = sum(1 for x, t in zip(X, y) if self.predict(x) == t)
        return correct / len(y)

p = Perceptron(n_features=2, learning_rate=1.0, epochs=5)
p.train([[1, 2], [1, 0]], [0, 1])
print(p.w, p.bias)   # [0.0, -2.0] 0.0