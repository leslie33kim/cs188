import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(nn.DotProduct(self.w, x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        flag = 1
        while flag:
            flag = False
            for x,y in dataset.iterate_once(batch_size):
                pred = self.get_prediction(x)
                mult = nn.as_scalar(y)
                if pred != mult:
                    self.w.update(x,mult)
                    flag = True
            if flag == False:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.size = 100
        self.learning = 0.001
        self.w1 = nn.Parameter(1, 50)
        self.b1 = nn.Parameter(1,50)
        self.w2 = nn.Parameter(50, 100)
        self.b2 = nn.Parameter(1,100)
        self.w3= nn.Parameter(100, 1)
        self.b3 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        first = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        firstt = nn.ReLU(first)
        second = nn.AddBias(nn.Linear(firstt, self.w2), self.b2)
        secondd = nn.ReLU(second)
        last = nn.AddBias(nn.Linear(secondd, self.w3), self.b3)
        return last

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 100
        par = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        loss = 10000000
        while loss > 0.02:
            for x,y in dataset.iterate_once(batch_size):
                loss = nn.as_scalar(self.get_loss(x,y))
                if loss > 0.02:
                    grad = nn.gradients(self.get_loss(x, y), par)
                    for i in range(6):
                        par[i].update(grad[i], -self.learning)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.learning = 0.1
        self.w1 = nn.Parameter(784, 100)
        self.b1 = nn.Parameter(1,100)
        self.w2 = nn.Parameter(100, 50)
        self.b2 = nn.Parameter(1,50)
        self.w3= nn.Parameter(50, 10)
        self.b3 = nn.Parameter(1,10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        first = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        firstt = nn.ReLU(first)
        second = nn.AddBias(nn.Linear(firstt, self.w2), self.b2)
        secondd = nn.ReLU(second)
        last = nn.AddBias(nn.Linear(secondd, self.w3), self.b3)
        return last

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 250
        par = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        tick = 0
        while True:
            if tick % 5 == 0:
                if dataset.get_validation_accuracy() >= 0.98:
                    return
            tick += 1
            for x,y in dataset.iterate_once(batch_size):
                loss = nn.as_scalar(self.get_loss(x,y))
                grad = nn.gradients(self.get_loss(x, y), par)
                for i in range(6):
                    par[i].update(grad[i], -self.learning)
