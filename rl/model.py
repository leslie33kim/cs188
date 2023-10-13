import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim


        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.learning_rate = 2
        self.numTrainingGames = 2500
        self.batch_size = 200
        self.w1 = nn.Parameter(state_dim, self.batch_size)
        self.b1 = nn.Parameter(1, self.batch_size)
        self.w2 = nn.Parameter(self.batch_size, self.batch_size * 2)
        self.b2 = nn.Parameter(1, self.batch_size * 2)
        self.w3= nn.Parameter(self.batch_size * 2, action_dim)
        self.b3 = nn.Parameter(1, action_dim)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        return nn.SquareLoss(self.run(states), Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        first = nn.AddBias(nn.Linear(states, self.w1), self.b1)
        firstt = nn.ReLU(first)
        second = nn.AddBias(nn.Linear(firstt, self.w2), self.b2)
        secondd = nn.ReLU(second)
        last = nn.AddBias(nn.Linear(secondd, self.w3), self.b3)
        return last

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        par = self.parameters
        grad = nn.gradients(self.get_loss(states, Q_target), par)
        for i in range(6):
            par[i].update(grad[i], -self.learning_rate)
