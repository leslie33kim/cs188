# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        eval_score = childGameState.getScore()
        foodlist = newFood.asList()
        foodScore = 100
        count = 0
        for food in foodlist:
            foodScore = min(foodScore, util.manhattanDistance(food, newPos) + (400* count))
            count += 1
            if count > 50:
                foodScore += 200

        eval_score -= foodScore

        closestGhost = 100
        for ghost in newGhostStates:
            ghostLoc = ghost.getPosition()
            closestGhost = min(closestGhost, util.manhattanDistance(ghostLoc, newPos))
            if closestGhost < 3:
                eval_score -= 5000

        return eval_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        pacmanAct = gameState.getLegalActions(0)
        pac_val = (None, -float('inf'))
        for act in pacmanAct:
            pac_next = gameState.getNextState(0,act)
            score = self.minmin(self.depth, pac_next, 1)[1]
            if pac_val[1] < score:
                pac_val = (act, score)
        return pac_val[0]


    def maxmax(self, depth, gState, index):
        if depth == 0 or gState.isWin() or gState.isLose():
            return (None, self.evaluationFunction(gState))
        max_val = (None,-float('inf'))
        for legalAction in gState.getLegalActions(0):
            nextState = gState.getNextState(0,legalAction)
            score = self.minmin(depth, nextState, 1)[1]
            if max_val[1] < score:
                max_val = (legalAction, score)
        return max_val

    def minmin(self, depth, gState, index):
        def isLastGhost(self, depth, gameState, index):
            if index >= gameState.getNumAgents() - 1:
                return True
            else:
                return False

        if depth == 0 or gState.isWin() or gState.isLose():
            return (None,self.evaluationFunction(gState))
        min_val = (None,float('inf'))
        for legalAction in gState.getLegalActions(index):
            nextState = gState.getNextState(index,legalAction)
            if isLastGhost(self,depth,nextState,index):
                score1 = self.maxmax(depth - 1, nextState, 0)[1]
                if min_val[1] > score1:
                    min_val = (legalAction, score1)
            else:
                score2 = self.minmin(depth, nextState, index + 1)[1]
                if min_val[1] > score2:
                    min_val = (legalAction, score2)
        return min_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        alpha = float('-inf')
        beta = float('inf')
        pac_vall = (None,0)
        pacmanAct = gameState.getLegalActions(0)
        pac_val = (None, -float('inf'))
        for act in pacmanAct:
            pac_next = gameState.getNextState(0,act)
            score = self.minmin(self.depth, pac_next, 1,alpha,beta)[1]
            if pac_val[1] < score:
                pac_val = (act, score)
            if alpha < score:
                alpha = score
        return pac_val[0]


    def maxmax(self, depth, gState, index,alpha,beta):
        max_vall = (None,0)
        if depth == 0 or gState.isWin() or gState.isLose():
            return (None, self.evaluationFunction(gState))
        max_val = (None,-float('inf'))
        for legalAction in gState.getLegalActions(0):
            nextState = gState.getNextState(0,legalAction)
            score = self.minmin(depth, nextState, 1,alpha,beta)[1]
            if max_val[1] < score:
                max_val = (legalAction, score)
            if beta < score:
                return max_val
            alpha = max(alpha, score)
        return max_val


    def minmin(self, depth, gState, index,alpha,beta):
        def isLastGhost(self, depth, gameState, index):
            if index >= gameState.getNumAgents() - 1:
                return True
            else:
                return False

        if depth == 0 or gState.isWin() or gState.isLose():
            return (None,self.evaluationFunction(gState))
        min_val = (None,float('inf'))
        min_vall = (None,0)
        for legalAction in gState.getLegalActions(index):
            nextState = gState.getNextState(index,legalAction)
            if isLastGhost(self,depth,nextState,index):
                score1 = self.maxmax(depth - 1, nextState, 0,alpha,beta)[1]
                if min_val[1] > score1:
                    min_val = (legalAction, score1)
                if alpha > score1:
                    return min_val
                beta = min(beta,score1)
            else:
                score2 = self.minmin(depth, nextState, index + 1,alpha,beta)[1]
                if min_val[1] > score2:
                    min_val = (legalAction, score2)
                if alpha > score2:
                    return min_val
                beta = min(beta,score2)
        return min_val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        pacmanAct = gameState.getLegalActions(0)
        pac_val = (None, -float('inf'))
        for act in pacmanAct:
            pac_next = gameState.getNextState(0,act)
            score = self.avg(self.depth, pac_next, 1)[1]
            if pac_val[1] < score:
                pac_val = (act, score)
        return pac_val[0]


    def maxmax(self, depth, gState, index):
        if depth == 0 or gState.isWin() or gState.isLose():
            return (None, self.evaluationFunction(gState))
        max_val = (None,-float('inf'))
        for legalAction in gState.getLegalActions(0):
            nextState = gState.getNextState(0,legalAction)
            score = self.avg(depth, nextState, 1)[1]
            if max_val[1] < score:
                max_val = (legalAction, score)
        return max_val

    def avg(self, depth, gState, index):
        def isLastGhost(self, depth, gameState, index):
            if index >= gameState.getNumAgents() - 1:
                return True
            else:
                return False
        score1 = 0
        score2 = 0
        if depth == 0 or gState.isWin() or gState.isLose():
            return (None,self.evaluationFunction(gState))
        avg_val = (None,float('inf'))
        for legalAction in gState.getLegalActions(index):
            nextState = gState.getNextState(index,legalAction)
            if isLastGhost(self,depth,nextState,index):
                score1 += self.maxmax(depth - 1, nextState, 0)[1]
                avg_val = (legalAction, score1 / len(gState.getLegalActions(index)))
            else:
                score2 += self.avg(depth, nextState, index + 1)[1]
                avg_val = (legalAction, score2 / len(gState.getLegalActions(index)))
        return avg_val


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    eval_score = currentGameState.getScore()
    foodlist = newFood.asList()
    foodScore = 100
    count = 0
    for food in foodlist:
        foodScore = min(foodScore, util.manhattanDistance(food, newPos) + (800* count))
        count += 1
        if count > 50:
            foodScore += 200

    eval_score -= foodScore
    scaredghostDist = 1000

    closestGhost = 100
    for ghost in newGhostStates:
        ghostLoc = ghost.getPosition()
        closestGhost = min(closestGhost, util.manhattanDistance(ghostLoc, newPos))
        if closestGhost < 3:
            eval_score -= 10000


    capScore = 100
    capsule = currentGameState.getCapsules()
    capcount = 0
    for cap in capsule:
        capScore = min(capScore, util.manhattanDistance(cap, newPos) + (600* capcount))
        capcount +=1
    eval_score -= capScore

    return eval_score

# Abbreviation
better = betterEvaluationFunction
