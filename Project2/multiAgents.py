from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose the best action based on evaluation function
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]
    def evaluationFunction(self, currentGameState, action):
        """
        Enhanced evaluation function for the ReflexAgent.
        """
        # Generate successor state and its attributes
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = currentGameState.getCapsules()

        # Start with the successor state's score
        score = successorGameState.getScore()

        # Food considerations
        foodList = newFood.asList()
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDistance = min(foodDistances)
            score += 10 / closestFoodDistance  # Encourage approaching food
        else:
            score += 100  # Bonus if all food is cleared

        # Ghost considerations
        for i, ghost in enumerate(newGhostStates):
            distanceToGhost = manhattanDistance(newPos, ghost.getPosition())
            if newScaredTimes[i] > 0:  # Ghost is scared
                if distanceToGhost <= newScaredTimes[i]:  # Can eat the ghost
                    score += 200 / (distanceToGhost + 1)  # Encourage eating scared ghosts
            else:  # Ghost is active
                if distanceToGhost <= 1:
                    score -= 1000  # Heavy penalty for getting too close to active ghosts
                score -= 10 / (distanceToGhost + 1)  # Penalize being near active ghosts

        # Capsule considerations
        if capsules:
            capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in capsules]
            closestCapsuleDistance = min(capsuleDistances)
            score += 20 / (closestCapsuleDistance + 1)  # Encourage moving toward capsules

        return score



def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides common elements to all multi-agent searchers.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for adversarial search.
    """

    def getAction(self, gameState):
        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (maximizing)
                return max(minimax(1, depth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
            else:  # Ghosts (minimizing)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return min(minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))

        legalMoves = gameState.getLegalActions(0)
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning for adversarial search.
    """

    def getAction(self, gameState):
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (maximizing)
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    value = max(value, alphaBeta(1, depth, state.generateSuccessor(agentIndex, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts (minimizing)
                value = float("inf")
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    value = min(value, alphaBeta(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        legalMoves = gameState.getLegalActions(0)
        alpha, beta = float("-inf"), float("inf")
        scores = []
        for action in legalMoves:
            value = alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            scores.append(value)
            alpha = max(alpha, value)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent for probabilistic reasoning.
    """

    def getAction(self, gameState):
        def expectimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman (maximizing)
                return max(expectimax(1, depth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
            else:  # Ghosts (expectation)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                legalActions = state.getLegalActions(agentIndex)
                if not legalActions:
                    return self.evaluationFunction(state)
                totalValue = sum(expectimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                                 for action in legalActions)
                return totalValue / len(legalActions)

        legalMoves = gameState.getLegalActions(0)
        scores = [expectimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]


def betterEvaluationFunction(currentGameState):
    """
    A better evaluation function that combines various aspects of the game state to guide Pacman effectively.
    """
    # Basic state information
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    capsules = currentGameState.getCapsules()

    # Start with the current score
    score = currentGameState.getScore()

    # Food-related considerations
    if food:
        foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food]
        closestFoodDistance = min(foodDistances)
        score += 10 / closestFoodDistance  # Encourage moving towards food
        score -= len(food) * 4  # Slight penalty for remaining food
    else:
        score += 100  # Bonus for clearing all food

    # Ghost-related considerations
    for i, ghost in enumerate(ghostStates):
        ghostPos = ghost.getPosition()
        distanceToGhost = manhattanDistance(pacmanPos, ghostPos)

        if scaredTimes[i] > 0:  # Ghost is scared
            if distanceToGhost <= scaredTimes[i]:  # Can eat the ghost
                score += 200 / (distanceToGhost + 1)  # Encourage eating scared ghosts
        else:  # Ghost is active
            if distanceToGhost <= 1:
                score -= 1000  # Avoid getting caught
            score -= 10 / (distanceToGhost + 1)  # Discourage being near ghosts

    # Capsule-related considerations
    if capsules:
        capsuleDistances = [manhattanDistance(pacmanPos, capsule) for capsule in capsules]
        closestCapsuleDistance = min(capsuleDistances)
        score += 20 / (closestCapsuleDistance + 1)  # Encourage moving towards capsules

    return score


better = betterEvaluationFunction
