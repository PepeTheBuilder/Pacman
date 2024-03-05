# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random
from collections import deque
import heapq


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def dfsRecursive(node, problem, directions):
    visited, done = [], [False]

    def recursive_helper(node):
        visited.append(node[0])

        if problem.isGoalState(node[0]):
            done[0] = True
            directions.insert(0, node[1])
            return

        neighbors = problem.getSuccessors(node[0])

        for neighbor in neighbors:
            if neighbor[0] not in visited and not done[0]:
                recursive_helper(neighbor)
                if done[0]:
                    directions.insert(0, node[1])

    recursive_helper(node)


def depthFirstSearchKnight(problem):
    aux = (problem.getStartState(), "South", 1)

    directionsDFS = []
    dfsRecursive(aux, problem, directionsDFS)

    if len(directionsDFS) > 0:
        directionsDFS.pop(0)

    flattened_actions = [action for sublist in directionsDFS for action in sublist]
    return flattened_actions


def depthFirstSearch(problem):
    aux = (problem.getStartState(), "South", 1)

    actions = []
    dfsRecursive(aux, problem, actions)

    if len(actions) > 0:
        actions.pop(0)

    return actions


def printShit(problem):
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    neighbors = []
    size = len(problem.getSuccessors(problem.getStartState()))
    aux = (problem.getStartState(), "South", 1)
    print "Abominatie "
    print aux
    print problem.getSuccessors(aux[0])[0]

    util.raiseNotDefined()


def randomMoves(problem):
    "* YOUR CODE HERE *"
    direction = []
    currentState = problem.getStartState()
    """print "random my nr is:", nrRandom, nr_alegeri"""
    while not problem.isGoalState(currentState):
        l = problem.getSuccessors(currentState)
        nrRandom = round(random.random() * 10, 0)
        nr_alegeri = len(l)
        nrRandom = nrRandom % nr_alegeri
        nrRandom = int(nrRandom)
        currentState = l[nrRandom][0]
        direction.append(l[nrRandom][1])

    return direction


def breadthFirstSearchKnight(problem):
    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    visited = []  # Changed from set to list
    queue = deque([(start_state, [])])  # Queue of state-action pairs

    while queue:
        current_state, actions = queue.popleft()

        if current_state in visited:
            continue

        visited.append(current_state)

        if problem.isGoalState(current_state):
            flattened_actions = [action for sublist in actions for action in sublist]
            return flattened_actions

        successors = problem.getSuccessors(current_state)

        for next_state, action, _ in successors:
            if next_state not in visited:
                queue.append((next_state, actions + [action]))

    return []  # Return an empty list if no solution is found


def breadthFirstSearch(problem):
    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    visited = []  # Changed from set to list
    queue = deque([(start_state, [])])  # Queue of state-action pairs

    while queue:
        current_state, actions = queue.popleft()

        if current_state in visited:
            continue

        visited.append(current_state)

        if problem.isGoalState(current_state):
            return actions

        successors = problem.getSuccessors(current_state)

        for next_state, action, _ in successors:
            if next_state not in visited:
                queue.append((next_state, actions + [action]))

    return []  # Return an empty list if no solution is found


def uniformCostSearch(problem):
    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    visited = set()
    priority_queue = [(0, start_state, [])]  # Priority queue of (cost, state, actions) tuples

    while priority_queue:
        current_cost, current_state, actions = heapq.heappop(priority_queue)

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoalState(current_state):
            return actions

        successors = problem.getSuccessors(current_state)

        for next_state, action, step_cost in successors:
            if next_state not in visited:
                next_cost = current_cost + step_cost
                heapq.heappush(priority_queue, (next_cost, next_state, actions + [action]))

    return []  # Return an empty list if no solution is found


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def find_goal_states(problem):
    start_state = problem.getStartState()
    goal_states = []

    visited = set()
    queue = deque([start_state])

    while queue:
        current_state = queue.popleft()

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoalState(current_state):
            goal_states.append(current_state)

        successors = problem.getSuccessors(current_state)

        for next_state, _, _ in successors:
            if next_state not in visited:
                queue.append(next_state)

    return goal_states


def aStarSearchKnight(problem, heuristic):
    startingLocation = problem.getStartState()
    solution = []
    if problem.isGoalState(startingLocation):
        return solution
    visited = []
    pCoada = util.PriorityQueue()
    pCoada.push((startingLocation, [], 0), 0)
    while not pCoada.isEmpty():
        current, solution, currentCost = pCoada.pop()
        if not (current in visited):
            visited.append(current)
            if problem.isGoalState(current):
                flattened_actions = [action for sublist in solution for action in sublist]
                return flattened_actions
            succ = problem.getSuccessors(current)
            for nextLocation, nextDirection, cost in succ:
                newCost = currentCost + cost
                heuristicCost = newCost + heuristic(nextLocation, problem)
                pCoada.push((nextLocation, solution + [nextDirection], newCost), heuristicCost)
    util.raiseNotDefined()


def aStarSearch(problem, heuristic):
    startingLocation = problem.getStartState()
    solution = []
    if problem.isGoalState(startingLocation):
        return solution
    visited = []
    pCoada = util.PriorityQueue()
    pCoada.push((startingLocation, [], 0), 0)
    while not pCoada.isEmpty():
        current, solution, currentCost = pCoada.pop()
        if not (current in visited):
            visited.append(current)
            if problem.isGoalState(current):
                return solution
            succ = problem.getSuccessors(current)
            for nextLocation, nextDirection, cost in succ:
                newCost = currentCost + cost
                heuristicCost = newCost + heuristic(nextLocation, problem)
                pCoada.push((nextLocation, solution + [nextDirection], newCost), heuristicCost)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
bfsk = breadthFirstSearchKnight
dfs = depthFirstSearch
dfsk = depthFirstSearchKnight
astar = aStarSearch
astark = aStarSearchKnight
ucs = uniformCostSearch
rand = randomMoves