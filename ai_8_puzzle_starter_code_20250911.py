# -*- coding: utf-8 -*-

"""

stine
csci {3,5,6}385 ai
2025-09-11

"""

import random
import heapq
import copy
import time

goal_state = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]


def generate_random_start_w_fixed_depth(depth):
  """
  # This function will produce start states that can be solved in <depth> actions
  # where <depth> is an integer value. I'd suggest starting small to avoid crashing.
  #
  """
  start_state = copy.deepcopy(goal_state)
  action_log = []

  for i in range(depth):
    actions = get_actions(start_state)

    # The following is to help avoid taking an action which simply reverses the
    # immediately preceding action.
    if len(action_log) > 0:
      last_action = action_log[-1] # This returns the last action done.
      if last_action == 'u':
        actions.remove('d')
      elif last_action == 'd':
        actions.remove('u')
      elif last_action == 'l':
        actions.remove('r')
      elif last_action == 'r':
        actions.remove('l')

    random_action_index = random.randint(0, len(actions) - 1)

    next_action = actions[random_action_index]

    start_state = transition_model(start_state, next_action)

    action_log.append(next_action)

  return start_state


def goal_test(state):
  """
  # If state is the same as goal_state, then the function returns True. Otherwise,
  # it returns False.
  """
  if state == goal_state:
    return True
  else:
    return False


def print_state(state):
  """
  # Prints the state with some formatting.
  """
  print(f'{state[0][0]}|{state[0][1]}|{state[0][2]}')
  print('— — —')
  print(f'{state[1][0]}|{state[1][1]}|{state[1][2]}')
  print('— — —')
  print(f'{state[2][0]}|{state[2][1]}|{state[2][2]}\n')


def get_blank_pos(state):
  """
  # For the given state, returns a tuple describing the position of the blank
  # such that blank_pos[0] is the row index and blank_pos[1] is the column index.
  """
  for row_index, row in enumerate(state):
    if 0 in row:
      col_index = row.index(0)
      blank_pos = (row_index, col_index)
      return blank_pos


def get_actions(state):
  """
  # Returns the available actions that can be taken from state.
  """
  actions = ['u', 'd', 'l', 'r']  # ie up, down, left, right

  # First, we need to know the position of the blank.
  blank_pos = get_blank_pos(state)

  # If the blank is on the top row (row_index == 0), then 'u' will not be a possible action.
  # On the other hand, if the blank is on the bottom row (row_index == 2), then 's' will not
  # be a possible action.
  if blank_pos[0] == 0:
    actions.remove('u')
  elif blank_pos[0] == 2:
    actions.remove('d')

  # If the blank is in the left column (col_index == 0), then 'l' will not be possible.
  # Else, if it is in the right column (col_index == 2), then 'r' will not be possible.
  if blank_pos[1] == 0:
    actions.remove('l')
  elif blank_pos[1] == 2:
    actions.remove('r')

  return actions

def transition_model(state, action):
  """
  # Given the current state and an action, return the state that would result
  # by carrying out the action from the state.
  """
  blank_pos = get_blank_pos(state)

  next_state = copy.deepcopy(state)

  if action == 'u':
    # The tile that will be switched will have the same column as the blank,
    # but will be one row higher (i.e., blank_pos[0] - 1)
    swap_pos = (blank_pos[0] - 1, blank_pos[1])

  elif action == 'd':
    # The tile to swap is below the blank, so it is in row blank_pos[0] + 1
    swap_pos = (blank_pos[0] + 1, blank_pos[1])

  elif action == 'l':
    # The tile to swap is to the left of the blank, so it is in column blank_pos[1] - 1
    swap_pos = (blank_pos[0], blank_pos[1] - 1)

  elif action == 'r':
    # The tile to swap is to the right of the blank, so it is in column blank_pos[1] + 1
    swap_pos = (blank_pos[0], blank_pos[1] + 1)

  # swap_pos[0] == row index of where the blank is going
  # swap_pos[1] == col index of where the blank is going
  tile_to_swap = state[swap_pos[0]][swap_pos[1]]

  next_state[blank_pos[0]][blank_pos[1]] = tile_to_swap
  next_state[swap_pos[0]][swap_pos[1]] = 0

  return next_state


def goal_test(state):
  if state == goal_state:
    return True
  else:
    return False


def breadth_first_search(start_state):
  frontier = []
  expansion_count = 0

  heapq.heappush(frontier,
                (0, (start_state, [])))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]

    expansion_count += 1  # x = x + 1

    if curr_state == goal_state:
      print(f"Breadth-First Search Expansion Count: {expansion_count}")
      return curr_path

    # If it doesn't pass the goal test, we move on to generating child nodes.
    # Get available actions from current state
    curr_actions = get_actions(curr_state)

    # To help mitigate expanding nodes which have themselves as grandparent,
    # let's make sure to not consider any action which simply reverses whatever
    # the last action was. For example, if we just moved the blank up a space,
    # then it would be kind of silly to generate a node by moving the blank back
    # down, right? Note that we shouldn't do this if we are expanding the start
    # node, since no prior actions have been taken yet, which is the case when
    # len(curr_path) == 0.
    if len(curr_path) > 0:
        last_action = curr_path[-1] # This returns the last element in curr_path

        if last_action == 'u':
          curr_actions.remove('d')
        elif last_action == 'd':
          curr_actions.remove('u')
        elif last_action == 'l':
          curr_actions.remove('r')
        elif last_action == 'r':
          curr_actions.remove('l')

    # Generate child node from each action
    for action in curr_actions:

      next_state = transition_model(curr_state, action)
      next_path = copy.deepcopy(curr_path)
      next_path.append(action)
      next_value = len(next_path)

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path)))

def h1(state):
  """
  # This heuristic function returns the number of misplaced tiles.
  """
  num_misplaced = 0

  for row_index, row in enumerate(state):
    for col_index, tile in enumerate(row):
      if tile != goal_state[row_index][col_index]:
        num_misplaced += 1

  return num_misplaced

def greedy_search(start_state):
  """
  # This function implements the greedy search algorithm using the h1 heuristic.
  """
  frontier = []
  expansion_count = 0
  heapq.heappush(frontier, (h1(start_state), (start_state, [])))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]

    expansion_count += 1

    if curr_state == goal_state:
      print(f"Greedy Search Expansion Count: {expansion_count}")
      return curr_path

    # Get available actions from current state
    curr_actions = get_actions(curr_state)

    # Generate child node from each action
    for action in curr_actions:
      next_state = transition_model(curr_state, action)
      next_path = copy.deepcopy(curr_path)
      next_path.append(action)
      # f(n) = h(n)
      next_value = h1(next_state)

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path)))

def a_star_search(start_state):
  """
  # This function implements the A* search algorithm using the h1 heuristic.
  """
  frontier = []
  expansion_count = 0
  heapq.heappush(frontier, (h1(start_state), (start_state, [])))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]

    expansion_count += 1

    if curr_state == goal_state:
      print(f"A* Search Expansion Count: {expansion_count}")
      return curr_path

    # Get available actions from current state
    curr_actions = get_actions(curr_state)

    # Generate child node from each action
    for action in curr_actions:
      next_state = transition_model(curr_state, action)
      next_path = copy.deepcopy(curr_path)
      next_path.append(action)
      # f(n) = g(n) + h(n)
      next_value = len(next_path) + h1(next_state)

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path)))

if __name__ == '__main__':
  # Generate a random start state
  num_of_moves = 5
  print(f"Generating a start state that can be solved in {num_of_moves} moves or less...\n")
  start = generate_random_start_w_fixed_depth(num_of_moves)
  print_state(start)

  # Solve the puzzle using breadth-first search.
  start_time_bfs = time.time()
  solution_bfs = breadth_first_search(start)
  end_time_bfs = time.time()
  print("Breadth-First Search Solution:", solution_bfs)
  print(f"Solution length: {len(solution_bfs)}")
  print(f"Time taken for BFS: {end_time_bfs - start_time_bfs:.5f} seconds\n")

  # Solve the puzzle using greedy search.
  start_time_greedy = time.time()
  solution_greedy = greedy_search(start)
  end_time_greedy = time.time()
  print("Greedy Search Solution:", solution_greedy)
  print(f"Solution length: {len(solution_greedy)}")
  print(f"Time taken for Greedy Search: {end_time_greedy - start_time_greedy:.5f} seconds")

  # Solve the puzzle using A* search.
  start_time_a_star = time.time()
  solution_a_star = a_star_search(start)
  end_time_a_star = time.time()
  print("A* Search Solution:", solution_a_star)
  print(f"Solution length: {len(solution_a_star)}")
  print(f"Time taken for A* Search: {end_time_a_star - start_time_a_star:.5f} seconds")