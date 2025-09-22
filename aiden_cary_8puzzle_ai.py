# -*- coding: utf-8 -*-
# Aiden Cary
# Implemented second heuristic and A* search

"""

stine
csci {3,5,6}385 ai
2025-09-11

"""

import random
import heapq
import copy
import time
import os

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


def breadth_first_search(start_state, print_states=False):
    frontier = []
    expansion_count = 0

    heapq.heappush(frontier, (0, (start_state, [])))

    while frontier:
        curr_value, curr_node = heapq.heappop(frontier)

        curr_state = curr_node[0]
        curr_path = curr_node[1]

        expansion_count += 1

        if print_states:
            print_state(curr_state)

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

def h2(state):
    """
    # This heuristic function returns the total Manhattan distance of all tiles
    # from their goal positions.
    """
    total_manhattan_distance = 0
    
    for row_index, row in enumerate(state):
        for col_index, tile in enumerate(row):
            if tile != 0:  # Skip the blank tile
                goal_row = (tile - 1) // 3
                goal_col = (tile - 1) % 3
                manhattan_distance = abs(row_index - goal_row) + abs(col_index - goal_col)
                total_manhattan_distance += manhattan_distance
    
    return total_manhattan_distance

def greedy_search(start_state, print_states=False, heuristic=False):
  frontier = []
  expansion_count = 0
  # Select heuristic function based on user's choice: False -> h1, True -> h2
  h = h2 if heuristic else h1
  # f(n) = h(n) for Greedy Best-First Search
  heapq.heappush(frontier, (h(start_state), (start_state, [])))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]

    expansion_count += 1

    if print_states:
      print_state(curr_state)

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
      next_value = h(next_state)

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path)))

def a_star_search(start_state, print_states=False, heuristic=False):
  frontier = []
  expansion_count = 0
  # Select heuristic function based on user's choice: False -> h1, True -> h2
  h = h2 if heuristic else h1
  # f(n) = g(n) + h(n) for A*
  heapq.heappush(frontier, (h(start_state), (start_state, [])))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]

    expansion_count += 1

    if print_states:
      print_state(curr_state)

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
      next_value = len(next_path) + h(next_state)

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path)))

def execute_search(start_state, search_function):
    """
    Executes a search algorithm, prints the solution, solution length, and time taken.

    Args:
        start_state (list): The initial state of the puzzle.
        search_function (function): The search algorithm function to execute.
    """
    start_time = time.time()
    solution = search_function(start_state)
    end_time = time.time()

    print(f"Solution: {solution}")
    print(f"Solution length: {len(solution)}")
    print(f"Time taken: {end_time - start_time:.5f} seconds\n")

def get_heuristic_choice():
    """
    Prompts the user to choose a heuristic for A* Search or Greedy Search.

    Returns:
        bool: False for h1 (misplaced tiles), True for h2 (Manhattan distance).
    """
    print("Choose heuristic:")
    print("1. h1 (Number of misplaced tiles)")
    print("2. h2 (Total Manhattan distance)")
    heuristic_choice = input("Enter your choice: ")
    if heuristic_choice == '1':
        return False
    elif heuristic_choice == '2':
        return True
    else:
        print("Invalid heuristic choice. Please try again.")
        return None

def print_menu():
    print("\n8-Puzzle Solver")
    print("1. Generate a new start state")
    print("2. Run Breadth-First Search")
    print("3. Run Greedy Search")
    print("4. Run A* Search")
    print("5. Run all search algorithms sequentially")
    print("6. Clear terminal")
    print("7. Exit")

if __name__ == '__main__':
  fixed_depth = 3

  print("Welcome to the Aiden's 8-Puzzle Solver!")
  print("=====================================")
  print(f"A default start state with a fixed depth of {fixed_depth} is automatically generated.\n")
  start_state = generate_random_start_w_fixed_depth(fixed_depth)
  print("Generated Start State:")
  print_state(start_state)

  while True:
    print_menu()
    choice = input("Enter your choice: ")

    if choice == '1':
      num_of_moves = int(input("Enter the number of moves to generate the start state: "))
      start_state = generate_random_start_w_fixed_depth(num_of_moves)
      print("Generated Start State:")
      print_state(start_state)

    elif choice in ['2', '3', '4', '5']:
      print_state_option = input("Do you want to print the state during the search? (y/n): ").lower() == 'y'

      if choice == '2':
        print("Running Breadth-First Search...")
        execute_search(start_state, lambda state: breadth_first_search(state, print_state_option))

      elif choice == '3':
        while True:
          heuristic = get_heuristic_choice()
          if heuristic is not None:
            break
        print("Running Greedy Search...")
        execute_search(start_state, lambda state: greedy_search(state, print_state_option, heuristic))

      elif choice == '4':
        while True:
          heuristic = get_heuristic_choice()
          if heuristic is not None:
            break
        print("Running A* Search...")
        execute_search(start_state, lambda state: a_star_search(state, print_state_option, heuristic))

      elif choice == '5':
        while True:
          heuristic = get_heuristic_choice()
          if heuristic is not None:
            break
        print("Running all search algorithms sequentially...\n")
        execute_search(start_state, lambda state: breadth_first_search(state, print_state_option))
        execute_search(start_state, lambda state: greedy_search(state, print_state_option, heuristic))
        execute_search(start_state, lambda state: a_star_search(state, print_state_option, heuristic))

    elif choice == '6':
      os.system('cls')
      start_state = generate_random_start_w_fixed_depth(fixed_depth)
      print("Generated Start State:")
      print_state(start_state)

    elif choice == '7':
      print("Exiting the program.")
      break

    else:
      print("Invalid choice. Please try again.")