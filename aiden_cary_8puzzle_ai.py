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
  return state == goal_state


def breadth_first_search(start_state, print_states=False):
    '''
    This function implements the Breadth-First Search algorithm to solve the 8-puzzle problem.
    It uses a priority queue (min-heap) to explore nodes in order of their depth
    '''
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
    # Manhattan distance is the sum of the absolute differences between
    # the current row/column indices and the goal row/column indices for each tile.
    """
    total_manhattan_distance = 0
    
    for row_index, row in enumerate(state):
        for col_index, tile in enumerate(row):
            if tile != 0:  # Skip the blank tile
                goal_row = (tile - 1) // 3 # Integer division to find the goal row
                goal_col = (tile - 1) % 3 # Modulus to find the goal column
                # Calculate Manhattan distance
                manhattan_distance = abs(row_index - goal_row) + abs(col_index - goal_col)
                total_manhattan_distance += manhattan_distance
    
    return total_manhattan_distance

def greedy_search(start_state, print_states=False, heuristic_fn=h1, log_values=False):
  '''
  This function implements the Greedy Best-First Search algorithm to solve the 8-puzzle problem.
  f(n) = h(n).
  '''
  frontier = []
  expansion_count = 0
  expanded_states = set()
  # f(n) = h(n) for Greedy Best-First Search; carry g for consistency though it's unused in priority
  heapq.heappush(frontier, (heuristic_fn(start_state), (start_state, [], 0)))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]
    curr_g = curr_node[2]

    # Serialize state to a tuple for set membership
    state_key = tuple(tuple(row) for row in curr_state)
    if state_key in expanded_states:
      # Skip states we've already expanded
      continue
    expanded_states.add(state_key)
    expansion_count += 1

    # Print the current state if requested
    if print_states:
      print_state(curr_state)

    # Log g/h/f values if requested
    if log_values:
      h_curr = heuristic_fn(curr_state)
      f_curr = h_curr  # Greedy uses f=h
      print(f"[Greedy][EXPAND] g={curr_g} h={h_curr} f={f_curr} path={''.join(curr_path) if curr_path else '∅'}")

    # Check for goal state
    if curr_state == goal_state:
      print(f"Greedy Search Expansion Count: {expansion_count}")
      return curr_path

    # Get available actions from current state
    curr_actions = get_actions(curr_state)
    
    # Prevent immediate backtracking (don't undo the last move)
    if len(curr_path) > 0:
      last_action = curr_path[-1] 
      reverse = {'u': 'd', 'd': 'u', 'l': 'r', 'r': 'l'}[last_action] # Get the reverse action using a dictionary
      if reverse in curr_actions:
        curr_actions.remove(reverse)

    # Generate child node from each action
    for action in curr_actions:
      next_state = transition_model(curr_state, action)
      next_path = copy.deepcopy(curr_path)
      next_path.append(action)
      next_g = curr_g + 1
      # f(n) = h(n)
      next_value = heuristic_fn(next_state)

      # Log g/h/f values if requested
      if log_values:
        print(f"[Greedy][GEN] a={action} g={next_g} h={heuristic_fn(next_state)} f={next_value}")

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path, next_g)))

def a_star_search(start_state, print_states=False, heuristic_fn=h1, log_values=False):
  '''
  This function implements the A* Search algorithm to solve the 8-puzzle problem.
  It uses a priority queue (min-heap) to explore nodes based on the f(n) = g(n) + h(n) value.
  '''
  frontier = []
  expansion_count = 0
  expanded_states = set()
  heapq.heappush(frontier, (heuristic_fn(start_state), (start_state, [], 0)))

  while frontier:
    curr_value, curr_node = heapq.heappop(frontier)

    curr_state = curr_node[0]
    curr_path = curr_node[1]
    curr_g = curr_node[2]

    # Serialize state to a tuple for set membership
    state_key = tuple(tuple(row) for row in curr_state)
    if state_key in expanded_states:
      # Skip states we've already expanded
      continue
    expanded_states.add(state_key)
    expansion_count += 1

    # Print the current state if requested
    if print_states:
      print_state(curr_state)

    # Log g/h/f values if requested
    if log_values:
      h_curr = heuristic_fn(curr_state)
      f_curr = curr_g + h_curr
      print(f"[A*][EXPAND] g={curr_g} h={h_curr} f={f_curr} path={''.join(curr_path) if curr_path else '∅'}")

    # Check for goal state
    if curr_state == goal_state:
      print(f"A* Search Expansion Count: {expansion_count}")
      return curr_path

    # Get available actions from current state
    curr_actions = get_actions(curr_state)

    # Prevent immediate backtracking (don't undo the last move)
    if len(curr_path) > 0:
      last_action = curr_path[-1]
      reverse = {'u': 'd', 'd': 'u', 'l': 'r', 'r': 'l'}[last_action]
      if reverse in curr_actions:
        curr_actions.remove(reverse)

    # Generate child node from each action
    for action in curr_actions:
      next_state = transition_model(curr_state, action)
      next_path = copy.deepcopy(curr_path)
      next_path.append(action)
      next_g = curr_g + 1
      # f(n) = g(n) + h(n)
      next_value = next_g + heuristic_fn(next_state)

      # Log g/h/f values if requested
      if log_values:
        print(f"[A*][GEN] a={action} g={next_g} h={heuristic_fn(next_state)} f={next_value}")

      # Add child to priority queue (our frontier)
      heapq.heappush(frontier, (next_value, (next_state, next_path, next_g)))

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
  """
  print("Choose heuristic:")
  print("1. h1 (Number of misplaced tiles)")
  print("2. h2 (Total Manhattan distance)")
  heuristic_choice = input("Enter your choice: ")
  if heuristic_choice == '1':
    return h1
  elif heuristic_choice == '2':
    return h2
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
    print("6. Clear terminal & Restart with fixed depth")
    print("7. Exit")

if __name__ == '__main__':
  
  # Hard coded fixed depth for testing
  fixed_depth = 3

  print("Welcome to the Aiden's 8-Puzzle Solver!")
  print("=====================================")
  print(f"A default start state with a fixed depth of {fixed_depth} is automatically generated.\n")
  start_state = generate_random_start_w_fixed_depth(fixed_depth)
  print("Generated Start State:")
  print_state(start_state)

  # Main loop
  while True:
    print_menu()
    choice = input("Enter your choice: ")

    # Generate a new start state
    if choice == '1':
      num_of_moves = int(input("Enter the number of moves to generate the start state: "))
      start_state = generate_random_start_w_fixed_depth(num_of_moves)
      print("Generated Start State:")
      print_state(start_state)

    # Options for printing states and/or logging g/h/f values
    elif choice in ['2', '3', '4', '5']:
      print_state_option = input("Do you want to print the state during the search? (y/n): ").lower() == 'y'
      # May also add option to write to file later but will need to modify execute_search function and search functions
      # write_to_file = input("Do you want to write the output to a file? (y/n): ").lower() == 'y'
      if choice in ['3', '4', '5']:
        log_values = input("Do you want to log g/h/f values during the search? (y/n): ").lower() == 'y'
      
      # Execute the chosen search algorithm(s)
      # Using lambda to pass additional parameters
      if choice == '2':
        print("Running Breadth-First Search...")
        execute_search(start_state, lambda state: breadth_first_search(state, print_state_option))

      elif choice == '3':
        while True:
          heuristic_fn = get_heuristic_choice()
          if heuristic_fn is not None:
            break
        print("Running Greedy Search...")
        execute_search(start_state, lambda state: greedy_search(state, print_state_option, heuristic_fn, log_values))

      elif choice == '4':
        while True:
          heuristic_fn = get_heuristic_choice()
          if heuristic_fn is not None:
            break
        print("Running A* Search...")
        execute_search(start_state, lambda state: a_star_search(state, print_state_option, heuristic_fn, log_values))

      elif choice == '5':
        while True:
          heuristic_fn = get_heuristic_choice()
          if heuristic_fn is not None:
            break
        print("Running all search algorithms sequentially...\n")
        execute_search(start_state, lambda state: breadth_first_search(state, print_state_option))
        execute_search(start_state, lambda state: greedy_search(state, print_state_option, heuristic_fn, log_values))
        execute_search(start_state, lambda state: a_star_search(state, print_state_option, heuristic_fn, log_values))
          

    # Clear terminal and regenerate start state at hard coded fixed depth
    elif choice == '6':
      os.system('cls')
      start_state = generate_random_start_w_fixed_depth(fixed_depth)
      print(f"Generated Start State at fixed depth of {fixed_depth}:")
      print_state(start_state)

    elif choice == '7':
      print("Exiting the program.")
      break

    else:
      print("Invalid choice. Please try again.")