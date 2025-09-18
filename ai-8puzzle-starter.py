import random
import heapq as heaps
import copy

# Create the goal state to represent the 8-puzzle state space
goal_state = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

# Will produce states states that can be solved in <depth> actions
# Where <depth> is an integer value.
# Ex. If the depth is 4, then the function will return a state that can be solved in 4 actions
def generate_random_start_depth_with_fixed_solution_depth(depth):
    start_state = copy.deepcopy(goal_state)
    for i in range(depth):
        # Get the list of valid actions from the current state
        actions = get_actions(start_state)

        # Randomly select an action from the list of valid actions
        random_action_index = random.randint(0, len(actions) - 1)
        
        last_action = actions[random_action_index]

        start_state = transition_model(start_state, last_action)

    return start_state

def get_actions(state):
    actions = ['u', 'd', 'l', 'r'] # up, down, left, right

    # Need to know the position of the blank tile (0)
    blank_pos = get_blank_pos(state)

    # If the blank is on the top row (row_index == 0), then 'u' is not a valid action
    # On the other hand, if the blank is on the bottom row (row_index == 2), then 'd' is not a valid action
    if blank_pos[0] == 0:
        actions.remove('u')
    elif blank_pos[0] == 2:
        actions.remove('d')

    # If the blank is on the left column (col_index == 0), then 'l' is not a valid action
    # On the other hand, if the blank is on the right column (col_index == 2), then 'r' is not a valid action
    if blank_pos[1] == 0:
        actions.remove('l')
    elif blank_pos[1] == 2:
        actions.remove('r')

    return actions

# Returns the (row_index, col_index) of the blank tile (0)
def get_blank_pos(state):
    for row_index, row in enumerate(state):
        if 0 in row:
            col_index = row.index(0)
            blank_pos = (row_index, col_index)
            return blank_pos

# Returns true or false if the state is the goal state
def goal_test(state):
    return state == goal_state

# Prints the state in a readable format
def print_state(state):
    for row in state:
        print(row)
    

def transition_model(state, action):
    blank_pos = get_blank_pos(state)

    if action == 'u':
        new_blank_pos = (blank_pos[0] - 1, blank_pos[1])

    elif action == 'd':
        new_blank_pos = (blank_pos[0] + 1, blank_pos[1])
    
    elif action == 'l':
        new_blank_pos = (blank_pos[0], blank_pos[1] - 1)

    elif action == 'r':
        new_blank_pos = (blank_pos[0], blank_pos[1] + 1)

    # Swap the blank tile with the adjacent tile
    title_to_swap = state[new_blank_pos[0]][new_blank_pos[1]]
    state[new_blank_pos[0]][new_blank_pos[1]] = 0
    state[blank_pos[0]][blank_pos[1]] = title_to_swap

    return state

def breadth_first_search(start_state):
    frontier = []
    expansion_count = 0

    heaps.heappush(frontier, (0, start_state, [])) # (priority, state, path)

    while frontier:
        curr_value, curr_node = heaps.heappop(frontier)

        curr_state = curr_node[0]
        curr_path = curr_node[1]

        expansion_count += 1

        if curr_state == goal_state:
            print("Goal found!")
            print("Number of expansions: ", expansion_count)
            return curr_path
        
        curr_actions = get_actions(curr_state)

        # Generate child node from each action
        for action in curr_actions:
            new_state = transition_model(copy.deepcopy(curr_state), action)
            next_path = copy.deepcopy(curr_path)
            next_path.append(action)
            next_value = len(next_path) # Priority is the path length for BFS
            
            # Add the child node to the frontier
            heaps.heappush(frontier, (next_value, (new_state, next_path)))



if __name__ == "__main__":
    start = generate_random_start_depth_with_fixed_solution_depth(4)
    print_state(start)
    

    

