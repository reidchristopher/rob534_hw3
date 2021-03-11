"""
random_explorer.py

Explores the world purely randomly
This file also demonstrates how you can use the provided
Maze class.
"""
import random
import time
import numpy as np
from maze import Maze


reward = 0
discount = 0.9
cur_discount = discount
# 10% chance to move in a random direction
# Python Maze class can also track the reward/discount (see below)
maze = Maze(noise=0.1, discount=discount)
maze.load_maze("mazes/maze0.txt")

print("Randomly moving in maze with manual calculation...")

# This shows how to move in the maze manually
for _ in range(25):
    # choose a random direction to move in
    move = random.randint(0,3)
    # try to move in the direction
    maze.noisy_move(move)
    # draw the maze
    maze.draw_maze()
    # Update the reward
    reward += maze.get_reward()*cur_discount
    # Update the discount factor
    cur_discount *= discount

    # Print out location of the agent
    print(maze.position)

    # Sleep for a bit so that the user can see the animation
    # Can remove this if running headless
    time.sleep(0.1)

# Report final reward
print("Final Reward:", reward)

# You can reset the maze with reset()
maze.reset()

reward = 0
cur_discount = discount

# This shows how to use the step() function to move in the maze
print("Using step function to move...")
for _ in range(25):
    # choose a random direction to move in
    move = random.randint(0,3)
    # try to move in the direction
    maze.step(move)
    # draw the maze - If you want to draw specific values in the squares
    # You can set the optional values array with the text or values you want to appear
    # In this example, we put random values
    maze.draw_maze(values=np.random.rand(*maze.rewards.shape))

    # Update the reward
    reward += maze.get_reward()*cur_discount
    # Update the discount factor
    cur_discount *= discount
    # These values are also tracked by the Maze class when step() is used
    assert reward == maze.reward_current
    assert cur_discount == maze.discount_current, f"{maze.discount_current} vs {cur_discount}"

    # Print out location of the agent
    print(maze.position)

    # Sleep for a bit so that the user can see the animation
    # Can remove this if running headless
    time.sleep(0.1)

# Report final reward
print("Final Reward:", reward)
print("Reward tracked by maze class:", maze.reward_current)


print("Now moving target too...")

# This shows how to use the step() function to move in the maze
# with the target moving (For part 2)
for _ in range(25):
    # choose a random direction to move in
    move = random.randint(0,3)
    # try to move in the direction
    # You can set the optional `move_target` parameter to True to move the target
    # You can also set the optional `show` parameter to True to draw the maze
    maze.step(move, move_target=True, show=True)

    # Update the reward
    reward += maze.get_reward()*cur_discount
    # Update the discount factor
    cur_discount *= discount

    # Check to see if you are at the goal
    goal_info = "Not At Goal"
    if maze.get_observation(maze.position):
        goal_info = "At Goal"

    # Print out location of the agent
    print(f"{maze.position}: {goal_info}")

    # Sleep for a bit so that the user can see the animation
    # Can remove this if running headless
    time.sleep(0.1)

# Report final reward - Just to show they are the same :)
print("Final Reward:", reward)
print("Reward tracked by maze class:", maze.reward_current)

# Need this or the figure will close immediately due to non-blocking
# To give you a chance to save the final image
input("Press Enter to close the Figure and end the program...")
