from maze import Maze
import numpy as np
import time
import matplotlib.pyplot as plt


class MDPMazeSolver:
    
    def __init__(self, noise):
        
        self.discount = 0.9
        self.maze = Maze(noise=noise, discount=self.discount)
        self.maze.load_maze("mazes/maze0.txt")
        self.values = np.zeros((6, 4, 4))
            
    def learn_values(self, num_iterations=None):
        
        done = False
        i = 1
        while not done:
            converged = True
            new_values = np.zeros(self.values.shape)
            for x in range(6):
                for y in range(4):
                    for a in range(4):
                        
                        for noisy_a in range(4):
                            # determine odds of taking the action
                            if noisy_a == a:
                                p = 1 - self.maze.noise
                            else:
                                p = self.maze.noise / 3
                                
                            # get the state transitioned to if taking that action
                            next_state = self.maze.is_move_valid(noisy_a, position=(x, y))
                            
                            if next_state is None:
                                next_state = (x, y)
                                
                            # find the value of that state
                            next_value = np.max(self.values[next_state[0], next_state[1]])
                            
                            new_values[x, y, a] += p * (self.maze.rewards[next_state[0], next_state[1]] + self.discount * next_value)
                        
                        if abs(self.values[x, y, a] - new_values[x, y, a]) / (self.values[x, y, a] + 1e-10) > 1e-6:
                            converged = False
                
            self.values = new_values
                
            i += 1
            if num_iterations is not None:
                if i > num_iterations:
                    done = True
            elif converged:
                print("Values converged after %d iterations" % i)
                done = True
                
    def get_action(self, state):
        
        return np.argmax(self.values[state[0], state[1]])
    
    def draw(self):
        
        self.maze.draw_maze(values=np.max(self.values, axis=2))
        
    def step(self, show):
        
        action = self.get_action(self.maze.position)
        
        self.maze.step(action, show=show, values=np.max(self.values, axis=2))
    
if __name__ == "__main__":
    
    noise = 0.2
    solver = MDPMazeSolver(noise=noise)
    
    num_iterations = None
    solver.learn_values(num_iterations)

    display_values = True
    if display_values:
        solver.draw()
        plt.savefig("figures/mdp/value_map_%.1f_noise.png" % noise)
        
        input("Press Enter to close the Figure and end the program...")
    
    show = False
    num_steps = 100
    for i in range(num_steps):
        
        solver.step(show)
        
        if show:
            time.sleep(0.1)
        
    print("Reward for MDP after %d steps with %.1f noise: %.3f" % (num_steps, noise, solver.maze.reward_current))