from maze import Maze
import numpy as np
import time
import matplotlib.pyplot as plt


class POMDPMazeSolver:
    
    def __init__(self, noise, method):
        
        self.method = method
        self.discount = 0.9
        self.maze = Maze(noise=noise, discount=self.discount)
        self.maze.load_maze("mazes/maze1.txt")
        self.values = np.zeros((6, 4, 6, 4, 4))
        
        self.target_belief = np.ones((6, 4))
        self.target_belief[1, 1] = 0.0
        self.target_belief /= 23
        
    def update_belief(self, robot_state, at_target):
        
        # modify belief to reflect current observation
        if at_target:
            self.target_belief *= 0.0
            self.target_belief[robot_state[0], robot_state[1]] = 1.0
        else:
            self.target_belief[robot_state[0], robot_state[1]] = 0.0
        
        new_belief = np.zeros((6, 4))
        # propagate belief
        for target_x in range(6):
            for target_y in range(4):
                
                for a in range(4):
                    next_state = self.maze.is_move_valid(a, position=(target_x, target_y))
                                    
                    if next_state is None:
                        next_state = (target_x, target_y)
                    
                    new_belief[next_state[0], next_state[1]] += self.target_belief[target_x, target_y] / 4
                    
        new_belief /= np.sum(new_belief)
        
        self.target_belief = new_belief
                
            
    def learn_values(self, num_iterations=None):
        
        done = False
        i = 1
        while not done:
            new_values = np.zeros(self.values.shape)
            for target_x in range(6):
                for target_y in range(4):
                    for robot_x in range(6):
                        for robot_y in range(4):
                            for a in range(4):
                    
                                for noisy_a in range(4):
                                # determine odds of taking the action
                                    if noisy_a == a:
                                        p = 1 - self.maze.noise
                                    else:
                                        p = self.maze.noise / 3
                                        
                                    # get the state transitioned to if taking that action
                                    next_state = self.maze.is_move_valid(noisy_a, position=(robot_x, robot_y))
                                    
                                    if next_state is None:
                                        next_state = (robot_x, robot_y)
                                        
                                    # find the value of that state
                                    next_value = np.max(self.values[target_x, target_y, next_state[0], next_state[1]])
                                    
                                    r_next = int((target_x, target_y) == (next_state[0], next_state[1]))
                                    new_values[target_x, target_y, robot_x, robot_y, a] += p * (r_next + self.discount * next_value)
                
            converged = np.all(np.abs(self.values - new_values) / (self.values + 1e-10) < 1e-6)
            self.values = new_values
                
            i += 1
            if num_iterations is not None:
                if i > num_iterations:
                    done = True
            elif converged:
                print("Values converged after %d iterations" % i)
                done = True
                
    def get_action(self, robot_state):
        
        if self.method == "most_likely":
            return self.get_action_most_likely(robot_state)
        elif self.method == "QMDP":
            return self.get_action_QMDP(robot_state)

    
    def get_action_most_likely(self, robot_state, target_state=None):
        
        if target_state is None:
            target_state = np.unravel_index(np.argmax(self.target_belief), self.target_belief.shape)
        
        return np.argmax(self.values[target_state[0], target_state[1], robot_state[0], robot_state[1]])
    
    def get_action_QMDP(self, robot_state):
        
        values = np.zeros(4)
        for action in self.maze.actions:
            for target_x in range(6):
                for target_y in range(4):
                    belief = self.target_belief[target_x, target_y]
                    
                    values[action] += belief * self.values[target_x, target_y, robot_state[0], robot_state[1], action]
                    
        return np.argmax(values)
    
    def draw(self, target_location):
        
        self.maze.draw_maze(values=np.max(self.values[target_location[0], target_location[1]], axis=2))
        
    def step(self, show):
        
        action = self.get_action(self.maze.position)
        
        self.maze.step(action, move_target=True, show=show, values=self.target_belief)
        
        self.update_belief(self.maze.position, self.maze.get_observation(self.maze.position))
    
if __name__ == "__main__":
    
    method = "most_likely"
    noise = 0.3
    solver = POMDPMazeSolver(noise=noise, method=method)
    
    num_iterations = None
    solver.learn_values(num_iterations)

    # solver.draw(target_location=solver.maze._get_target_location())
    
    # input("Press Enter to close the Figure and end the program...")
    
    show = False
    for i in range(100):
        
        solver.step(show)
        
        if noise == 0.3 and show:
            plt.savefig("figures/pomdp/%s_map%d.png" % (method, i))
        
        if show:
            time.sleep(0.1)
        
    print("Reward using %s after 100 steps with %.1f noise: %.3f" % (method, noise, solver.maze.reward_current))