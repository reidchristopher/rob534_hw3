from maze import Maze
import numpy as np
import time


class POMDPMazeSolver:
    
    def __init__(self, noise, method):
        
        self.method = method
        self.discount = 0.9
        self.maze = Maze(noise=noise, discount=self.discount)
        self.maze.load_maze("mazes/maze1.txt")
        self.values = np.zeros((6,  4, 6, 4))
        
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
                
                _, next_states, _ = self.get_action_most_likely((target_x, target_y), (0, 0))
                
                scale = 1 / len(next_states)
                
                for state in next_states:
                    
                    new_belief[state[0], state[1]] += self.target_belief[target_x, target_y] * scale
                    
        new_belief /= np.sum(new_belief)
        
        self.target_belief = new_belief
                
            
    def learn_values(self, num_iterations=None):
        
        done = False
        i = 1
        while not done:
            converged = True
            for target_x in range(6):
                for target_y in range(4):
                    for robot_x in range(6):
                        for robot_y in range(4):
                    
                            original_value = self.values[target_x, target_y, robot_x, robot_y]
                            
                            if (robot_x, robot_y) == (target_x, target_y):
                                self.values[target_x, target_y, robot_x, robot_y] = 1.0
                                continue
                            
                            max_state, next_states, max_action = self.get_action_most_likely((robot_x, robot_y), (target_x, target_y))
                            
                            s = 0
                            for action, state in enumerate(next_states):
                                if action == max_action:
                                    p = 1 - self.maze.noise
                                else:
                                    p = self.maze.noise / (len(next_states) - 1)
                                    
                                next_value = self.values[target_x, target_y, state[0], state[1]]
                                s += p * next_value
                            
                            self.values[target_x, target_y, robot_x, robot_y] =  s * self.discount
                            
                            if abs(original_value - self.values[target_x, target_y, robot_x, robot_y]) / (original_value + 1e-6) > 0.001:
                                converged = False
                
            i += 1
            if num_iterations is not None:
                if i > num_iterations:
                    done = True
            elif converged:
                done = True
                
    def get_action(self, robot_state):
        

        if self.method == "most_likely":
            return self.get_action_most_likely(robot_state)
        elif self.method == "QMDP":
            return self.get_action_QMDP(robot_state)

    
    def get_action_most_likely(self, robot_state, target_state=None):
        
        if target_state is None:
            target_state = np.unravel_index(np.argmax(self.target_belief), self.target_belief.shape)
        
        max_value = None
        max_state = None
        max_action = None
        next_states = []
        for action in self.maze.actions:
            
            next_state = self.maze.is_move_valid(action, position=robot_state)
            
            if next_state is None:
                next_state = robot_state
            
            next_states.append(next_state)
            
            next_value = self.values[target_state[0], target_state[1], next_state[0], next_state[1]]
                
            if max_value is None:
                max_value = next_value
                max_state = next_state
                max_action = action
            elif next_value > max_value:
                max_value = next_value
                max_state = next_state
                max_action = action
            
        return max_state, next_states, max_action
    
    def get_action_QMDP(self, robot_state):
        
        max_value = None
        max_state = None
        max_action = None
        next_states = []
        for action in self.maze.actions:
            value = 0
            for target_x in range(6):
                for target_y in range(4):
                    belief = self.target_belief[target_x, target_y]
                
            next_state = self.maze.is_move_valid(action, position=robot_state)
    
            if next_state is None:
                next_state = robot_state
    
            next_states.append(next_state)
    
    def draw(self, target_location):
        
        self.maze.draw_maze(values=self.values[target_location[0], target_location[1]])
        
    def step(self):
        
        _, _, max_action = self.get_action(self.maze.position)
        
        self.maze.step(max_action, move_target=True, show=True, values=self.target_belief)
        
        self.update_belief(self.maze.position, self.maze.get_observation(self.maze.position))
    
if __name__ == "__main__":
    
    solver = POMDPMazeSolver(noise=0.0, method="most_likely")
    
    num_iterations = None
    solver.learn_values(num_iterations)

    # solver.draw(target_location=solver.maze._get_target_location())
    
    # input("Press Enter to close the Figure and end the program...")
    
    for i in range(1000):
        
        solver.step()
        
        time.sleep(0.03)