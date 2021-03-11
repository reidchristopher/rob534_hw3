from maze import Maze
import numpy as np



class MDPMazeSolver:
    
    def __init__(self, noise):
        
        self.discount = 0.9
        self.maze = Maze(noise=noise, discount=self.discount)
        self.maze.load_maze("mazes/maze0.txt")
        self.values = np.zeros((6,  4))
            
    def learn_values(self, num_iterations=None):
        
        done = False
        i = 1
        while not done:
            converged = True
            for x in range(6):
                for y in range(4):
                    original_value = self.values[x, y]
                    
                    max_state, next_states, _ = self.get_action((x, y))
                    
                    s = 0
                    for state in next_states:
                        if state == max_state:
                            p = 1 - self.maze.noise
                        else:
                            p = self.maze.noise / (len(next_states) - 1)
                            
                        next_value = self.values[state[0], state[1]]
                        s += p * next_value
                    
                    self.values[x, y] =  self.maze.rewards[x, y] + s * self.discount
                    
                    if abs(original_value - self.values[x, y]) / (original_value + 1e-10) > 1e-6:
                        converged = False
                
            i += 1
            if num_iterations is not None:
                if i > num_iterations:
                    done = True
            elif converged:
                done = True
                
    def get_action(self, state):
        
        max_value = None
        max_state = None
        max_action = None
        next_states = []
        for action in self.maze.actions:
            
            next_state = self.maze.is_move_valid(action, position=state)
            
            if next_state is None:
                next_state = state
            
            next_states.append(next_state)
            next_value = self.values[next_state[0], next_state[1]]
            if max_value is None:
                max_value = next_value
                max_state = next_state
                max_action = action
            elif next_value > max_value:
                max_value = next_value
                max_state = next_state
                max_action = action
            
        return max_state, next_states, max_action
    
    def draw(self):
        
        self.maze.draw_maze(values=self.values)
    
if __name__ == "__main__":
    
    solver = MDPMazeSolver(noise=0.1)
    
    num_iterations = None
    solver.learn_values(num_iterations)
    
    solver.draw()
    
    input("Press Enter to close the Figure and end the program...")