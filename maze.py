"""
Adapted from class code
Major difference is a move from a 1D internal memory allocation to a proper 2D memory allocation. 
Thus, there is no need to translate between index and maze (x,y) positions.

Code by Scott Chow and Connor Yates.

"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Maze:
    def __init__(self, noise=0.1, discount=0.9):
        self.noise = noise
        self.actions = [0, 1, 2, 3]
        # Position starts off as the initial (0, 0)
        self.position = (0, 0)
        self.rewards = None
        self.adjacent = None
        self.figure = None

        # Simulaton Parameters
        self.time_step = 0
        self.discount_start = discount
        self.discount_current = discount
        self.reward_current = 0
        
        # Remember filename for resetting
        self.filename = None

    def step(self, move, move_target=False, show=False, values=None):
        """
        Takes a single step in time and updates simulation parameters

        Moves:
            0 == North
            1 == East
            2 == South
            3 == West

        :param: move, Int, one of the four moves
        :param: move_target, Bool, if True, moves the target
        :param: show, Bool, if True, draw the maze
        :param: values, 2d array of strings the same size as self.rewards, labels to be displayed on the grid 
        :return: the non-discounted reward for the move
        """
        if show:
            self.draw_maze(values)
        self.noisy_move(move)

        reward = self.get_reward()
        self.reward_current += reward*self.discount_current
        self.discount_current *= self.discount_start
        self.time_step += 1

        # Move the target if enabled
        if move_target:
            self.move_target()
        return reward

    def reset(self):
        """
        Resets all simulation parameters to their starting values/0

        :return: None
        """
        self.position = (0, 0)
        self.time_step = 0
        self.reward_current = 0
        self.discount_current = self.discount_start
        self.load_maze(self.filename)

    def get_start(self):
        """
        As per class version, this returns both the starting index and the number of nodes in the maze
        This also returns the start as a tuple (x, y) instead of a single index.

        :return: (0, 0), and the size of the world
        """
        shape = self.adjacent.shape
        return (0, 0), shape[0]*shape[1]

    def get_reward(self, position=None):
        """
        Provides the reward for the given position
        uses the current position if no position provided

        :return: reward at position
        """
        if position is None:
            position = self.position

        return self.rewards[position[0], position[1]]

    def load_maze(self, filename):
        """
        Loads a file from the given pathname
        
        :param: filename: path to the file to load
        :return: None
        """
        self.filename = filename
        # Side note, I want to find Jeremy Kubica circa 2003
        # in a time machine and question *why on this good earth*
        # he thought this was an appropriate way to store the map data
        with open(filename, 'r') as f:
            line = f.readline()
            # Extract map size from first line
            rows, cols = [int(x) for x in line.strip('\n').split(" ")]
            self.adjacent = np.zeros((cols, rows, 4))

            # Load the adjacency matrix
            for ind in range(rows*cols):
                line = f.readline()
                adjs = [int(x) for x in line.strip("\n").split(" ")]
                x = math.floor(ind/rows)
                y = ind % rows
                self.adjacent[x,y,:] = adjs
            
            # Read blank line
            _ = f.readline()

            # Load the rewards
            self.rewards = np.zeros((cols, rows))
            line = f.readline()
            data = [int(x) for x in line.strip("\n").split(" ")]
            for ind in range(rows*cols):
                x = math.floor(ind/rows)
                y = ind % rows
                self.rewards[x,y] = data[ind]
            
            self.x_max = cols-1
            self.y_max = rows-1

            self.x_max_range = cols
            self.y_max_range = rows

            self.rows = rows
            self.cols = cols

    def is_position_valid(self, position):
        """
        Checks if position is within bounds

        :param: position, (x,y) position
        :return: True if input position is within bound
        """
        x, y = position
        return (x >= 0) and (y >= 0) and (x <= self.x_max) and (y <= self.y_max)


    def is_move_valid(self, move, position=None):
        """
        Checks if a given move (m \in 0,1,2,3) is valid with the current location and maze structure.
        Returns the successor position if it is valid
        Otherwise returns None

        Moves:
            0 == North
            1 == East
            2 == South
            3 == West

        :position: (x,y) position of starting location
                         defaults to self.position if not provided

        :return: (x,y) new position after the applying the move
                 If move was invalid, returns None
        """

        if position is None:
            position = self.position

        nx = position[0]
        ny = position[1]

        # If the transition is invalid, just return the current position
        # Note: move-1 because Python is 0-indexed
        if self.adjacent[nx, ny, move] == 0:
            return None

        # Move while checking potential collisions

        # moves north
        if move == 0:
            if ny > 0:
                ny -= 1
            else:
                return None

        # move east
        elif move == 1:
            if nx < self.x_max:
                nx += 1
            else:
                return None

        # move south
        elif move == 2:
            if ny < self.y_max:
                ny += 1
            else:
                return None

        # move west
        elif move == 3:
            if nx > 0:
                nx -= 1
            else:
                return None

        return (nx, ny)

    def move_maze(self, move, position=None, noise=None):
        """
        Performs the move with noise, which means a chance of a uniformly random movement.

        :param: move: Integer in [0, 1, 2, 3], corresponding to each direction.
        :param: position: Tuple, x,y position of starting location
                         Defaults to self.position if not provided
        :param: noise: Float, chance of unsuccessful move
                      Defaults to self.noise if not provided
        :return: (x,y) new position after the applying the move
                 If move was invalid, returns previous position
        """
        if noise is None:
            noise = self.noise

        if position is None:
            position = self.position

        # select a random move that is not the original move
        if random.random() < noise:
            moves = [0,1,2,3]
            moves.remove(move)

            # Remove all non-valid moves from consideration
            # for m in moves:
            #     if self.adjacent[position[0]][position[1]][m] == 0:
            #         moves.remove(m)

            move_actual = random.choice(moves)
        else:
            move_actual = move

        next_position = self.is_move_valid(move_actual, position=position)

        # return previous position if move was invalid
        if next_position is None:
            return position
        
        return next_position

    def move_target(self, target_position=None):
        """
        Moves the target North, West, East, or South of its current position randomly
        If target is at a wall, chance that it does not move

        Moves the target to target_position if provided

        :param: target_position, (x,y), where to put the target

        :return: None
        """

        current_x, current_y = self._get_target_location()

        # Move north with probability 0.25
        # Otherwise move in other directions with probabiliy 0.75
        if target_position is None:
            new_x, new_y = self.move_maze(1, position=(current_x, current_y), noise=0.75)
        else:
            new_x, new_y = target_position
        # Update rewards with new target location
        self.rewards[current_x, current_y] = 0
        self.rewards[new_x, new_y] = 1

    def noisy_move(self, move):
        """
        Performs the move with noise, which means a chance of a uniformly random movement.

        :param: move: Integer in [0, 1, 2, 3], corresponding to each direction.
        :return: None
        """

        self.position = self.move_maze(move, position=self.position, noise=self.noise)

    def get_observation(self, current_loc):
        """
        Returns whether current_loc is the target

        :param: current_loc: Tuple, (x,y) position to check
        :return: Bool, True if current_loc is the target
        """
        return self.rewards[current_loc[0], current_loc[1]]==1

    # Helper functions

    def _get_target_location(self):
        """
        Returns the target location of the maze
        
        :return: Tuple, (x,y) location of target
        """
        current_target_locs = np.argwhere(self.rewards == 1)

        if len(current_target_locs) == 0:
            raise ValueError("No target location on map")
        elif len(current_target_locs) > 1:
            raise ValueError("Multiple target locations on map")
        
        return current_target_locs[0]

    # Maze Visualization Functions

    def draw_maze(self, values=None, title=""):
        """ 
        Creates a figure if it does not exist yet
        Otherwise updates the figure with the current position
        the values within the grid default to self.rewards, but can be provided as an input       

        :param: values, matrix the same size as self.rewards to be displayed on the grid 
        :param: title, String, Title of maze figure
        :return: None
        """
        if values is None:
            values = self.rewards

        # If Figure has not been made yet, make the figure
        if self.figure is None:
            self.figure, self.ax, self.current_box, self.target_box, self.value_labels = self._create_plot(title)
            plt.show(block=False)

        # Draw current position
        self._update_plot(values)
        # Render new position on screen
        plt.pause(0.005)

    def _create_plot(self, rewards=None, title=""):
        """ 
        Generates the plot of the current maze state

        :param: title, String, title of the plot
        :return: (plt.figure, plt.ax, patches.rectangle, List of ax.text), 
                 returns the Figure, Axes, Rectangle objects, and text objects
                 to be used in future updates
        
        Notes: Based on
        https://stackoverflow.com/questions/30222747/drawing-a-grid-in-python-with-colors-corresponding-to-different-values 
        """

        # parameters for the axis
        x_axis_size, y_axis_size = self.rewards.shape
        title = title
        xlabel= "x"
        ylabel= "y"
        xticklabels = range(0, x_axis_size)
        yticklabels = range(0, y_axis_size)
        fig, ax = plt.subplots()

        # set axis limits
        ax.set_xlim(xticklabels[0], xticklabels[-1]+1)
        ax.set_ylim(yticklabels[0], yticklabels[-1]+1)

        # Let the grid be aligned on the major axes
        ax.set_xticks(np.array(xticklabels), minor=False)
        ax.set_yticks(np.array(yticklabels), minor=False)

        # But offset the minor axes so that the labels appear
        # between grids
        ax.set_xticks(np.array(xticklabels)+0.5, minor=True)
        ax.set_yticks(np.array(yticklabels)+0.5, minor=True)

        # Don't show tick labels for major axes
        ax.set_xticklabels("", minor=False)
        ax.set_yticklabels("", minor=False)

        # Show tick labels for minor axis
        ax.set_xticklabels(xticklabels, minor=True)
        ax.set_yticklabels(yticklabels, minor=True)

        # set title and x/y labels
        ax.set_title(title, y=1.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Put x-axis at top of image
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        plt.grid(True, which="major", linestyle='--')
        plt.tight_layout()

        # Turn off all the ticks
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.xaxis.get_minor_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_minor_ticks():
            t.tick1On = False
            t.tick2On = False

        text_objects = []   

        for x in range(self.x_max_range):
            for y in range(self.y_max_range):
                txt = ax.text(x+0.5, y+0.5, "%.3f" % self.rewards[x,y], ha="center", va="center")
                text_objects.append(txt)

        # Invert the axis
        ax.invert_yaxis()

        # Draw walls
        self._draw_walls(ax=ax)

        # Draw current position as red rectangle
        rect_pos = self.position[0]+0.25, self.position[1]+0.25
        rect = patches.Rectangle(rect_pos, 0.5,0.5, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Draw target position as green rectangle
        target_position = self._get_target_location()
        target_origin = target_position[0]+0.15, target_position[1]+0.15
        rect_target = patches.Rectangle(target_origin, 0.7,0.7, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_target)

        return fig, ax, rect, rect_target, text_objects

    def _update_plot(self, values=None):
        """
        Updates the rendering to show the current position and the provided rewards 
        
        :param: values, np array same shape as self.rewards containing values for each grid
                        if None, defaults to self.rewards
        :return: None
        """
        if values is None:
            values = self.rewards

        rect_pos = self.position[0]+0.25, self.position[1]+0.25
        self.current_box.set_xy(rect_pos)

        target_loc = self._get_target_location()
        target_origin = target_loc[0]+0.15, target_loc[1]+0.15
        self.target_box.set_xy(target_origin)

        value_label_ind = -1
        for x in range(self.x_max_range):
            for y in range(self.y_max_range):
                value_label_ind += 1
                if type(values[x,y]) is str:
                    self.value_labels[value_label_ind].set_text(values[x,y])
                else:
                    self.value_labels[value_label_ind].set_text("%.3f" % values[x,y])

    def _draw_walls(self, ax=None):
        """ 
        Renders the Walls on the input ax 

        :param: ax, (plt.Axes), Axes object to draw walls on
        :return: None
        """
        if ax is None:
            ax = self.ax

        for x in range(self.x_max_range):
            for y in range(self.y_max_range):
                # North Wall
                if self.adjacent[x,y,0] == 0:
                    ax.plot([x, x+1], [y, y], "b-", linewidth=4)

                # East Wall
                if self.adjacent[x,y,1] == 0:
                    ax.plot([x+1, x+1], [y, y+1], "b-", linewidth=4)

                # South Wall
                if self.adjacent[x,y,2] == 0:
                    ax.plot([x, x+1], [y+1, y+1], "b-", linewidth=4)

                # West Wall
                if self.adjacent[x,y,3] == 0:
                    ax.plot([x, x], [y, y+1], "b-", linewidth=4)

    @staticmethod
    def export_image(filename):
        """ Exports the image """
        plt.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":

    # Some basic tests showing maze is working

    import os

    maze = Maze()
    maze.load_maze(os.path.join("mazes", "maze0.txt"))
    print("Testing reward...")
    print("Test reward is 0 at (0,0):", maze.get_reward() == 0)
    maze.position = (3, 3)  # should be reward 1
    print("Test reward is 1 at (3,3):", maze.get_reward() == 1)
    print("Testing move...")

    # These tests based on running is_move_valid(maze, maze_index_from_XY(maze,1,1), 4)
    # Note that since MATLAB is 1-indexed, (0,0) here corresponds to (1,1) there
    maze.position = (0, 0)
    new_position = maze.is_move_valid(0)
    print("Move from (0, 0) North is invalid:", new_position is None)
    print(new_position)

    maze.position = (0, 0)
    new_position = maze.is_move_valid(1)
    print("Move from (0, 0) East is valid:", new_position == (1, 0))
    print(new_position)

    maze.position = (0, 0)
    new_position = maze.is_move_valid(2)
    print("Move from (0, 0) South is valid:", new_position == (0, 1))
    print(new_position)

    maze.position = (0, 0)
    new_position = maze.is_move_valid(3)
    print("Move from (0, 0) West is invalid:", new_position is None)
    print(new_position)
    
    maze.draw_maze()

    original_target_loc = maze._get_target_location()

    assert np.all(original_target_loc == np.array((5,1))),\
            f"Incorrect Target Location, expected (5,1), got {original_target_loc}"

    assert maze.get_observation((5,1)), "Get Observation incorrect"

    tries = 0
    while (tries < 100) and (np.all(maze._get_target_location() == original_target_loc)):
        maze.move_target()
        tries += 1
    
    if tries == 100:
        print("Unsuccessful at moving target in 100 tries. Make sure move_target works")
    else:
        print(f"Successfully moved target in {tries} tries")
        print(f"New target location {maze._get_target_location()}")

    # Need this or the figure will close immediately due to non-blocking
    input("Press Enter to end the program and close the Figure...")

