"""
Custom Minigrid environment with configurable barriers.
Extends EmptyEnv to add barrier placement functionality.
"""

import numpy as np
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Wall


class BarrierGridEnv(EmptyEnv):
    """
    Empty grid environment with configurable barriers (walls).

    Extends the standard EmptyEnv to allow placing horizontal or vertical
    wall barriers for testing representation adaptation.
    """

    def __init__(
        self,
        size=20,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=None,
        barriers=None,
        **kwargs
    ):
        """
        Args:
            size: Grid size (creates size x size grid)
            agent_start_pos: Starting position of agent
            agent_start_dir: Starting direction (0=right, 1=down, 2=left, 3=up)
            max_steps: Maximum steps per episode
            barriers: List of barrier specifications. Each can be:
                - ('horz', x, y, length): horizontal wall starting at (x,y)
                - ('vert', x, y, length): vertical wall starting at (x,y)
                - (x, y): single wall tile at (x,y)
        """
        self.barriers = barriers or []

        super().__init__(
            size=size,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Call parent to create empty grid with boundary walls
        super()._gen_grid(width, height)

        # Remove the goal (we don't need it for SR experiments)
        self.grid.set(width - 2, height - 2, None)

        # Add barriers
        for barrier in self.barriers:
            if len(barrier) == 4:  # (type, x, y, length)
                barrier_type, x, y, length = barrier
                if barrier_type == 'horz':
                    self.grid.horz_wall(x, y, length, obj_type=Wall)
                elif barrier_type == 'vert':
                    self.grid.vert_wall(x, y, length, obj_type=Wall)
            elif len(barrier) == 2:  # (x, y) single tile
                x, y = barrier
                self.put_obj(Wall(), x, y)

    def get_transition_matrix(self):
        """
        Compute the transition matrix for this environment.
        Returns a (num_states, num_states) matrix where T[s, s'] is the
        probability of transitioning from state s to state s' under a
        uniform random policy over movement actions.

        States are indexed row-major: state = row * width + col
        (excluding boundary walls)
        """
        # Internal grid size (excluding boundary walls)
        width = self.width - 2
        height = self.height - 2
        num_states = width * height

        T = np.zeros((num_states, num_states))

        for row in range(height):
            for col in range(width):
                # Grid coordinates (accounting for boundary wall)
                x, y = col + 1, row + 1

                # State index
                state = row * width + col

                # Skip if current position has a wall
                cell = self.grid.get(x, y)
                if cell is not None and cell.type == 'wall':
                    continue

                # Check 4 neighboring cells
                neighbors = []

                # Right
                if col < width - 1:
                    neighbor_cell = self.grid.get(x + 1, y)
                    if neighbor_cell is None or neighbor_cell.type != 'wall':
                        neighbors.append(row * width + (col + 1))

                # Down
                if row < height - 1:
                    neighbor_cell = self.grid.get(x, y + 1)
                    if neighbor_cell is None or neighbor_cell.type != 'wall':
                        neighbors.append((row + 1) * width + col)

                # Left
                if col > 0:
                    neighbor_cell = self.grid.get(x - 1, y)
                    if neighbor_cell is None or neighbor_cell.type != 'wall':
                        neighbors.append(row * width + (col - 1))

                # Up
                if row > 0:
                    neighbor_cell = self.grid.get(x, y - 1)
                    if neighbor_cell is None or neighbor_cell.type != 'wall':
                        neighbors.append((row - 1) * width + col)

                # Uniform transition to all valid neighbors
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        T[state, neighbor] = 1.0 / len(neighbors)

        return T


def create_baseline_env_with_barrier():
    """
    Create the baseline environment matching test_baseline.py:
    - 20x20 internal grid (22x22 including boundary walls)
    - Horizontal barrier at row 10, cols 8-12 (internal coordinates)

    In minigrid grid coordinates (with boundary), this is:
    - row 11 (y=11), cols 9-13 (x=9 to x=13)
    """
    env = BarrierGridEnv(
        size=22,
        barriers=[('horz', 9, 11, 5)],  # horizontal wall of length 5
        agent_start_pos=(1, 1),
        render_mode=None
    )
    return env


def create_open_env(size=22):
    """
    Create an open grid environment with no barriers.

    Args:
        size: Grid size (default 22 = 20x20 internal + 2 for boundary walls)
    """
    env = BarrierGridEnv(
        size=size,
        barriers=[],
        agent_start_pos=(1, 1),
        render_mode=None
    )
    return env
