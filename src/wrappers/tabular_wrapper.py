from gymnasium.core import ObservationWrapper

class TabularWrapper(ObservationWrapper):
  """Wrapper to convert a GridWorld environment into a tabular one."""

  def __init__(self, env, tile_size=8, get_rgb=False):
    super().__init__(env)

    self.tile_size = tile_size
    self.get_rgb = get_rgb
    self.pos_to_state = {}
    self.state_to_pos = {}
    curr_state = 0
    for y in range(self.height):
      for x in range(self.width):
        curr_cell = env.grid.get(x, y)
        grid_pos = x + y * env.width
        if (curr_cell is not None and curr_cell.type != 'goal' and
            curr_cell.type != 'lava'):
          self.pos_to_state[grid_pos] = -1
        else:
          self.pos_to_state[grid_pos] = curr_state
          self.state_to_pos[curr_state] = [x, y]
          curr_state += 1
    self.num_states = curr_state
    obs_space_dict = {'state': gym.spaces.Discrete(self.num_states)}
    if self.get_rgb and 'image' in self.observation_space.spaces:
      # Re-use the image obs spec from the env we are wrapping.
      obs_space_dict['image'] = self.observation_space['image']
    self.observation_space = gym.spaces.Dict(obs_space_dict)

  def observation(self, obs=None):
    if obs is None:
      obs = self.gen_obs()
    env = self.unwrapped
    agent_x_pos, agent_y_pos = env.agent_pos
    state_number = self.pos_to_state[agent_x_pos + agent_y_pos * env.width]

    if self.get_rgb:
      rgb_img = env.render(
          mode='rgb_array', highlight=False, tile_size=self.tile_size)

      return {'state': state_number, 'image': rgb_img}
    else:
      return {'state': state_number}