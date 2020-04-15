from abc import ABC, abstractmethod
import numpy as np
import pyrobotdesign as rd

class ForwardSpeedTask(ABC):
  def __init__(self, time_step=1.0/240, discount_factor=0.99, interval=16,
               horizon=16, episode_len=128):
    self.time_step = time_step
    self.discount_factor = discount_factor
    self.interval = interval
    self.horizon = horizon
    self.episode_len = episode_len

    self.objective_fn = rd.DotProductObjective()
    self.objective_fn.base_dir_weight = np.array([-2.0, 0.0, 0.0])
    self.objective_fn.base_up_weight = np.array([0.0, 2.0, 0.0])
    self.objective_fn.base_vel_weight = np.array([2.0, 0.0, 0.0])

    # Maximum reasonable result (designs achieving higher results are rejected)
    self.result_bound = 10.0

  def get_objective_fn(self):
    return self.objective_fn

  @abstractmethod
  def add_terrain(self, sim):
    pass

class FlatTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over flat
  terrain.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [40.0, 1.0, 10.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class RidgedTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over ridged
  terrain.
  """

  def __init__(self, seed=0, **kwargs):
    super().__init__(**kwargs)
    self.seed = seed

    self.floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [20.0, 1.0, 10.0])
    self.bump = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [0.1, 0.2, 10.0])

  def add_terrain(self, sim):
    rng = np.random.RandomState(self.seed)
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    for i in range(20):
      sim.add_prop(self.bump,
                   [rng.normal(0.5, 0.1) + i, -0.2 + 0.02 * (i + 1), 0.0],
                   rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class GapTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over
  terrain with several large gaps.
  """

  def __init__(self, x_min=-20.0, x_max=20.0, seed=0, **kwargs):
    super().__init__(**kwargs)
    self.seed = seed

    rng = np.random.RandomState(self.seed)
    gap_centers = np.arange(0.5, x_max, 1.0)
    gap_centers += rng.normal(0.0, 0.1, size=gap_centers.shape)
    gap_widths = np.linspace(0.1, 0.5, num=len(gap_centers))
    platform_x_min = np.concatenate(([x_min], gap_centers + 0.5 * gap_widths))
    platform_x_max = np.concatenate((gap_centers - 0.5 * gap_widths, [x_max]))
    platform_x = 0.5 * (platform_x_min + platform_x_max)
    platform_half_widths = 0.5 * (platform_x_max - platform_x_min)

    self.platform_x = platform_x
    self.platforms = [
        rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [half_width, 1.0, 10.0]) for
        half_width in platform_half_widths]
    self.floor_x = 0.5 * (x_min + x_max)
    self.floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.5,
                         [0.5 * (x_max - x_min), 1.0, 10.0])

  def add_terrain(self, sim):
    rng = np.random.RandomState(self.seed)
    for i, (x, platform) in enumerate(zip(self.platform_x, self.platforms)):
      sim.add_prop(platform, [x, -1.0 + rng.normal(0.0, 0.01 * i), 0.0],
                   rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    sim.add_prop(self.floor, [self.floor_x, -2.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class StairsTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible on stairs.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [10.0, 1.0, 10.0])
    self.step = rd.Prop(rd.PropShape.BOX, 0.0, 0.5, [0.25, 1.0, 10.0])

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
    for i in range(18):
      sim.add_prop(self.step, [1.25 + 0.5 * i, -1.0 + 0.15 * (i + 1), 0.0],
                   rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class FrozenLakeTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible on a flat,
  low-friction surface.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.floor = rd.Prop(rd.PropShape.BOX, 0.0, 0.05, [20.0, 1.0, 10.0])
    self.floor.color = [0.8, 0.9, 1.0]

  def add_terrain(self, sim):
    sim.add_prop(self.floor, [0.0, -1.0, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))

class HillTerrainTask(ForwardSpeedTask):
  """
  Task where the objective is to move forward as quickly as possible over
  terrain with hills.
  """

  def __init__(self, seed=0, **kwargs):
    super().__init__(**kwargs)
    self.seed = seed

    self.rng = np.random.RandomState(self.seed)
    y = np.clip(self.rng.normal(0.5, 0.125, size=(97, 33)), 0.0, 1.0)
    self.heightfield = rd.HeightfieldProp(0.5, [30.0, 0.25, 10.0], y)

  def add_terrain(self, sim):
    sim.add_prop(self.heightfield, [20.0, -0.25, 0.0],
                 rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
