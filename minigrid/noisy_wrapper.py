from __future__ import annotations
from pydoc import render_doc

from minigrid import minigrid_env

from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium.core import ActType, ObsType
import gymnasium as gym

from gymnasium.core import ObservationWrapper, ObsType, Wrapper

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj
import pygame
import numpy as np
import math
import minigrid.core.actions


class NoisyObservation(Wrapper):
    def __init__(self, env, d=2, noise=0):
        """A wrapper that always regenerate an environment with the same set of seeds.

        Args:
            env: The environment to apply the wrapper
            seeds: A list of seed to be applied to the env
            seed_idx: Index of the initial seed in seeds
        """
        super().__init__(env)

        # Current position and direction of the agent
        self.observed_agent_dir: np.ndarray | tuple[int, int] = None
        self.observed_agent_pos: int = None
        self.d = d  # Diameter of noise
        self.noise = noise

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        self.observed_agent_pos = (-1, -1)
        pos = self.env.place_obj(None, top, size, max_tries=max_tries)
        self.observed_agent_pos = pos

        if rand_dir:
            self.observed_agent_dir = self.env._rand_int(0, 4)

        return super().place_agent(top, size, rand_dir, max_tries)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        ret = super().reset(seed=seed)
        self.observed_agent_dir = self.env.agent_dir
        self.observed_agent_pos = self.env.place_obj(
            None, self.env.agent_pos, (self.d, self.d), max_tries=20
        )
        return ret

    @property
    def observed_dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert (
            self.observed_agent_dir >= 0 and self.observed_agent_dir < 4
        ), f"Invalid agent_dir: {self.observed_agent_dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.observed_agent_dir]

    @property
    def observed_right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.observed_dir_vec
        return np.array((-dy, dx))

    @property
    def observed_front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.observed_agent_pos + self.observed_dir_vec

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step
        """
        if action in [actions]
        ret = super().step(action)
        # Get the position in front of the agent
        fwd_pos = self.observed_front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.env.actions.left:
            self.observed_agent_dir -= 1
            if self.observed_agent_dir < 0:
                self.observed_agent_dir += 4

        # Rotate right
        elif action == self.env.actions.right:
            self.observed_agent_dir = (self.observed_agent_dir + 1) % 4

        # Move forward
        elif action == self.env.actions.forward:
            print("jeje forward!!")
            if fwd_cell is None or fwd_cell.can_overlap():
                self.observed_agent_pos = tuple(fwd_pos)
                print("1")
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                print("2")
                # reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
                print("3")

        # Pick up an object
        # elif action == self.actions.pickup:
        #     if fwd_cell and fwd_cell.can_pickup():
        #         if self.carrying is None:
        #             self.carrying = fwd_cell
        #             self.carrying.cur_pos = np.array([-1, -1])
        #             self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        # elif action == self.actions.drop:
        #     if not fwd_cell and self.carrying:
        #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
        #         self.carrying.cur_pos = fwd_pos
        #         self.carrying = None

        # Toggle/activate an object
        # elif action == self.env.actions.toggle:
        #     if fwd_cell:
        #         fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.env.actions.done:
            pass

        else:
            print(f"Unknown action: {action}")
            # raise ValueError(f"Unknown action: {action}")
        return ret

    def stochastic_transition(self, action):

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.env.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.env.grid.get(*fwd_pos)

        # Rotate left
        if action == self.env.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.env.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.env.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.env.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.end.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.env.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.env.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.env.step_count >= self.env.max_steps:
            truncated = True

        if self.env.render_mode == "human":
            self.render()

        obs = self.env.gen_obs()

        return obs, reward, terminated, truncated, {}

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.env.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.env.dir_vec
        r_vec = self.env.right_vec
        top_left = (
            self.env.agent_pos
            + f_vec * (self.env.agent_view_size - 1)
            - r_vec * (self.env.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.env.width, self.env.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.env.agent_view_size):
            for vis_i in range(0, self.env.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.env.grid.render(
            tile_size,
            self.env.agent_pos,
            self.env.agent_dir,
            self.observed_agent_pos,
            self.observed_agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            # TODO
            print("ACHTUNG! agent_pov is True")
            return self.env.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        img = self.get_frame(self.env.highlight, self.env.tile_size, self.env.agent_pov)

        if self.env.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.env.render_size is None:
                self.env.render_size = img.shape[:2]
            if self.env.window is None:
                pygame.init()
                pygame.display.init()
                self.env.window = pygame.display.set_mode(
                    (self.env.screen_size, self.env.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.env.clock is None:
                self.env.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(
                bg, (self.env.screen_size, self.env.screen_size)
            )

            font_size = 22
            text = self.env.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.env.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.env.clock.tick(self.env.metadata["render_fps"])
            pygame.display.flip()

        elif self.env.render_mode == "rgb_array":
            return img
