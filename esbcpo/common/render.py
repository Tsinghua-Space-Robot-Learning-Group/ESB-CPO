import os
import cv2
import imageio
import numpy as np

from PIL import Image, ImageDraw
from core.game import Game
from core.utils import arr_to_str


class DMCWrapper(Game):
    def __init__(self, env, image_based=False, cvt_string=True, save_video=False, save_path=None):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.act_space['action'].shape[0])
        self.image_based = image_based
        self.cvt_string = cvt_string
        self.save_video = save_video
        self.save_path = save_path
        self.frames = []
        self.reward_lst = [0]

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.shape[0])]

    def get_max_steps(self):
        return self.env.max_episode_steps

    def process_output(self, output):
        if self.image_based:
            observation = output['image']
        else:
            #TODO: add state based
            observation = output['observation']
            observation = np.asarray(observation, dtype="float32")

        if self.save_video:
            self.frames.append(output['image'])

        if self.cvt_string:
            observation = arr_to_str(observation)

        reward = output['reward']
        done = output['is_last']
        info = {}
        return observation, reward, done, info

    def step(self, action):
        output = self.env.step(action)
        observation, reward, done, info = self.process_output(output)

        self.reward_lst.append(reward)

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.frames = []
        self.reward_lst = [0]

        output = self.env.reset(**kwargs)
        observation, reward, done, info = self.process_output(output)

        return observation

    def close(self):
        if self.save_video:
            writer = imageio.get_writer(self.save_path)
            self.reward_lst[0] = np.sum(self.reward_lst)
            # print('reward is {}'.format(self.reward_lst))
            for frame, reward in zip(self.frames, self.reward_lst):
                frame = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame)
                draw.text((70, 70), '{}'.format(reward), fill=(255, 255, 255))
                frame = np.array(frame)
                writer.append_data(frame)
            writer.close()

        self.env.close()

class DMC:
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ["MUJOCO_GL"] = "egl"
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if domain == "manip":
            from dm_control import manipulation

            self._env = manipulation.load(task + "_vision")
        elif domain == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite

            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                pentaped_walk=2,
                pentaped_run=2,
                pentaped_escape=2,
                pentaped_fetch=2,
                biped_walk=2,
                biped_run=2,
                biped_escape=2,
                biped_fetch=2,
                triped_walk=2,
                triped_run=2,
                triped_escape=2,
                triped_fetch=2,
                hexaped_walk=2,
                hexaped_run=2,
                hexaped_escape=2,
                hexaped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []

        self._observation_space = _spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        spaces["obaservation"] = self.observation_space

        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    @property
    def observation_space(self):
        return self._observation_space
    

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs["observation"] = self._flatten_obs(time_step.observation)
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        
        return obs

    def _flatten_obs(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0)

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs["observation"] = self._flatten_obs(time_step.observation)        
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )

        return obs

    def close(self):
        return self._env.close()

