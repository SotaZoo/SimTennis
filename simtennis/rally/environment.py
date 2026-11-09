"""Whole-point Gymnasium task against an opponent frozen for each episode."""
from dataclasses import asdict, dataclass
import math

import gymnasium as gym
import numpy as np

from .world import RallyWorld, player_action


@dataclass(frozen=True)
class RallyConfig:
    feed_probability: float = 0.
    difficulty: float = 1.
    max_seconds: float = 30.
    discount: float = .999
    shaping_scale: float = .05
    contact_reward: float = .2
    return_reward: float = 1.
    win_reward: float = 10.
    prelude_seconds: float = 0.
    receive_only: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError('Configuration must be finite')
        if not (0 <= self.feed_probability <= 1 and 0 <= self.difficulty <= 1
                and .02 <= self.max_seconds <= 120 and 0 < self.discount <= 1
                and 0 <= self.shaping_scale <= 1 and 0 <= self.contact_reward <= 10
                and 0 <= self.return_reward <= 10 and 0 < self.win_reward <= 100
                and 0 <= self.prelude_seconds <= 1.5
                and abs(self.prelude_seconds/.02-round(self.prelude_seconds/.02)) < 1e-8):
            raise ValueError('Invalid task configuration')


class RallyEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, opponents, config=None):
        super().__init__()
        if not opponents:
            raise ValueError('Supply at least one frozen opponent')
        self.opponents = opponents
        self.config = config or RallyConfig()
        self.world = RallyWorld()
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (34,), np.float32)
        self._done = True
        self.physics_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.episode_seed = int(seed if seed is not None else rng.integers(1_000_000, 2_000_000_000))
        options = options or {}
        self.learner = int(options.get('learner', rng.integers(2)))
        self.opponent_index = int(options.get('opponent', rng.integers(len(self.opponents))))
        self.opponent = self.opponents[self.opponent_index]
        self.mode = options.get('mode', 'feed' if rng.random() < self.config.feed_probability else 'serve')
        self.server = int(options.get('server', rng.integers(2)))
        if self.config.receive_only and 'learner' not in options:
            self.learner = 1-self.server
        self.deuce = bool(options.get('deuce', rng.integers(2)))
        self.attempt = 1
        self.lets = 0
        self.elapsed = 0.
        self.hits = np.zeros(2, dtype=int)
        self.returns = np.zeros(2, dtype=int)
        self.applied_steps = 0
        self.action_clip_counts = np.zeros(4, dtype=int)
        if self.mode == 'feed':
            self.world.reset_feed(self.episode_seed, self.learner, self.config.difficulty)
        elif self.mode == 'serve':
            self._serve()
        else:
            raise ValueError('Use feed or serve mode')
        self._potential = self._distance_potential()
        self._done = False
        for _ in range(round(self.config.prelude_seconds/self.world.control_seconds)):
            _, _, done, truncated, info = self.step(np.zeros(4))
            if done or truncated or self.returns.any() or self.hits[self.learner] > int(self.server == self.learner and self.mode == 'serve'):
                raise RuntimeError('Prelude resolved or returned a ball; qualify a shorter interval')
        self._potential = self._distance_potential()
        return self.world.observation(self.learner), self._info()

    def _serve(self):
        from .serve import setup_serve
        setup_serve(self.world, self.episode_seed+self.lets*1009, self.server, self.deuce, self.attempt,
                    difficulty=self.config.difficulty)

    def _distance_potential(self):
        if not self.world.incoming(self.learner):
            return 0.
        return -float(np.linalg.norm(np.asarray(self.world.state['bodies']['ball']['position'])
                                     - self.world.stringbed(self.learner)))

    def _info(self, winner=-1, outcome='live'):
        return {'winner': winner, 'won': winner == self.learner, 'outcome': outcome,
                'learner': self.learner, 'opponent': self.opponent_index, 'seed': self.episode_seed,
                'mode': self.mode, 'server': self.server, 'deuce': self.deuce,
                'time': self.elapsed, 'hits': self.hits.tolist(), 'returns': self.returns.tolist(),
                'learner_returns': int(self.returns[self.learner]),
                'opponent_returns': int(self.returns[1-self.learner]),
                'applied_steps': self.applied_steps,
                'attempt': self.attempt, 'lets': self.lets,
                'action_clip_fraction': (self.action_clip_counts/max(1, self.applied_steps)).tolist()}

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset a completed point')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Expected four finite effort values in [-1,1]')
        w = self.world
        action_applied = w.incoming(self.learner)
        actions = [None, None]
        for player in (0, 1):
            if player == self.learner:
                if action_applied:
                    self.applied_steps += 1
                    self.action_clip_counts += np.abs(action) > .95
                actions[player] = player_action(w, player, lambda obs: action)
            else:
                actions[player] = player_action(w, player, self.opponent)
        scripted_service = self.mode == 'serve' and w.rules.last_hitter == -1
        before_hits = np.asarray(w.rules.racket_hits)
        before_returns = np.asarray(w.rules.legal_returns)
        w.step(actions)
        self.last_actions = np.asarray(actions).tolist()
        self.physics_steps += 1
        self.elapsed += w.control_seconds
        if self.mode == 'serve':
            from .serve import update_serve
            update_serve(w)
        new_hits = np.asarray(w.rules.racket_hits)-before_hits
        new_returns = np.asarray(w.rules.legal_returns)-before_returns
        self.hits += new_hits
        self.returns += new_returns
        reward = (self.config.contact_reward*new_hits[self.learner]*(not scripted_service)
                  + self.config.return_reward*new_returns[self.learner])
        winner, outcome = w.rules.winner, w.rules.reason or 'live'
        terminated = w.rules.result == 'point'
        if w.rules.result == 'fault':
            if self.attempt == 2:
                terminated, winner, outcome = True, 1-self.server, 'double_fault'
            else:
                self.attempt = 2
                self._serve()
        elif w.rules.result == 'let':
            self.lets += 1
            self._serve()
        potential = 0. if terminated else self._distance_potential()
        reward += self.config.shaping_scale*(self.config.discount*potential-self._potential)
        reward -= .001*float(np.mean(action*action)) if action_applied else 0.
        self._potential = potential
        if terminated:
            reward += self.config.win_reward*(1 if winner == self.learner else -1)
        truncated = not terminated and self.elapsed+1e-9 >= self.config.max_seconds
        if truncated:
            outcome = 'timeout'
        self._done = terminated or truncated
        return w.observation(self.learner), float(reward), terminated, truncated, self._info(winner, outcome)

    def close(self):
        self.world = None
