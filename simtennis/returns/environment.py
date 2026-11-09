"""Force-controlled single-return task. Physics owns motion and ball outcomes."""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
import sys

import gymnasium as gym
import numpy as np

from .model import ROOT, model_xml


def native_module():
    folder = ROOT / '.scratch/rl-build/python'
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        module = importlib.import_module('_tennis_native')
    except ImportError as error:
        raise RuntimeError('Build the native connection with tools/build_return_policy.py') from error
    hashes = ''.join(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ('native/simulation.cpp', 'native/simulation.hpp', 'native/python.cpp'))
    if getattr(module, 'build_id', None) != hashlib.sha256(hashes.encode()).hexdigest():
        raise RuntimeError('Native connection is stale. Rebuild with tools/build_return_policy.py')
    return module


@dataclass(frozen=True)
class TaskConfig:
    control_seconds: float = .02
    max_seconds: float = 6
    force_limit: float = 700
    torque_limit: float = 40
    face_angle: float = .30
    difficulty: float = 0
    shaping_discount: float = .99
    distribution: str = 'court'
    shaping_scale: float = .5

    def __post_init__(self):
        if not all(math.isfinite(v) for k, v in asdict(self).items() if k != 'distribution'):
            raise ValueError('Task configuration must be finite')
        if self.distribution not in ('court', 'narrow', 'curriculum'):
            raise ValueError('Use court, curriculum or the historical narrow profile')
        if not (.000125 <= self.control_seconds <= .25 and self.max_seconds >= self.control_seconds
                and 0 < self.force_limit <= 700 and 0 < self.torque_limit <= 40
                and abs(self.face_angle) <= .5 and 0 <= self.difficulty <= 1
                and 0 < self.shaping_discount <= 1 and 0 <= self.shaping_scale <= 1):
            raise ValueError('Invalid task limits or timing')


class ReturnOutcome:
    """One impact cluster, then first landing. Repeat records cannot add hits.

    Separated racket contacts within 20 ms and a total 50 ms span form one impact
    cluster; a later contact is a re-hit. Sustained contact over 50 ms is a carry proxy.
    These are explicit mechanical approximations, not intent recognition.
    """
    radius = .033
    half_width = 4.115
    half_length = 11.885

    def __init__(self, timestep):
        self.timestep = timestep
        self.outcome = 'live'
        self.hit = False
        self.bounces = 0
        self.seen = set()
        self.last_racket_tick = -1
        self.first_racket_tick = -1

    def update(self, contacts):
        for event in sorted(contacts, key=lambda e: (e['first_tick'], e['surface'])):
            if self.outcome != 'live':
                break
            key = (event['surface'], event['player'], event['first_tick'])
            if event['surface'] == 'racket' and event['player'] == 0:
                if event['duration'] > .05:
                    self.outcome = 'sustained_contact'
                    break
                span = (event['last_tick']-self.first_racket_tick)*self.timestep
                if self.hit and span > .05+1e-9:
                    self.outcome = 'repeated_hit'
                    break
                if key not in self.seen:
                    gap = (event['first_tick'] - self.last_racket_tick)*self.timestep
                    if self.hit and gap > .02+1e-9:
                        self.outcome = 'repeated_hit'
                        break
                    if not self.hit:
                        self.first_racket_tick = event['first_tick']
                    self.hit = True
                self.last_racket_tick = max(self.last_racket_tick, event['last_tick'])
            elif key not in self.seen and event['surface'] == 'court':
                x, _, z = event['position']
                if self.hit:
                    inside = abs(x) <= self.half_width+self.radius and -self.half_length-self.radius <= z <= self.radius
                    self.outcome = 'success' if inside else 'out'
                else:
                    self.bounces += 1
                    if self.bounces >= 2:
                        self.outcome = 'miss'
            self.seen.add(key)
        return self.outcome


def stringbed_position(state):
    pose = state['players'][0]['bodies']['racket']
    w, x, y, z = pose['orientation']
    # Stringbed centre is 0.43 m along the racket's local X axis in the source model.
    return np.asarray(pose['position']) + .43*np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])


class ReturnEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = config or TaskConfig(**kwargs)
        self.xml = model_xml(self.config.face_angle)
        self.model_hash = hashlib.sha256(self.xml.encode()).hexdigest()
        self._sim = native_module().Simulation(self.xml)
        self.state = self._sim.snapshot()
        ratio = self.config.control_seconds/self.state['timestep']
        if abs(ratio-round(ratio)) > 1e-8:
            raise ValueError('Control interval must contain an integer number of physical ticks')
        self.ticks = round(ratio)
        self.action_space = gym.spaces.Box(-1, 1, (3,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (27,), np.float32)
        self.curriculum_difficulty = self.config.difficulty
        self._done = True

    def set_difficulty(self, difficulty):
        if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
            raise ValueError('Feed difficulty must be in [0, 1]')
        self.curriculum_difficulty = difficulty

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sim.reset()
        d = self.curriculum_difficulty
        self.episode_difficulty = d
        rng = self.np_random
        if self.config.distribution in ('court', 'curriculum'):
            if self.config.distribution == 'court':
                d = 1.
            # Independent samples throughout; d=1 is exactly the court profile.
            # The easier end narrows opponent feeds around a late post-bounce
            # interception. It does not condition the feed on the sampled car.
            xcentre = .8*(1-d)
            xspread = .1+2.9*d
            carspread = .25+2.75*d
            carcentre = 9.5-d
            depthspread = .25+2.25*d
            car_start = [float(rng.uniform(-carspread, carspread)),
                         float(rng.uniform(carcentre-depthspread, carcentre+depthspread))]
            origin = np.array([rng.uniform(xcentre-xspread, xcentre+xspread), 1.2,
                               rng.uniform(-8-(.2+1.8*d), -8+(.2+1.8*d))])
            aim = np.array([rng.uniform(xcentre-xspread, xcentre+xspread), .033,
                            rng.uniform(8.5-d-(.2+2.3*d), 8.5-d+(.2+2.3*d))])
            flight = rng.uniform(1.275-(.025+.15*d), 1.275+(.025+.15*d))
            velocity = (aim-origin)/flight + np.array([0, .5*9.81*flight, 0])
            # This is a nominal gravity-only aim. Native drag/contact dynamics
            # determine the actual trajectory; the feed never targets the car.
            feed = {'position': origin.tolist(), 'velocity': velocity.tolist(), 'spin': [0., 0., 0.]}
        else:
            car_start = [0., 8.]
            feed = {
                'position': [float(rng.uniform(.7-.35*d, .9+.35*d)),
                             float(rng.uniform(1.02, 1.08)),
                             float(rng.uniform(6.9-.5*d, 7.1+.5*d))],
                'velocity': [0., 0., float(rng.uniform(7.5-d, 8.5+d))],
                'spin': [0., 0., 0.],
            }
        if options and 'car_start' in options:
            car_start = options['car_start']
        if options and 'feed' in options:
            feed = options['feed']
        self._sim.set_player_start(car_start)
        self._sim.feed_ball(feed['position'], feed['velocity'], feed.get('spin', [0, 0, 0]))
        self.feed = feed
        self.car_start = list(car_start)
        self.state = self._sim.snapshot()
        self.rules = ReturnOutcome(self.state['timestep'])
        self._done = False
        self._potential = self._distance_potential()
        return self._observation(), self._info()

    def _distance_potential(self):
        if self.rules.hit:
            return 0.
        ball = np.asarray(self.state['bodies']['ball']['position'])
        return -float(np.linalg.norm(ball-stringbed_position(self.state)))

    def _observation(self):
        p = self.state['players'][0]
        values = (list(np.asarray(self.state['bodies']['ball']['position'])/[5, 5, 15])
                  + list(np.asarray(self.state['ball_velocity'])/30)
                  + list(np.asarray(self.state['ball_spin'])/200)
                  + [p['position'][0]/5, p['position'][2]/15, p['velocity'][0]/10, p['velocity'][2]/10]
                  + [p['joint_angles'][0]/math.pi, p['joint_velocities'][0]/30]
                  + list(stringbed_position(self.state)/[5, 5, 15])
                  + list(np.asarray(p['racket_velocity'])/30)
                  + [float(self.rules.hit), self.rules.bounces/2, self.state['time']/self.config.max_seconds]
                  + [(self.state['tick']-self.rules.last_racket_tick)*self.state['timestep']/self.config.max_seconds
                     if self.rules.hit else -1.,
                     max((e['duration']/.05 for e in self.state['contacts']
                          if e['surface'] == 'racket' and not e['complete']), default=0.),
                     (self.state['tick']-self.rules.first_racket_tick)*self.state['timestep']/self.config.max_seconds
                     if self.rules.hit else -1.])
        return np.asarray(values, dtype=np.float32)

    def _info(self):
        return {'outcome': self.rules.outcome, 'is_success': self.rules.outcome == 'success',
                'hit': self.rules.hit, 'incoming_bounces': self.rules.bounces,
                'time': self.state['time'], 'warnings': self.state['warnings'],
                'feed': self.feed, 'car_start': self.car_start,
                'feed_difficulty': self.episode_difficulty, 'model_hash': self.model_hash}

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset before advancing a completed episode')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Action must contain three finite values in [-1, 1]')
        # Bound resultant planar force as well as individual action components.
        force = action[:2]*self.config.force_limit/max(1., float(np.linalg.norm(action[:2])))
        if not self._sim.set_effort(self.state['time'], [action[2]*self.config.torque_limit], force):
            raise RuntimeError('Native simulation rejected a validated effort command')
        was_hit = self.rules.hit
        self._sim.step(self.ticks)
        self.state = self._sim.snapshot()
        if self.state['faulted'] or self.state['warnings']:
            raise RuntimeError('Numerical failure in return simulation')
        self.rules.update(self.state['contacts'])
        ball = self.state['bodies']['ball']['position']
        if self.rules.outcome == 'live' and (abs(ball[0]) > 20 or abs(ball[2]) > 25 or ball[1] > 20):
            self.rules.outcome = 'escaped'
        terminated = self.rules.outcome != 'live'
        truncated = not terminated and self.state['time']+1e-9 >= self.config.max_seconds
        potential = 0. if terminated else self._distance_potential()
        reward = self.config.shaping_scale*(self.config.shaping_discount*potential-self._potential) - .001*float(np.mean(action*action))
        self._potential = potential
        if self.rules.hit and not was_hit:
            reward += 1.
        if terminated:
            reward += 10. if self.rules.outcome == 'success' else -2.
        if truncated:
            self.rules.outcome = 'timeout'
        self._done = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info()

    def close(self):
        self._sim = None
