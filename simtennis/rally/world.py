"""Continuous two-car physics. Set orchestration may reset only between points."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import math

import numpy as np

from simtennis.returns.environment import native_module, stringbed_position
from .model import model_xml
from .rules import PointRules


def rotation(player):
    return np.array([1., 1., 1.]) if player == 0 else np.array([-1., 1., -1.])


@dataclass
class Turn:
    started: float = 0.
    first: int = -1
    last: int = -1
    bounces: int = 0


@dataclass(frozen=True)
class WorldCheckpoint:
    """In-memory native integration state and the rules that interpret it."""
    model_hash: str
    native: object
    history: dict


class RallyWorld:
    control_seconds = .02
    force_limit = 700.
    torque_limit = 40.

    def __init__(self):
        self.xml = model_xml()
        self.model_hash = hashlib.sha256(self.xml.encode()).hexdigest()
        self._sim = native_module().Simulation(self.xml)
        self.state = self._sim.snapshot()
        self.ticks = round(self.control_seconds / self.state['timestep'])
        self.rules = PointRules(self.state['timestep'])
        self.turns = [Turn(), Turn()]
        self._seen = {}
        self.events = []
        self.setup = {}
        self.serve_contact_tick = -1

    def capture(self):
        """Capture a complete branch point; no poses or contacts are reconstructed."""
        names = ('rules', 'turns', '_seen', 'events', 'setup', 'serve_contact_tick')
        return WorldCheckpoint(self.model_hash, self._sim.capture(),
                               deepcopy({name: getattr(self, name) for name in names}))

    def restore(self, checkpoint):
        """Restore the same physical model and a private copy of its rule history."""
        if not isinstance(checkpoint, WorldCheckpoint) or checkpoint.model_hash != self.model_hash:
            raise ValueError('Checkpoint must belong to the same rally model')
        self._sim.restore(checkpoint.native)
        for name, value in deepcopy(checkpoint.history).items():
            setattr(self, name, value)
        self.state = self._sim.snapshot()

    def _reset(self):
        self._sim.reset()
        self.turns = [Turn(), Turn()]
        self._seen = {}
        self.events = []
        self.serve_contact_tick = -1

    def reset_feed(self, seed, receiver=0, difficulty=1.):
        """Declared training launcher; never used to start a scored set point."""
        if receiver not in (0, 1) or not 0 <= difficulty <= 1:
            raise ValueError('Invalid receiver or curriculum difficulty')
        self._reset()
        rng = np.random.default_rng(seed)
        d = difficulty
        carspread, carcentre, depthspread = .25+2.75*d, 9.5-d, .25+2.25*d
        start = [float(rng.uniform(-carspread, carspread)),
                 float(rng.uniform(carcentre-depthspread, carcentre+depthspread))]
        centre, spread = .8*(1-d), .1+2.9*d
        origin = np.array([rng.uniform(centre-spread, centre+spread), 1.2,
                           rng.uniform(-8-(.2+1.8*d), -8+(.2+1.8*d))])
        aim = np.array([rng.uniform(centre-spread, centre+spread), .033,
                        rng.uniform(8.5-d-(.2+2.3*d), 8.5-d+(.2+2.3*d))])
        flight = rng.uniform(1.275-(.025+.15*d), 1.275+(.025+.15*d))
        velocity = (aim-origin)/flight + np.array([0, .5*9.81*flight, 0])
        r = rotation(receiver)
        self._sim.set_player_start(np.asarray(start)*r[::2], player=receiver)
        self._sim.set_player_start([0, -8*r[2]], player=1-receiver)
        self._sim.feed_ball(origin*r, velocity*r, [0, 0, 0])
        self.state = self._sim.snapshot()
        self.rules.begin_rally(last_hitter=1-receiver)
        self.setup = {'kind': 'feed', 'seed': int(seed), 'receiver': receiver,
                      'difficulty': difficulty, 'car_start': start,
                      'feed': {'position': origin.tolist(), 'velocity': velocity.tolist(), 'spin': [0., 0., 0.]}}
        return self.observation(receiver)

    def stringbed(self, player):
        return stringbed_position({'players': [self.state['players'][player]]})

    def observation(self, player):
        """Old 27 values + opponent planar state + serve constraint + lift state."""
        r = rotation(player)
        p, opponent = self.state['players'][player], self.state['players'][1-player]
        turn = self.turns[player]
        hit = turn.first >= 0
        values = (list(np.asarray(self.state['bodies']['ball']['position'])*r/[5, 5, 15])
                  + list(np.asarray(self.state['ball_velocity'])*r/30)
                  + list(np.asarray(self.state['ball_spin'])*r/200)
                  + [p['position'][0]*r[0]/5, p['position'][2]*r[2]/15,
                     p['velocity'][0]*r[0]/10, p['velocity'][2]*r[2]/10]
                  + [p['joint_angles'][0]/math.pi, p['joint_velocities'][0]/30]
                  + list(self.stringbed(player)*r/[5, 5, 15])
                  + list(np.asarray(p['racket_velocity'])*r/30)
                  + [float(hit), turn.bounces/2, (self.state['time']-turn.started)/6]
                  + [(self.state['tick']-turn.last)*self.state['timestep']/6 if hit else -1.,
                     max((e['duration']/.05 for e in self.state['contacts']
                          if e['surface'] == 'racket' and e['player'] == player and not e['complete']), default=0.),
                     (self.state['tick']-turn.first)*self.state['timestep']/6 if hit else -1.]
                  + [opponent['position'][0]*r[0]/5, opponent['position'][2]*r[2]/15,
                     opponent['velocity'][0]*r[0]/10, opponent['velocity'][2]*r[2]/10,
                     float(self.rules.serving and self.rules.last_hitter == 1-player),
                     p['joint_angles'][1]/math.pi, p['joint_velocities'][1]/30])
        return np.asarray(values, dtype=np.float32)

    def incoming(self, player):
        return self.rules.last_hitter == 1-player

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (2, 4) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError('Both players require four finite efforts in [-1,1]')
        if self.rules.result != 'live':
            raise RuntimeError('Reset before advancing a resolved point')
        for player, action in enumerate(actions):
            force = action[:2]*self.force_limit/max(1., float(np.linalg.norm(action[:2])))
            force *= rotation(player)[::2]
            swing, lift = self.state['players'][player]['joint_angles']
            # Static gravity torque of the .4 kg shaft at .225 m and .32 kg
            # racket at .75 m, derived from this model. This compensates weight,
            # not inertia/contact; the learner still supplies lift effort.
            gravity = 9.81*(.4*.225+.32*.75)*math.cos(swing)*math.cos(lift)
            torques = [action[2]*self.torque_limit,
                       float(np.clip(action[3]*self.torque_limit+gravity, -self.torque_limit, self.torque_limit))]
            if not self._sim.set_effort(self.state['time'], torques, force, player=player):
                raise RuntimeError('Native simulation rejected a validated effort')
        self._sim.step(self.ticks)
        self.state = self._sim.snapshot()
        if self.state['warnings'] or self.state['faulted']:
            raise RuntimeError('Numerical failure in rally simulation')
        # Process once at contact onset, then extend impact durations. Contacts
        # belong to the native engine; this layer changes history, never physics.
        for event in sorted(self.state['contacts'], key=lambda e: (e['first_tick'], e['surface'])):
            key = (event['surface'], event['player'], event['first_tick'])
            before = self.rules.last_hitter
            before_hits = list(self.rules.racket_hits)
            self.rules.update([event])
            if event['surface'] == 'racket' and event['player'] in (0, 1):
                player = event['player']
                if self.rules.racket_hits[player] > before_hits[player]:
                    if before == -1 and self.setup.get('kind') == 'serve':
                        self.serve_contact_tick = event['first_tick']
                    self.turns[1-player] = Turn(started=event['first_tick']*self.state['timestep'])
                    self.turns[player].first = event['first_tick']
                if self.turns[player].first >= 0:
                    self.turns[player].last = max(self.turns[player].last, event['last_tick'])
            elif event['surface'] == 'court' and key not in self._seen and before in (0, 1):
                self.turns[1-before].bounces += 1
            if key not in self._seen:
                self.events.append(dict(event))
                self._seen[key] = len(self.events)-1
            else:
                self.events[self._seen[key]] = dict(event)
        return self.state


def recovery_action(world, player, target=(0., 8.), angle=-.7):
    """Stationary ready posture reached through bounded forces, never teleporting."""
    p = world.state['players'][player]
    r = rotation(player)
    position = np.asarray(p['position'])[::2]*r[::2]
    velocity = np.asarray(p['velocity'])[::2]*r[::2]
    # Damped second-order response; mass is 20 kg, limits match native actuators.
    force = 100*(np.asarray(target)-position)-90*velocity
    torque = 20*(angle-p['joint_angles'][0])-3*p['joint_velocities'][0]
    return np.clip(np.r_[force/world.force_limit, torque/world.torque_limit, hold_lift(world, player)], -1, 1)


def hold_lift(world, player):
    p = world.state['players'][player]
    # Swing changes lift-axis inertia dramatically as the beam aligns with Z.
    # A critically damped 12 rad/s hold remains sampled stably at 50 Hz. Inertia
    # coefficients come from the shaft/racket/armature in the reduced model.
    swing, lift = p['joint_angles']
    inertia = .224*math.cos(swing)**2+.0051*math.sin(swing)**2
    torque = -inertia*(144*lift+24*p['joint_velocities'][1])
    return float(np.clip(torque/world.torque_limit, -1, 1))


def player_action(world, player, policy):
    if world.setup.get('kind') == 'serve' and player == world.rules.server:
        from .serve import server_action, serve_active
        if serve_active(world):
            return server_action(world)
    if world.incoming(player):
        action = np.asarray(policy(world.observation(player)))
        if action.shape == (3,):
            # Historical policy has no lift output: hold its original posture
            # through the same bounded actuator, not a locked joint.
            return np.r_[action, hold_lift(world, player)]
        return action
    return recovery_action(world, player)
