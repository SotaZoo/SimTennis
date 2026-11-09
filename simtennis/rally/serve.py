"""Qualified physical underarm serves; this controller is scripted, not learned.

The toss initializes a free ball once between attempts. Thereafter, only bounded
car forces, swing torque and lift torque act. The low-bouncing recipes preserve
ordinary legal serves; the receiver must position its racket in height as well
as across the court. Qualification concerns this model, not real equipment.
"""
import math

import numpy as np

from .world import hold_lift, recovery_action, rotation


# Native-search recipes: angle places the toss relative to the racket pivot.
# Delay is tested against the 50 Hz control clock; actual onset is .46 s.
RECIPES = {
        True: {'angle': .20, 'torque': 35.5, 'delay': .45, 'prepare': -.7},
        False: {'angle': -.173, 'torque': 37., 'delay': .45, 'prepare': -1.},
}


def setup_serve(world, seed, server=0, deuce=True, attempt=1, difficulty=1.):
    """Reset between attempts; independently sample receiver and small serve variation."""
    if (server not in (0, 1) or attempt not in (1, 2) or int(seed) != seed
            or seed < 0 or not 0 <= difficulty <= 1):
        raise ValueError('A serve needs a nonnegative seed, player 0/1, and attempt 1/2')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), attempt]))
    base = RECIPES[bool(deuce)]
    angle = base['angle'] + rng.uniform(-.003, .003)
    torque = base['torque'] + rng.uniform(-.03, .03)
    side = 1 if deuce else -1
    start = np.array([side + rng.uniform(-.1, .1), 12.5 + rng.uniform(-.05, .05)])
    toss = np.array([start[0] + .88*math.cos(angle), 1.2 + rng.uniform(-.0005, .0005),
                     start[1] - .88*math.sin(angle)])
    # An easier receiver stance broadens to the original independent start
    # distribution. The serve itself is unchanged and no incoming ball is filtered.
    d = difficulty
    centre, spread = side*2.4*(1-d), .2+1.8*d
    zcentre, zspread = 8.9-.4*d, .2+1.3*d
    receiver_start = np.array([rng.uniform(centre-spread, centre+spread),
                               rng.uniform(zcentre-zspread, zcentre+zspread)])
    r = rotation(server)
    world._reset()
    world._sim.set_player_start(start*r[::2], server)
    world._sim.set_player_start(receiver_start*rotation(1-server)[::2], 1-server)
    world._sim.feed_ball(toss*r, [0., 2.5, 0.], [0., 0., 0.])
    world.state = world._sim.snapshot()
    world.rules.begin(server, bool(deuce))
    world.setup = {
        'kind': 'serve', 'seed': int(seed), 'server': server, 'deuce': bool(deuce), 'difficulty': difficulty,
        'attempt': attempt, 'angle': float(angle), 'torque': float(torque),
        'delay': base['delay'], 'prepare': base['prepare'], 'server_start': start.tolist(),
        'receiver_start': receiver_start.tolist(),
        'toss': {'position': toss.tolist(), 'velocity': [0., 2.5, 0.], 'spin': [0., 0., 0.]},
    }
    return world.observation(1-server)


def serve_active(world):
    """Keep the actual impact's braking phase before handing control to recovery."""
    if world.setup.get('kind') != 'serve' or world.rules.result != 'live':
        return False
    server = world.setup['server']
    first = world.serve_contact_tick
    return first < 0 or world.state['time'] - first*world.state['timestep'] < .06


def server_action(world):
    """Return canonical normalized efforts without changing state or predicting contact."""
    if world.setup.get('kind') != 'serve':
        raise ValueError('Set up a serve before requesting its action')
    server = world.setup['server']
    if not serve_active(world):
        return recovery_action(world, server)
    p = world.state['players'][server]
    r = rotation(server)[::2]
    position = np.asarray(p['position'])[::2]*r
    velocity = np.asarray(p['velocity'])[::2]*r
    force = 80*(np.asarray(world.setup['server_start'])-position)-50*velocity
    if world.serve_contact_tick >= 0:
        torque = -12*p['joint_velocities'][0]
    elif world.state['time'] >= world.setup['delay']:
        torque = world.setup['torque']
    else:
        torque = 180*(world.setup['prepare']-p['joint_angles'][0])-12*p['joint_velocities'][0]
    # World.step adds the model-derived gravity bias for every controller.
    return np.clip(np.r_[force/world.force_limit,
                         torque/world.torque_limit, hold_lift(world, server)], -1., 1.)


def update_serve(world):
    """Declare an observable stance fault or a scripted swing that misses its toss."""
    if (world.setup.get('kind') != 'serve' or world.rules.result != 'live'
            or world.rules.last_hitter != -1):
        return
    server = world.setup['server']
    position = np.asarray(world.state['players'][server]['position'])*rotation(server)
    side = 1 if world.setup['deuce'] else -1
    # The car footprint substitutes for human feet: wheels extend 0.405 m
    # laterally and the chassis extends 0.48 m toward the baseline.
    legal = (position[2]-.48 > world.rules.half_length
             and position[0]*side-.405 > 0
             and position[0]*side+.405 < world.rules.half_width)
    if not legal:
        world.rules.service_fault('service_stance')
    elif world.state['time'] >= 1.2:
        world.rules.missed_serve()
