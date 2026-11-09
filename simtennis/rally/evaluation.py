"""Frozen paired point evaluations, scored sets and native action replay."""
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import time

import numpy as np

from .environment import RallyConfig
from .rules import SetScore
from .world import RallyWorld, player_action


def score_snapshot(score):
    return asdict(score) | {'server': score.server, 'point_text': [score.point_text(i) for i in (0, 1)]}


def annotate_score(episode, records, before, after):
    """A service retry cannot advance the point score shown in its replay."""
    episode.update(score_before=before, score_after=after)
    for record in records:
        record['score_after'] = before
    if records:
        records[-1]['score_after'] = after


def display_state(state):
    """Compact measured poses; full contact events are retained once per attempt."""
    return {k: state[k] for k in ('tick', 'time', 'timestep', 'bodies', 'players',
                                 'ball_velocity', 'ball_spin', 'warnings', 'faulted')}


def _rally_hits(world):
    """Accepted racket strikes excluding this attempt's actual service contact."""
    hits = list(world.rules.racket_hits)
    if world.setup.get('kind') == 'serve' and world.serve_contact_tick >= 0:
        server = world.setup['server']
        hits[server] -= min(1, hits[server])
    return hits


def play_point(world, policies, seed, *, server=0, deuce=True, mode='serve',
               receiver=0, difficulty=1., max_seconds=30., trace=False, score=None, point_index=0):
    attempt, lets, elapsed, steps = 1, 0, 0., 0
    hits, returns = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    rally_hits = np.zeros(2, dtype=int)
    attempts, clip, incoming_steps = [], np.zeros((2, 4)), np.zeros(2)
    peak_speed = np.zeros(2)
    while True:
        if mode == 'serve':
            from .serve import setup_serve, update_serve
            setup_serve(world, int(seed)+lets*1009, server, deuce, attempt, difficulty=difficulty)
        else:
            world.reset_feed(int(seed), receiver, difficulty)
        frames = [{'actions': None, 'state': display_state(world.state)}] if trace else None
        setup = dict(world.setup)
        while world.rules.result == 'live' and elapsed+1e-9 < max_seconds:
            actions = [player_action(world, p, policies[p]) for p in (0, 1)]
            for p in (0, 1):
                if world.incoming(p):
                    incoming_steps[p] += 1
                    clip[p] += np.abs(actions[p]) > .95
            world.step(actions)
            if mode == 'serve':
                update_serve(world)
            elapsed += world.control_seconds
            steps += 1
            for p in (0, 1):
                peak_speed[p] = max(peak_speed[p], float(np.linalg.norm(world.state['players'][p]['velocity'])))
            if frames is not None:
                frames.append({'actions': np.asarray(actions).tolist(), 'state': display_state(world.state)})
        rules = world.rules
        hits += rules.racket_hits
        attempt_rally_hits = _rally_hits(world)
        rally_hits += attempt_rally_hits
        returns += rules.legal_returns
        if trace:
            attempts.append({'seed': int(seed), 'setup': setup, 'point_index': point_index,
                             'attempt': attempt, 'score_before': score or {},
                             'outcome': rules.reason or 'timeout', 'winner': rules.winner,
                             'result': rules.result, 'returns': list(rules.legal_returns),
                             'hits': list(rules.racket_hits), 'rally_hits': attempt_rally_hits,
                             'frames': frames, 'events': world.events})
        if rules.result == 'point':
            winner, outcome = rules.winner, rules.reason
            break
        if rules.result == 'fault' and attempt == 2:
            winner, outcome = 1-server, 'double_fault'
            break
        if elapsed+1e-9 >= max_seconds:
            winner, outcome = -1, 'timeout'
            break
        if rules.result == 'fault':
            attempt = 2
        elif rules.result == 'let':
            lets += 1
        else:
            raise RuntimeError('Unresolved point escaped the simulation loop')
    episode = {'seed': int(seed), 'winner': winner, 'outcome': outcome,
               'server': server, 'deuce': deuce, 'mode': mode,
               'time': elapsed, 'steps': steps, 'attempt': attempt, 'lets': lets,
               'max_seconds': max_seconds,
               'hits': hits.tolist(), 'rally_hits': rally_hits.tolist(), 'returns': returns.tolist(),
               'peak_speed': peak_speed.tolist(),
               'action_clip_fraction': (clip/np.maximum(1, incoming_steps[:, None])).tolist()}
    return episode, attempts


def summarize(episodes):
    wins = sum(e['winner'] == e['learner'] for e in episodes)
    timeouts = sum(e['winner'] == -1 for e in episodes)
    count = len(episodes)
    by_opponent = {}
    for opponent in sorted({e['opponent'] for e in episodes}):
        group = [e for e in episodes if e['opponent'] == opponent]
        by_opponent[str(opponent)] = {'points': len(group),
            'wins': sum(e['winner'] == e['learner'] for e in group)}
    by_role = {}
    for role in ('serving', 'receiving'):
        group = [e for e in episodes if (e['server'] == e['learner']) == (role == 'serving')]
        by_role[role] = {'points': len(group), 'wins': sum(e['winner'] == e['learner'] for e in group)}
    lengths = [sum(e['returns']) for e in episodes]
    return {'points': count, 'wins': wins, 'win_fraction': wins/max(1, count), 'timeouts': timeouts,
            'mean_returns': float(np.mean(lengths)) if count else 0.,
            'learner_returns': sum(e['returns'][e['learner']] for e in episodes),
            'learner_hits': sum(e['hits'][e['learner']] for e in episodes),
            'learner_rally_hits': sum(e['rally_hits'][e['learner']] for e in episodes),
            'opponent_rally_hits': sum(e['rally_hits'][1-e['learner']] for e in episodes),
            'rally_length_counts': dict(Counter(lengths)),
            'counts': dict(Counter(e['outcome'] for e in episodes)),
            'by_opponent': by_opponent, 'by_role': by_role, 'episodes': episodes}


def evaluate_points(policy, opponents, seeds, config=None, trace_count=0):
    config = config or RallyConfig()
    started = time.perf_counter()
    world = RallyWorld()
    episodes, attempts = [], []
    for index, seed in enumerate(seeds):
        learner, server, deuce = index % 2, (index//2) % 2, (index//4) % 2 == 0
        if config.receive_only:
            learner = 1-server
        opponent = (index//8) % len(opponents)
        policies = [None, None]
        # A stage prelude is explicit native zero effort before learner control;
        # primary validation passes the default zero-prelude config.
        def staged(observation):
            return np.zeros(4) if world.state['time']+1e-9 < config.prelude_seconds else policy(observation)
        policies[learner], policies[1-learner] = staged, opponents[opponent]
        mode = 'feed' if np.random.default_rng(int(seed)).random() < config.feed_probability else 'serve'
        score = SetScore(server=server, ends=learner)
        # Point reports consistently draw the candidate as identity 0.
        score.server = server ^ learner
        episode, records = play_point(world, policies, int(seed), server=server, deuce=deuce,
            mode=mode, receiver=learner, difficulty=config.difficulty, max_seconds=config.max_seconds,
            trace=index < trace_count, score=score_snapshot(score), point_index=index)
        episode.update(learner=learner, opponent=opponent)
        before = score_snapshot(score)
        if episode['winner'] != -1:
            score.award(episode['winner'] ^ learner)
        annotate_score(episode, records, before, score_snapshot(score))
        episodes.append(episode)
        attempts.extend(records)
    return summarize(episodes) | {'seconds': time.perf_counter()-started, 'seeds': list(map(int, seeds)),
                                  'schema': 'rally-v1', 'model_hash': world.model_hash, 'attempts': attempts}


def play_set(policies, seed, *, first_server=0, ends=0, max_points=256, max_seconds=60., trace=True):
    """A full standard set; administrative limits never manufacture a winner."""
    return _play_scored(policies, seed, SetScore(server=first_server, ends=ends),
                        max_points, max_seconds, trace)


def play_contest(policies, seed, *, first_server=0, ends=0, max_seconds=30., trace=True):
    """One short competitive episode, with at most five physical points."""
    from .contest import ContestScore
    return _play_scored(policies, seed, ContestScore(first_server=first_server, ends=ends),
                        5, max_seconds, trace) | {'format': 'first-to-three'}


def evaluate_contests(policy, opponents, seeds, config=None):
    """Small competitive check; point counts remain separate from contest wins.

    Adjacent contests reverse the opening server. Four per opponent also balance
    physical ends. Each evaluation seed owns five disjoint physical point seeds.
    """
    config = config or RallyConfig()
    if config.feed_probability or config.difficulty != 1 or config.prelude_seconds or config.receive_only:
        raise ValueError('Contest evaluation requires ordinary full-difficulty serves')
    started = time.perf_counter()
    contests, episodes = [], []
    for index, seed in enumerate(seeds):
        opponent, ends = (index//2) % len(opponents), (index//(2*len(opponents))) % 2
        result = play_contest([policy, opponents[opponent]], int(seed)*5,
                              first_server=index % 2, ends=ends,
                              max_seconds=config.max_seconds, trace=False)
        for point in result['episodes']:
            # summarize() works in physical slots, whereas scored replays use
            # persistent identities. Convert only this untraced summary view.
            episodes.append({**point, 'winner': point['winner_slot'],
                             'learner': ends, 'opponent': opponent})
        contests.append({key: result[key] for key in ('seed', 'score', 'complete', 'points', 'mean_returns')}
                        | {'opponent': opponent, 'evaluation_seed': int(seed)})
    return summarize(episodes) | {'format': 'first-to-three', 'contests': contests,
        'contest_wins': sum(c['score']['winner'] == 0 for c in contests),
        'incomplete_contests': sum(not c['complete'] for c in contests),
        'seconds': time.perf_counter()-started, 'seeds': list(map(int, seeds))}


def _play_scored(policies, seed, score, max_points, max_seconds, trace):
    started = time.perf_counter()
    world = RallyWorld()
    episodes, attempts = [], []
    for index in range(max_points):
        before = score_snapshot(score)
        physical = [policies[score.slot(p)] for p in (0, 1)]
        episode, records = play_point(world, physical, seed+index,
            server=score.slot(score.server), deuce=score.deuce_court(), max_seconds=max_seconds,
            trace=trace, score=before, point_index=index)
        episode['winner_slot'] = episode['winner']
        if episode['winner'] != -1:
            episode['winner'] ^= score.ends
            score.award(episode['winner'])
        annotate_score(episode, records, before, score_snapshot(score))
        episodes.append(episode)
        attempts.extend(records)
        if score.winner != -1 or episode['winner'] == -1:
            break
    return {'schema': 'rally-v1', 'model_hash': world.model_hash, 'seed': seed,
            'score': score_snapshot(score), 'complete': score.winner != -1,
            'points': len(episodes), 'episodes': episodes, 'attempts': attempts,
            'seconds': time.perf_counter()-started,
            'mean_returns': float(np.mean([sum(e['returns']) for e in episodes]))}


def assert_replay_state(actual, expected):
    def poses(left, right):
        if left.keys() != right.keys():
            raise ValueError('Replayed body set differs')
        for name, pose in left.items():
            np.testing.assert_allclose(pose['position'], right[name]['position'], atol=1e-8, rtol=0)
            orientation = np.asarray(right[name]['orientation'])
            if np.dot(pose['orientation'], orientation) < 0:
                orientation = -orientation  # q and -q describe the same pose.
            np.testing.assert_allclose(pose['orientation'], orientation, atol=1e-8, rtol=0)

    poses(actual['bodies'], expected['bodies'])
    for field in ('tick', 'time', 'timestep', 'ball_velocity', 'ball_spin', 'warnings', 'faulted'):
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-8, rtol=0)
    if len(actual['players']) != len(expected['players']):
        raise ValueError('Replayed player count differs')
    for player, saved in zip(actual['players'], expected['players']):
        poses(player['bodies'], saved['bodies'])
        for field in ('position', 'velocity', 'joint_angles', 'joint_velocities', 'torques', 'force', 'racket_velocity'):
            np.testing.assert_allclose(player[field], saved[field], atol=1e-8, rtol=0)


def replay_attempt(world, attempt):
    frames, setup = attempt['frames'], attempt['setup']
    if len(frames) < 2 or frames[0]['actions'] is not None:
        raise ValueError('A replay attempt needs an initial state and physical action steps')
    if setup['kind'] == 'serve':
        from .serve import setup_serve, update_serve
        setup_serve(world, setup['seed'], setup['server'], setup['deuce'], setup['attempt'],
                    difficulty=setup.get('difficulty', 1.))
    elif setup['kind'] == 'feed':
        world.reset_feed(setup['seed'], setup['receiver'], setup['difficulty'])
    else:
        raise ValueError('Unknown replay setup')
    if world.setup != setup:
        raise AssertionError('Recorded setup differs from its seeded reconstruction')
    for index, frame in enumerate(frames):
        if index:
            world.step(frame['actions'])
            if setup['kind'] == 'serve':
                update_serve(world)
        assert_replay_state(world.state, frame['state'])
    rules = world.rules
    observed = {'result': rules.result, 'winner': rules.winner, 'outcome': rules.reason or 'timeout',
                'returns': list(rules.legal_returns), 'hits': list(rules.racket_hits)}
    if any(attempt[key] != value for key, value in observed.items()):
        raise AssertionError('Replayed physical point decisions differ')
    observed['rally_hits'] = _rally_hits(world)
    if 'rally_hits' in attempt and attempt['rally_hits'] != observed['rally_hits']:
        raise AssertionError('Replayed rally hits differ from the attempt')
    return observed | {'steps': len(frames)-1,
                       'time': world.state['time']-frames[0]['state']['time']}


def replay_point(world, episode, attempts):
    """Resolve physical attempts into one result, including retries and double faults."""
    number, lets, steps, elapsed = 1, 0, 0, 0.
    hits, returns = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    rally_hits = np.zeros(2, dtype=int)
    for index, attempt in enumerate(attempts):
        setup = attempt['setup']
        if (attempt['seed'] != episode['seed'] or attempt['attempt'] != number
                or setup['kind'] != episode['mode'] or setup['seed'] != episode['seed']+lets*1009):
            raise AssertionError('Recorded service attempt sequence differs')
        if setup['kind'] == 'serve' and (setup['server'] != episode['server']
                or setup['deuce'] != episode['deuce'] or setup['attempt'] != number):
            raise AssertionError('Recorded server or service half differs')
        observed = replay_attempt(world, attempt)
        steps += observed['steps']
        elapsed += observed['time']
        hits += observed['hits']
        rally_hits += observed['rally_hits']
        returns += observed['returns']
        final = index == len(attempts)-1
        winner, outcome = -1, 'timeout'
        if observed['result'] == 'point':
            winner, outcome = observed['winner'], observed['outcome']
        elif observed['result'] == 'fault' and number == 2:
            winner, outcome = 1-episode['server'], 'double_fault'
        elif elapsed+1e-9 < episode['max_seconds']:
            if observed['result'] == 'fault':
                number = 2
            elif observed['result'] == 'let':
                lets += 1
            else:
                raise AssertionError('A live point was truncated before its declared limit')
            if final:
                raise AssertionError('Replay omits a required service attempt')
            continue
        if not final:
            raise AssertionError('Replay continues after a resolved point')
    expected = {'outcome': outcome, 'steps': steps, 'attempt': number, 'lets': lets,
                'hits': hits.tolist(), 'returns': returns.tolist()}
    if any(episode[key] != value for key, value in expected.items()):
        raise AssertionError('Episode summary differs from replayed attempts')
    if 'rally_hits' in episode and episode['rally_hits'] != rally_hits.tolist():
        raise AssertionError('Episode rally hits differ from replayed attempts')
    np.testing.assert_allclose(episode['time'], elapsed, atol=1e-8, rtol=0)
    return winner, steps


def replay(path):
    """Verify recorded points; only a complete reconstructed set earns set_complete."""
    contents = Path(path).read_bytes()
    report = json.loads(contents)
    world = RallyWorld()
    if report.get('schema') != 'rally-v1' or report['model_hash'] != world.model_hash:
        raise ValueError('Replay model or schema differs from the current experiment')
    if not report.get('attempts') or not report.get('episodes'):
        raise ValueError('Replay requires nonempty point and action records')
    groups = [(index, list(records)) for index, records in
              groupby(report['attempts'], key=lambda a: a['point_index'])]
    indices = [index for index, _ in groups]
    if indices != sorted(set(indices)) or any(not 0 <= i < len(report['episodes']) for i in indices):
        raise ValueError('Replay point order is invalid')
    is_set = 'complete' in report or 'score' in report
    if is_set and (indices != list(range(len(report['episodes']))) or report['points'] != len(indices)):
        raise ValueError('Set verification requires every point and attempt')
    first = groups[0][1][0]['score_before']
    if first['server'] not in (0, 1) or first['ends'] not in (0, 1):
        raise ValueError('Initial server and ends must identify the two players')
    is_contest = report.get('format') == 'first-to-three'
    if is_contest:
        from .contest import ContestScore
        score = ContestScore(first_server=first['first_server'], ends=first['ends'],
                             first_deuce=first['first_deuce'])
    else:
        score = SetScore(server=first['server'], ends=first['ends'])
    steps = 0
    for index, attempts in groups:
        episode = report['episodes'][index]
        if not is_set:
            score = SetScore(server=episode['server'] ^ episode['learner'], ends=episode['learner'])
        before = score_snapshot(score)
        if score.winner != -1 or episode['score_before'] != before:
            raise AssertionError('Recorded score before the point differs')
        if is_set and (episode['mode'] != 'serve' or episode['server'] != score.slot(score.server)
                or episode['deuce'] != score.deuce_court() or episode['seed'] != report['seed']+index):
            raise AssertionError('Set service order, service half or seed differs')
        winner, count = replay_point(world, episode, attempts)
        steps += count
        identity = winner ^ score.ends if winner != -1 else -1
        if episode['winner'] != (identity if is_set else winner):
            raise AssertionError('Recorded point winner differs')
        if is_set and episode['winner_slot'] != winner:
            raise AssertionError('Recorded physical winner differs')
        if winner != -1:
            score.award(identity)
        after = score_snapshot(score)
        if episode['score_after'] != after:
            raise AssertionError('Recorded score after the point differs')
        for position, attempt in enumerate(attempts):
            expected = after if position == len(attempts)-1 else before
            if attempt['score_before'] != before or attempt['score_after'] != expected:
                raise AssertionError('Recorded attempt score transition differs')
    complete = is_set and score.winner != -1
    if is_set and (report['score'] != score_snapshot(score) or report['complete'] != complete):
        raise AssertionError('Reported set score or completion differs from physical points')
    # Full-point summaries include untraced episodes. Check their arithmetic;
    # native verification above still covers only the recorded action groups.
    for field, other in (('learner_rally_hits', False), ('opponent_rally_hits', True)):
        if field in report:
            if any('rally_hits' not in e or 'learner' not in e for e in report['episodes']):
                raise AssertionError('Rally-hit summary requires episode counts and learner identities')
            count = sum(e['rally_hits'][e['learner'] ^ int(other)] for e in report['episodes'])
            if report[field] != count:
                raise AssertionError('Reported rally-hit summary differs from episode counts')
    return {'attempts': len(report['attempts']), 'steps': steps, 'points': len(groups),
            'verified': True, 'set_complete': complete and not is_contest,
            'contest_complete': complete and is_contest,
            'report_sha256': hashlib.sha256(contents).hexdigest()}
