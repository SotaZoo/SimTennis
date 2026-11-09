"""Reward-trained returns from reproducible, physically generated incoming states.

Frozen actors generate starting states, never action labels. Prefixes run the
ordinary service, effort and contact code; resetting restores complete native
and rule history. The small bank stays in memory inside each collector.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import math
import time

import gymnasium as gym
import numpy as np

from .environment import RallyConfig, RallyEnv
from .serve import setup_serve, update_serve
from .world import player_action


@dataclass(frozen=True)
class PracticeConfig:
    bank_seed: int = 74000
    bank_size: int = 8
    lead_seconds: float = .8
    kind: str = 'mixed'
    generation_seconds: float = 8.
    lead_min_seconds: float | None = None
    miss_distance_scale: float = 0.

    def __post_init__(self):
        if (not isinstance(self.bank_seed, int) or not 0 <= self.bank_seed < 2**32
                or not isinstance(self.bank_size, int) or not 1 <= self.bank_size <= 128
                or not math.isfinite(self.lead_seconds) or not 0 <= self.lead_seconds <= 4
                or abs(self.lead_seconds/.02-round(self.lead_seconds/.02)) > 1e-8
                or not math.isfinite(self.generation_seconds) or not 1 <= self.generation_seconds <= 30
                or not math.isfinite(self.miss_distance_scale) or self.miss_distance_scale < 0
                or (self.lead_min_seconds is not None
                    and (not math.isfinite(self.lead_min_seconds)
                         or not 0 <= self.lead_min_seconds <= self.lead_seconds
                         or abs(self.lead_min_seconds/.02-round(self.lead_min_seconds/.02)) > 1e-8))
                or self.kind not in ('first', 'later', 'mixed')):
            raise ValueError('Invalid return-practice configuration')


def generate_bank(world, policies, config, difficulty):
    """Replay each declared case twice, retaining only the selected branch states.

    The first pass measures contact time (or closest approach on misses). The
    second captures first-return states at a fixed lead or every control step
    within the requested lead range, clamped to the turn start. All later replies
    always start at incoming-turn onset to retain all repositioning time.
    Source failures and absent/illegal later shots remain in the compact summary.
    """
    started = time.perf_counter()
    cases, records, native_steps = [], [], 0
    lead = round(config.lead_seconds/world.control_seconds)
    minimum_lead = lead if config.lead_min_seconds is None else round(config.lead_min_seconds/world.control_seconds)
    for index in range(config.bank_size):
        seed, server, deuce = config.bank_seed+index, index % 2, (index//2) % 2 == 0
        source = (index//4) % len(policies)
        setup_serve(world, seed, server, deuce, difficulty=difficulty)
        actions, turns = [], {}
        for step in range(round(config.generation_seconds/world.control_seconds)):
            if world.rules.result != 'live':
                break
            count = sum(world.rules.racket_hits)
            if count >= 1:
                receiver = 1-world.rules.last_hitter
                distance = float(np.linalg.norm(np.asarray(world.state['bodies']['ball']['position'])
                                                - world.stringbed(receiver)))
                turns.setdefault(count, []).append((step, distance, receiver))
            command = [player_action(world, player, policies[source]) for player in (0, 1)]
            actions.append(np.asarray(command).copy())
            world.step(command)
            update_serve(world)
            native_steps += 1
        targets = {}
        record = {'seed': seed, 'server': server, 'deuce': deuce, 'source': source,
                  'outcome': world.rules.reason or 'timeout',
                  'source_returns': list(world.rules.legal_returns), 'cases': []}
        for number in range(1, max(2, max(turns, default=0))+1):
            label = 'first' if number == 1 else 'later'
            candidates = turns.get(number, [])
            if not candidates:
                record['cases'].append({'kind': label, 'incoming_turn': number,
                                        'status': 'no_incoming_turn'})
                continue
            # A later incoming shot must actually land legally or be volleyed;
            # otherwise it was an opponent fault, not a missed return obligation.
            # Accepted strikes alternate. Serves do not count as legal returns;
            # the nth incoming turn requires the hitter's floor(n/2)th return.
            hitter = 1-candidates[0][2]
            if number > 1 and world.rules.legal_returns[hitter] < number//2:
                status = 'source_shot_unresolved' if world.rules.result == 'live' else 'source_shot_not_legal'
                record['cases'].append({'kind': label, 'incoming_turn': number, 'status': status})
                continue
            if config.kind != 'mixed' and config.kind != label:
                continue
            closest = min(candidates, key=lambda value: value[1])
            first_step = (candidates[0][0] if number > 1
                          else max(candidates[0][0], closest[0]-lead))
            last_step = (first_step if number > 1
                         else max(candidates[0][0], closest[0]-minimum_lead))
            metadata = {'kind': label, 'incoming_turn': number,
                        'seed': seed, 'source': source, 'server': server,
                        'deuce': deuce, 'learner': candidates[0][2],
                        'anchor_step': closest[0], 'source_closest_distance': closest[1]}
            for step in range(first_step, last_step+1):
                targets[step] = metadata | {'prefix_steps': step}
            selection = ({'prefix_steps': first_step} if first_step == last_step else
                         {'prefix_steps_min': first_step, 'prefix_steps_max': last_step,
                          'prefix_state_count': last_step-first_step+1})
            record['cases'].append(metadata | selection | {'status': 'included'})
        setup_serve(world, seed, server, deuce, difficulty=difficulty)
        for step, command in enumerate(actions):
            if step in targets:
                cases.append({'checkpoint': world.capture(), **targets[step]})
            if not targets or step >= max(targets):
                break
            world.step(command)
            update_serve(world)
            native_steps += 1
        records.append(record)
    if not cases:
        raise ValueError('The declared source bank produced no eligible incoming states')
    counts = dict(Counter(case['kind'] for case in cases))
    summary = {'config': asdict(config), 'cases': len(cases), 'counts': counts,
               'source_case_counts': dict(Counter(kind for kind, seed in
                    {(case['kind'], case['seed']) for case in cases})),
               'incoming_turn_counts': dict(Counter(number for number, seed in
                    {(case['incoming_turn'], case['seed']) for case in cases})),
               'generation_seconds': time.perf_counter()-started,
               'generation_control_steps': native_steps, 'sources': records}
    return cases, summary


class ReturnPracticeEnv(RallyEnv):
    """Use whole-point transitions, terminating on one newly earned legal return."""
    def __init__(self, opponents, config=None, practice_config=None):
        super().__init__(opponents, config or RallyConfig(contact_reward=1., return_reward=10.,
                                                         win_reward=2., max_seconds=6.))
        self.practice_config = practice_config or PracticeConfig()
        if self.config.prelude_seconds:
            raise ValueError('Practice prefixes use lead_seconds, not a second prelude')
        self.bank, self.bank_summary = generate_bank(self.world, opponents, self.practice_config,
                                                    self.config.difficulty)
        # Balance reply types, then physical source cases, then preparation time.
        # Extra cached prefixes must not silently change which shots are sampled.
        groups = {}
        for index, case in enumerate(self.bank):
            groups.setdefault(case['kind'], {}).setdefault(case['seed'], []).append(index)
        self._reset_groups = [list(sources.values()) for sources in groups.values()]
        self.bank_summary['reset_sampling'] = ('uniform_cached_state' if self.practice_config.lead_min_seconds is None
                                               else 'uniform_kind_then_source_then_prefix')

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if self.practice_config.lead_min_seconds is None:
            index = options.get('case', int(self.np_random.integers(len(self.bank))))
        elif 'case' in options:
            index = options['case']
        else:
            sources = self._reset_groups[int(self.np_random.integers(len(self._reset_groups)))]
            prefixes = sources[int(self.np_random.integers(len(sources)))]
            index = prefixes[int(self.np_random.integers(len(prefixes)))]
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.bank):
            raise ValueError('Practice case must index the declared bank')
        case = self.bank[index]
        self.case_index, self.case_kind = int(index), case['kind']
        self.world.restore(case['checkpoint'])
        self.episode_seed = case['seed']
        self.learner, self.server, self.deuce = case['learner'], case['server'], case['deuce']
        self.opponent_index = case['source']
        self.opponent = self.opponents[self.opponent_index]
        self.mode, self.attempt, self.lets, self.elapsed = 'practice', 1, 0, 0.
        self.hits = np.zeros(2, dtype=int)
        self.returns = np.zeros(2, dtype=int)
        self.applied_steps = 0
        self.action_clip_counts = np.zeros(4, dtype=int)
        self._potential = self._distance_potential()
        self._closest_distance = -self._potential
        self._done = False
        return self.world.observation(self.learner), self._info()

    def _info(self, winner=-1, outcome='live'):
        return super()._info(winner, outcome) | {'practice_kind': self.case_kind,
                'practice_case': self.case_index, 'applied_steps': self.applied_steps,
                'incoming_turn': self.bank[self.case_index]['incoming_turn'],
                'practice_lead_seconds': (self.bank[self.case_index]['anchor_step']
                    - self.bank[self.case_index]['prefix_steps'])*self.world.control_seconds,
                'is_success': bool(self.returns[self.learner])}

    def step(self, action):
        before_hits = self.hits[self.learner]
        potential = self._potential
        applied = self.world.incoming(self.learner)
        observation, _, terminated, truncated, info = super().step(action)
        if applied:
            distance = float(np.linalg.norm(np.asarray(self.world.state['bodies']['ball']['position'])
                                            - self.world.stringbed(self.learner)))
            self._closest_distance = min(self._closest_distance, distance)
        success = bool(self.returns[self.learner])
        if success:
            terminated, truncated = True, False
        next_potential = 0. if terminated else self._potential
        reward = (self.config.contact_reward*float(self.hits[self.learner]-before_hits)
                  + self.config.shaping_scale*(self.config.discount*next_potential-potential)
                  - (.001*float(np.mean(np.asarray(action)**2)) if applied else 0.))
        if success:
            reward += self.config.return_reward
            info.update(winner=self.learner, won=True, outcome='legal_return')
        elif terminated or truncated:
            reward -= self.config.win_reward
            info.update(winner=-1, won=False)
        # Trajectory feedback distinguishes contactless misses. It changes the
        # acquisition objective, not native outcomes or the actor's observations.
        miss_reward = (-self.practice_config.miss_distance_scale*self._closest_distance
                       if (terminated or truncated) and not self.hits[self.learner] else 0.)
        reward += miss_reward
        info.update(closest_approach_m=self._closest_distance, miss_distance_reward=miss_reward)
        self._potential = next_potential
        self._done = terminated or truncated
        return observation, reward, terminated, truncated, info


def evaluate_practice(policy, opponents, practice_config=None, rally_config=None, seeds=range(16), *, env=None):
    """Evaluate cached cases in order; reset seeds identify reproducible repeats."""
    started = time.perf_counter()
    owned = env is None
    env = env or ReturnPracticeEnv(opponents, rally_config, practice_config)
    episodes = []
    try:
        for index, seed in enumerate(seeds):
            observation, info = env.reset(seed=int(seed), options={'case': index % len(env.bank)})
            reward_sum = 0.
            while True:
                observation, reward, done, truncated, info = env.step(policy(observation))
                reward_sum += reward
                if done or truncated:
                    break
            episodes.append(info | {'reset_seed': int(seed), 'reward': reward_sum})
        by_kind = {}
        for kind in ('first', 'later'):
            group = [episode for episode in episodes if episode['practice_kind'] == kind]
            by_kind[kind] = {'episodes': len(group), 'successes': sum(x['is_success'] for x in group)}
        successes = sum(x['is_success'] for x in episodes)
        return {'episodes': episodes, 'count': len(episodes), 'successes': successes,
                'points': len(episodes), 'wins': successes,
                'win_fraction': successes/max(1, len(episodes)),
                'timeouts': sum(x['outcome'] == 'timeout' for x in episodes),
                'mean_returns': successes/max(1, len(episodes)),
                'counts': dict(Counter(x['outcome'] for x in episodes)),
                'success_fraction': successes/max(1, len(episodes)), 'by_kind': by_kind,
                'applied_steps': sum(x['applied_steps'] for x in episodes),
                'bank': env.bank_summary, 'seconds': time.perf_counter()-started}
    finally:
        if owned:
            env.close()
