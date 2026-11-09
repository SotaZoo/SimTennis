"""Native branch fidelity and focused reward accounting, without trained fixtures."""
from dataclasses import replace
import math
import unittest

import numpy as np


def physical_test_actor(observation):
    """A fixed test fixture: one legal native return and a nearby failed case.

    This actor is only a physical setup fixture, not a learning demonstration.
    Its clock offset is the measured service impact for fixture seed 86000.
    """
    o = observation
    position, velocity = np.asarray([o[9]*5, o[10]*15]), np.asarray([o[11]*10, o[12]*10])
    angle, speed, lift, lift_speed = o[13]*math.pi, o[14]*30, o[32]*math.pi, o[33]*30
    force = 500*(np.array([2.537996873868552, 7.593054527225315])-position)-200*velocity
    swing = (180*(-.7-angle)-12*speed if o[23]*6+.57 < 1.4276081448529903
             else 4.024642608610729)
    inertia = .224*math.cos(angle)**2+.0051*math.sin(angle)**2
    torque = -inertia*(144*(lift+.5967997835391167)+24*lift_speed)
    return np.clip(np.r_[force/700, swing/40, torque/40], -1, 1)


class RallyCheckpointTests(unittest.TestCase):
    def test_restore_reproduces_native_contacts_observations_and_history(self):
        from simtennis.rally.world import RallyWorld, player_action
        world = RallyWorld()
        world.reset_feed(14)
        for _ in range(30):
            world.step([player_action(world, player, lambda observation: np.zeros(4)) for player in (0, 1)])
        checkpoint = world.capture()
        initial = world.state
        def continue_point():
            trajectory = []
            for _ in range(170):
                if world.rules.result != 'live':
                    break
                world.step([player_action(world, player, lambda observation: np.zeros(4)) for player in (0, 1)])
                trajectory.append((world.observation(0).tolist(), world.state,
                                   list(world.rules.racket_hits), list(world.rules.legal_returns),
                                   world.rules.reason))
            return trajectory
        expected = continue_point()
        self.assertTrue(any(frame[1]['contacts'] for frame in expected))
        world.restore(checkpoint)
        self.assertEqual(world.state, initial)
        self.assertEqual(continue_point(), expected)
        world.restore(checkpoint)
        self.assertEqual(world.state, initial, 'A continuation must not mutate its saved checkpoint')
        with self.assertRaisesRegex(ValueError, 'same rally model'):
            world.restore(replace(checkpoint, model_hash='different'))
        self.assertEqual(world.state, initial)


class ReturnPracticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from simtennis.rally.practice import PracticeConfig, ReturnPracticeEnv
        cls.env = ReturnPracticeEnv([physical_test_actor],
                    practice_config=PracticeConfig(bank_seed=86000, bank_size=2))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_bank_preserves_first_and_later_cases_ends_and_source_failure(self):
        env = self.env
        self.assertEqual(env.bank_summary['counts'], {'first': 2, 'later': 1})
        self.assertEqual({case['learner'] for case in env.bank}, {0, 1})
        self.assertGreater(env.bank_summary['generation_seconds'], 0)
        self.assertGreater(env.bank_summary['generation_control_steps'], 0)
        failure = env.bank_summary['sources'][1]
        self.assertEqual(failure['seed'], 86001)
        self.assertEqual(failure['outcome'], 'rally_out')
        self.assertIn({'kind': 'later', 'incoming_turn': 2,
                       'status': 'source_shot_not_legal'}, failure['cases'])
        for index, case in enumerate(env.bank):
            observation, info = env.reset(options={'case': index})
            self.assertEqual(observation.shape, (34,))
            self.assertTrue(env.world.incoming(env.learner))
            self.assertEqual(info['hits'], [0, 0])
            self.assertEqual(info['returns'], [0, 0])
            self.assertEqual(info['applied_steps'], 0)
            self.assertFalse(info['is_success'])
            self.assertGreater(sum(env.world.rules.racket_hits), 0)
            if case['kind'] == 'later':
                self.assertEqual(env.world.rules.racket_hits, [1, 1])
                # Preserve all movement time after the opposing shot.
                self.assertLessEqual(env.world.state['time']
                    - env.world.rules.first_racket_tick*env.world.state['timestep'], .02+1e-9)

    def test_new_return_ends_practice_without_inherited_serve_credit(self):
        env = self.env
        observation, info = env.reset(options={'case': 0})
        initial_time = env.world.state['time']
        for _ in range(300):
            observation, reward, done, truncated, info = env.step(physical_test_actor(observation))
            if done or truncated:
                break
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertTrue(info['is_success'])
        self.assertEqual(info['learner_returns'], 1)
        self.assertEqual(info['hits'][env.learner], 1)
        self.assertEqual(info['outcome'], 'legal_return')
        self.assertGreater(info['applied_steps'], 0)
        self.assertAlmostEqual(info['time'], env.world.state['time']-initial_time, places=9)
        self.assertGreater(reward, 9.)
        with self.assertRaisesRegex(RuntimeError, 'Reset'):
            env.step(np.zeros(4))

    def test_later_failure_receives_no_credit_for_preceding_opponent_return(self):
        env = self.env
        case = next(index for index, value in enumerate(env.bank) if value['kind'] == 'later')
        env.reset(options={'case': case})
        for _ in range(300):
            _, _, done, truncated, info = env.step(np.zeros(4))
            if done or truncated:
                break
        self.assertTrue(done or truncated)
        self.assertFalse(info['is_success'])
        self.assertEqual(info['learner_returns'], 0)
        self.assertEqual(info['hits'][env.learner], 0)
        self.assertEqual(info['returns'][1-env.learner], 1)

    def test_cached_evaluation_is_reproducible_and_retains_env(self):
        from simtennis.rally.practice import evaluate_practice
        env = self.env
        before = env.bank_summary['generation_seconds']
        first = evaluate_practice(physical_test_actor, [physical_test_actor], seeds=range(3), env=env)
        second = evaluate_practice(physical_test_actor, [physical_test_actor], seeds=range(3), env=env)
        self.assertEqual(first['episodes'], second['episodes'])
        self.assertEqual(first['points'], 3)
        self.assertEqual(first['wins'], 1)
        self.assertEqual(first['timeouts'], 0)
        self.assertEqual(first['by_kind']['later']['episodes'], 1)
        self.assertEqual(first['bank']['generation_seconds'], before)
        self.assertIsNotNone(env.world)

    def test_reward_uses_recorded_config(self):
        env = self.env
        previous = env.config
        env.config = replace(previous, contact_reward=0., return_reward=5., win_reward=3., shaping_scale=0.)
        try:
            observation, _ = env.reset(options={'case': 0})
            while True:
                observation, reward, done, truncated, info = env.step(physical_test_actor(observation))
                if done or truncated:
                    break
            self.assertTrue(info['is_success'])
            self.assertEqual(reward, 5.)
        finally:
            env.config = previous

    def test_no_feed_or_position_rewrites_after_restoring_a_case(self):
        env = self.env
        original = env.world._sim
        class ReadOnlySetup:
            def __getattr__(self, name):
                if name in ('feed_ball', 'set_player_start', 'reset'):
                    raise AssertionError('A practice continuation must not rewrite setup')
                return getattr(original, name)
        env.world._sim = ReadOnlySetup()
        try:
            observation, _ = env.reset(options={'case': 0})
            while True:
                observation, _, done, truncated, _ = env.step(physical_test_actor(observation))
                if done or truncated:
                    break
        finally:
            env.world._sim = original

    def test_miss_feedback_changes_only_terminal_reward_not_native_trajectory(self):
        env = self.env
        previous = env.practice_config
        self.addCleanup(setattr, env, 'practice_config', previous)

        def run(case, scale, actor):
            env.practice_config = replace(previous, miss_distance_scale=scale)
            observation, _ = env.reset(options={'case': case})
            distances = [np.linalg.norm(np.asarray(env.world.state['bodies']['ball']['position'])
                                       - env.world.stringbed(env.learner))]
            rows = []
            while True:
                incoming = env.world.incoming(env.learner)
                observation, reward, done, truncated, info = env.step(actor(observation))
                if incoming:
                    distances.append(np.linalg.norm(np.asarray(env.world.state['bodies']['ball']['position'])
                                                    - env.world.stringbed(env.learner)))
                rows.append((observation.copy(), env.world.state, reward, done, truncated, info))
                if done or truncated:
                    return rows, min(distances)

        outcomes = set()
        for case, actor in ((0, physical_test_actor), (1, lambda o: np.zeros(4)),
                            (2, physical_test_actor)):
            base, _ = run(case, 0., actor)
            shaped, closest = run(case, 1.7, actor)
            self.assertEqual(len(base), len(shaped))
            for a, b in zip(base, shaped):
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1], b[1])
                self.assertEqual(a[3:5], b[3:5])
                self.assertEqual(a[5]['outcome'], b[5]['outcome'])
                expected = (-1.7*closest if (b[3] or b[4]) and
                            not b[5]['hits'][env.learner] else 0.)
                self.assertAlmostEqual(b[2]-a[2], expected, places=10)
                self.assertAlmostEqual(b[5]['miss_distance_reward'], expected, places=10)
            last = shaped[-1][5]
            self.assertAlmostEqual(last['closest_approach_m'], closest, places=10)
            outcomes.add((last['is_success'], bool(last['hits'][env.learner])))
        self.assertEqual(outcomes, {(True, True), (False, True), (False, False)})

    def test_miss_feedback_rejects_invalid_coefficients(self):
        from simtennis.rally.practice import PracticeConfig
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                PracticeConfig(miss_distance_scale=value)

    def test_bank_keeps_third_incoming_turn_without_inheriting_prior_legality(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from simtennis.rally.practice import PracticeConfig, generate_bank

        class ScriptedHistory:
            """Source-history fixture; native branch fidelity is checked separately."""
            control_seconds = .02
            # Two legal rally shots, then an out shot by the first receiver.
            history = [(0, -1, [0, 0]), (1, 0, [0, 0]), (1, 0, [0, 0]),
                       (2, 1, [0, 0]), (2, 1, [0, 1]), (3, 0, [0, 1]),
                       (3, 0, [1, 1]), (4, 1, [1, 1]), (4, 1, [1, 1])]

            def reset(self):
                self.index = -1
                self.step(None)

            def step(self, actions):
                self.index += 1
                count, hitter, returns = self.history[self.index]
                self.rules = SimpleNamespace(racket_hits=[(count+1)//2, count//2],
                    last_hitter=hitter, legal_returns=returns,
                    result='point' if self.index == 8 else 'live', reason='rally_out')
                self.state = {'bodies': {'ball': {'position': [1., 1., float(self.index)]}}}

            def stringbed(self, player):
                return np.zeros(3)

            def capture(self):
                return self.index

        world = ScriptedHistory()
        with patch('simtennis.rally.practice.setup_serve', side_effect=lambda *a, **k: world.reset()), \
             patch('simtennis.rally.practice.update_serve'), \
             patch('simtennis.rally.practice.player_action', return_value=np.zeros(4)):
            bank, summary = generate_bank(world, [object()], PracticeConfig(bank_size=1), 1.)
        self.assertEqual([case['incoming_turn'] for case in bank], [1, 2, 3])
        self.assertEqual([case['checkpoint'] for case in bank], [1, 3, 5])
        self.assertEqual(summary['incoming_turn_counts'], {1: 1, 2: 1, 3: 1})
        self.assertIn({'kind': 'later', 'incoming_turn': 4,
                       'status': 'source_shot_not_legal'}, summary['sources'][0]['cases'])

    def test_preparation_range_caches_native_states_and_balances_reply_types(self):
        from simtennis.rally.practice import PracticeConfig, ReturnPracticeEnv
        config = PracticeConfig(bank_seed=86000, bank_size=2, lead_seconds=.5, lead_min_seconds=.2)
        env = ReturnPracticeEnv([physical_test_actor], practice_config=config)
        self.addCleanup(env.close)
        self.assertEqual(env.bank_summary['counts'], {'first': 32, 'later': 1})
        self.assertEqual(env.bank_summary['source_case_counts'], {'first': 2, 'later': 1})
        # The persistent summary stays source-sized, not prefix-sized.
        self.assertEqual(sum(len(source['cases']) for source in env.bank_summary['sources']), 4)
        included = env.bank_summary['sources'][0]['cases'][0]
        self.assertEqual(included['prefix_state_count'], 16)
        self.assertNotIn('checkpoint', included)
        observed = set()
        for index, case in enumerate(env.bank):
            if case['kind'] != 'first':
                continue
            observation, info = env.reset(options={'case': index})
            observed.add(round(info['practice_lead_seconds'], 2))
            self.assertEqual(info['hits'], [0, 0])
            self.assertEqual(info['returns'], [0, 0])
            self.assertEqual(env.world.rules.racket_hits[env.learner], 0)
            # Every range state must also be the exact state of a fixed-lead bank.
            if case['seed'] == 86000 and info['practice_lead_seconds'] in (.2, .5):
                fixed = ReturnPracticeEnv([physical_test_actor], practice_config=replace(config,
                    bank_size=1, lead_min_seconds=None, lead_seconds=info['practice_lead_seconds']))
                try:
                    expected, _ = fixed.reset(options={'case': 0})
                    np.testing.assert_array_equal(observation, expected)
                    actual = env.step(np.array([.1, -.1, .05, -.02]))
                    replay = fixed.step(np.array([.1, -.1, .05, -.02]))
                    np.testing.assert_array_equal(actual[0], replay[0])
                    self.assertEqual(actual[1:4], replay[1:4])
                finally:
                    fixed.close()
        self.assertEqual(observed, {round(.2+.02*index, 2) for index in range(16)})
        # Many cached first prefixes must not drown out the one later source case.
        first, seeds, leads = 0, set(), set()
        for index in range(400):
            _, info = env.reset(seed=index)
            first += info['practice_kind'] == 'first'
            if info['practice_kind'] == 'first':
                seeds.add(info['seed'])
                leads.add(round(info['practice_lead_seconds'], 2))
        self.assertTrue(150 <= first <= 250)
        self.assertEqual(seeds, {86000, 86001})
        self.assertEqual(leads, observed)

    def test_fixed_preparation_keeps_existing_seeded_case_selection(self):
        env = self.env
        for seed in range(12):
            _, info = env.reset(seed=seed)
            self.assertEqual(info['practice_case'], int(np.random.default_rng(seed).integers(len(env.bank))))
        from simtennis.rally.practice import PracticeConfig
        for value in (-.1, .501, .8, float('nan'), float('inf')):
            with self.subTest(minimum=value), self.assertRaises(ValueError):
                PracticeConfig(lead_seconds=.5, lead_min_seconds=value)


if __name__ == '__main__':
    unittest.main()
