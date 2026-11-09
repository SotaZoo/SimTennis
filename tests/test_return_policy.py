import unittest
try:
    import numpy as np
    import gymnasium
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install the optional rl dependencies')
class ReturnEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        from simtennis.returns.environment import ReturnEnv
        env = ReturnEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_seeded_reset_effort_and_independent_worlds(self):
        env = self.make_env()
        other = self.make_env()
        first, _ = env.reset(seed=11)
        second, _ = other.reset(seed=11)
        np.testing.assert_array_equal(first, second)
        before = env.state
        obs, _, terminated, truncated, _ = env.step(np.array([.5, -.5, .5], np.float32))
        self.assertFalse(terminated or truncated)
        self.assertGreater(env.state['players'][0]['velocity'][0], before['players'][0]['velocity'][0])
        self.assertLess(env.state['players'][0]['velocity'][2], before['players'][0]['velocity'][2])
        self.assertGreater(env.state['players'][0]['joint_velocities'][0], 0)
        self.assertEqual(other.state['time'], 0)
        np.testing.assert_array_equal(obs, other.step([.5, -.5, .5])[0])
        np.testing.assert_array_equal(env.reset(seed=11)[0], first)

    def test_invalid_actions_leave_physics_unchanged(self):
        env = self.make_env()
        env.reset(seed=3)
        for action in ([0, 0], [0, 0, np.nan], [0, np.inf, 0], [2, 0, 0]):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.state['time'], 0)

    def test_court_resets_cover_both_axes_and_angles_independently(self):
        env = self.make_env(distribution='court')
        cars, launches, angles = [], [], []
        for seed in range(200):
            obs, info = env.reset(seed=seed)
            car = env.state['players'][0]
            np.testing.assert_allclose(car['position'][::2], info['car_start'])
            self.assertEqual(car['velocity'], [0, 0, 0])
            self.assertEqual(env.state['time'], 0)
            self.assertLess(info['feed']['position'][2], -5)
            cars.append(info['car_start'])
            launches.append(info['feed']['position'][::2])
            angles.append(np.arctan2(info['feed']['velocity'][0], info['feed']['velocity'][2]))
            np.testing.assert_array_equal(env.reset(seed=seed)[0], obs)
        cars, launches = np.asarray(cars), np.asarray(launches)
        self.assertGreater(np.ptp(cars[:, 0]), 5.5)
        self.assertGreater(np.ptp(cars[:, 1]), 4.5)
        self.assertGreater(np.ptp(launches[:, 0]), 5.5)
        self.assertLess(min(angles), -.2)
        self.assertGreater(max(angles), .2)
        # Feed sampling must not follow a repositioned car, even during replay.
        _, original = env.reset(seed=17)
        _, moved = env.reset(seed=17, options={'car_start': [-2, 10]})
        self.assertEqual(original['feed'], moved['feed'])
        self.assertEqual(moved['car_start'], [-2, 10])
        self.assertLess(abs(np.corrcoef(cars[:, 0], launches[:, 0])[0, 1]), .2)

    def test_start_placement_is_bounded_and_reset_only(self):
        env = self.make_env()
        env.reset(seed=0)
        state = env._sim.snapshot()
        for position in ([np.nan, 8], [0, np.inf], [6, 8], [0, 14]):
            with self.assertRaises(ValueError):
                env._sim.set_player_start(position)
            self.assertEqual(env._sim.snapshot(), state)
        for index in (-1, 1):
            with self.assertRaises(ValueError):
                env._sim.set_player_start([0, 8], player=index)
            self.assertEqual(env._sim.snapshot(), state)
        env._sim.set_player_start([-3, 11])
        placed = env._sim.snapshot()
        np.testing.assert_allclose(placed['players'][0]['position'], [-3, .25, 11])
        self.assertEqual(placed['bodies']['ball'], state['bodies']['ball'])
        self.assertEqual(placed['players'][0]['joint_angles'], state['players'][0]['joint_angles'])
        env.step([0, 0, 0])
        advanced = env._sim.snapshot()
        with self.assertRaises(ValueError):
            env._sim.set_player_start([0, 8])
        self.assertEqual(env._sim.snapshot(), advanced)

    def test_curriculum_endpoint_matches_court_and_easy_starts_vary(self):
        try:
            easy = self.make_env(distribution='curriculum', difficulty=0)
        except ValueError as error:
            self.fail(f'The gradual profile must be explicitly supported: {error}')
        full = self.make_env(distribution='curriculum', difficulty=1)
        court = self.make_env(distribution='court')
        starts = []
        for seed in range(20):
            a, info = full.reset(seed=seed)
            b, expected = court.reset(seed=seed)
            np.testing.assert_allclose(a, b, atol=1e-7)
            np.testing.assert_allclose(info['car_start'], expected['car_start'])
            np.testing.assert_allclose(info['feed']['position'], expected['feed']['position'])
            np.testing.assert_allclose(info['feed']['velocity'], expected['feed']['velocity'])
            _, initial = easy.reset(seed=seed)
            self.assertLess(initial['feed']['position'][2], -7)
            starts.append(initial['car_start'])
        self.assertGreater(np.ptp(np.asarray(starts)[:, 0]), .4)
        self.assertGreater(np.ptp(np.asarray(starts)[:, 1]), .4)
        _, before = easy.reset(seed=17)
        _, after = easy.reset(seed=17, options={'car_start': [-2, 10]})
        self.assertEqual(before['feed'], after['feed'])

    def test_long_feed_shaping_distinguishes_movement_before_ball_arrives(self):
        toward, away = self.make_env(distribution='court'), self.make_env(distribution='court')
        toward.reset(seed=17)
        away.reset(seed=17)
        toward_reward = toward.step([0, -.5, 0])[1]
        away_reward = away.step([0, .5, 0])[1]
        self.assertGreater(toward_reward, away_reward)

    def test_timeout_is_truncation_not_miss(self):
        env = self.make_env(max_seconds=.02)
        env.reset(seed=3)
        _, _, terminated, truncated, info = env.step([0, 0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['outcome'], 'timeout')
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])

    def test_upstream_environment_contract(self):
        from stable_baselines3.common.env_checker import check_env
        check_env(self.make_env(), warn=True)

    def test_vector_timeout_keeps_final_observation_before_reset(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec = DummyVecEnv([lambda: self.make_env(max_seconds=.02)])
        self.addCleanup(vec.close)
        vec.reset()
        obs, _, done, info = vec.step([[.2, .2, .2]])
        self.assertTrue(done[0])
        self.assertTrue(info[0]['TimeLimit.truncated'])
        self.assertEqual(info[0]['terminal_observation'][23], 1.)
        self.assertEqual(obs[0][23], 0.)

    def test_numerical_instability_is_a_defect_not_a_missed_return(self):
        env = self.make_env()
        env.reset(seed=0, options={'feed': {'position': [.8, 1., 7.], 'velocity': [0., 0., 1e9]}})
        with self.assertRaisesRegex(RuntimeError, 'numerical'):
            env.step([0, 0, 0])

    def test_physical_return_and_trace_replay(self):
        env = self.make_env(distribution='narrow')
        other = self.make_env(distribution='narrow')
        for current in (env, other):
            current.reset(seed=0)
            while True:
                _, _, done, truncated, info = current.step([0, 0, .25])
                if done or truncated:
                    break
            self.assertEqual(info['outcome'], 'success')
            self.assertTrue(info['is_success'])
            self.assertEqual(info['warnings'], 0)
        self.assertEqual(env.state, other.state)
        contact = next(e for e in env.state['contacts'] if e['surface'] == 'racket')
        self.assertGreater(contact['incoming_velocity'][2], 0)
        self.assertLess(contact['outgoing_velocity'][2], 0)


@unittest.skipIf(np is None, 'Install the optional rl dependencies')
class ReturnRulesTests(unittest.TestCase):
    def event(self, surface, tick, x=0, z=-5, duration=.004):
        return {'surface': surface, 'player': 0 if surface == 'racket' else -1,
                'first_tick': tick, 'last_tick': tick+32,
                'position': [x, .03, z], 'duration': duration, 'complete': True}

    def rules(self):
        from simtennis.returns.environment import ReturnOutcome
        return ReturnOutcome(.000125)

    def test_first_landing_line_footprint_and_own_court(self):
        for x, z, expected in [(0, -5, 'success'), (4.148, -5, 'success'),
                               (4.149, -5, 'out'), (0, -11.918, 'success'),
                               (0, -11.919, 'out'), (0, 4, 'out')]:
            rules = self.rules()
            rules.update([self.event('racket', 0), self.event('court', 200, x, z)])
            self.assertEqual(rules.outcome, expected)

    def test_net_touch_can_still_be_a_valid_rally_return(self):
        rules = self.rules()
        rules.update([self.event('racket', 0), self.event('net', 100), self.event('court', 200)])
        self.assertEqual(rules.outcome, 'success')

    def test_repeat_records_merge_brief_impact_but_reject_separate_hit(self):
        rules = self.rules()
        first = self.event('racket', 0)
        rules.update([first]); rules.update([first])
        self.assertEqual(rules.outcome, 'live')
        rules.update([first, self.event('racket', 80)])
        self.assertEqual(rules.outcome, 'live')
        rules.update([first, self.event('racket', 80), self.event('racket', 400)])
        self.assertEqual(rules.outcome, 'repeated_hit')

    def test_sustained_contact_and_second_incoming_bounce(self):
        rules = self.rules()
        rules.update([self.event('racket', 0, duration=.051)])
        self.assertEqual(rules.outcome, 'sustained_contact')
        rules = self.rules()
        first = self.event('court', 0, z=8)
        rules.update([first]); rules.update([first])
        self.assertEqual(rules.bounces, 1)
        rules.update([first, self.event('court', 500, z=10)])
        self.assertEqual(rules.outcome, 'miss')

    def test_chain_of_short_contacts_cannot_count_as_one_stroke(self):
        rules = self.rules()
        contacts = [self.event('racket', tick, duration=.001) for tick in range(0, 8000, 80)]
        for contact in contacts:
            rules.update([contact])
        rules.update([self.event('court', 9000)])
        self.assertNotEqual(rules.outcome, 'success')

    def test_active_repeat_contact_cannot_grow_beyond_stroke_span(self):
        rules = self.rules()
        first = self.event('racket', 0, duration=.001)
        first['last_tick'] = 8
        second = self.event('racket', 80, duration=.001)
        second['last_tick'] = 88
        rules.update([first, second])
        second['last_tick'] = 472
        second['duration'] = .049
        rules.update([first, second, self.event('court', 1000)])
        self.assertNotEqual(rules.outcome, 'success')


if __name__ == '__main__':
    unittest.main()
