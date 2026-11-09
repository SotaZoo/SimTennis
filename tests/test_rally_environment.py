import unittest

import numpy as np


class RallyPlantTests(unittest.TestCase):
    def test_ignored_action_has_no_cost_at_opponent_contact(self):
        from simtennis.rally.environment import RallyEnv
        opponent = lambda observation: np.zeros(4)
        idle, pressing = RallyEnv([opponent]), RallyEnv([opponent])
        self.addCleanup(idle.close)
        self.addCleanup(pressing.close)
        for env in (idle, pressing):
            env.reset(seed=23, options={'learner': 1, 'server': 0, 'deuce': True})
        for _ in range(100):
            self.assertFalse(idle.world.incoming(1))
            left = idle.step(np.zeros(4))
            right = pressing.step(np.full(4, .5))
            np.testing.assert_array_equal(left[0], right[0])
            self.assertAlmostEqual(left[1], right[1], places=12)
            if idle.world.incoming(1):
                self.assertEqual(pressing.applied_steps, 0)
                break
        else:
            self.fail('Physical serve never handed control to the receiver')

    def test_same_efforts_produce_mirrored_players(self):
        from simtennis.rally.model import model_xml
        from simtennis.returns.environment import native_module
        sim = native_module().Simulation(model_xml())
        sim.feed_ball([0, 10, 0], [0, 0, 0])
        sim.set_effort(0, [.8, 1.2], [140, -210], player=0)
        sim.set_effort(0, [.8, 1.2], [-140, 210], player=1)
        sim.step(160)
        state = sim.snapshot()
        a, b = state['players']
        np.testing.assert_allclose(a['position'], np.array(b['position'])*[-1, 1, -1], atol=1e-12)
        np.testing.assert_allclose(a['velocity'], np.array(b['velocity'])*[-1, 1, -1], atol=1e-12)
        np.testing.assert_allclose(a['joint_angles'], b['joint_angles'], atol=1e-12)
        self.assertEqual(len(a['joint_angles']), 2)
        self.assertFalse(state['warnings'] or state['faulted'])

    def test_feed_observation_preserves_return_policy_inputs(self):
        from simtennis.rally.world import RallyWorld
        from simtennis.returns.environment import ReturnEnv
        previous = ReturnEnv(distribution='court')
        self.addCleanup(previous.close)
        world = RallyWorld()
        expected, info = previous.reset(seed=19)
        world.reset_feed(seed=19)
        np.testing.assert_allclose(world.observation(0)[:27], expected, atol=1e-7)
        self.assertEqual(world.setup['feed'], info['feed'])
        world.step([[0, 0, 0, -.5], [0, 0, 0, .5]])
        self.assertLess(world.state['players'][0]['joint_angles'][1], 0)
        self.assertGreater(world.state['players'][1]['joint_angles'][1], 0)

    def test_world_replay_mirror_and_invalid_action_atomicity(self):
        from simtennis.rally.world import RallyWorld
        south, north = RallyWorld(), RallyWorld()
        south.reset_feed(seed=23, receiver=0)
        north.reset_feed(seed=23, receiver=1)
        np.testing.assert_array_equal(south.observation(0), north.observation(1))
        for _ in range(20):
            actions = [[.2, -.1, .05, -.1], [-.1, .1, .1, .2]]
            south.step(actions)
            north.step(actions[::-1])
            np.testing.assert_allclose(south.observation(0), north.observation(1), atol=1e-6)
        before = south.state
        for actions in ([[0, 0, 0, 0]], [[0, 0, 0, 0], [0, float('nan'), 0, 0]], [[2, 0, 0, 0], [0, 0, 0, 0]]):
            with self.assertRaises(ValueError):
                south.step(actions)
            self.assertEqual(south._sim.snapshot(), before)

    def test_lift_reaches_low_balls_and_ground_contact_limits_penetration(self):
        from simtennis.rally.world import RallyWorld, recovery_action
        world = RallyWorld()
        world.reset_feed(seed=0)
        low = float('inf')
        for _ in range(60):
            action = recovery_action(world, 0, angle=0)
            action[3] = -.5
            world.step([action, recovery_action(world, 1)])
            low = min(low, world.stringbed(0)[1])
        self.assertLess(low, .3)
        self.assertGreater(world.stringbed(0)[1], .01)
        self.assertFalse(world.state['warnings'] or world.state['faulted'])


if __name__ == '__main__':
    unittest.main()
