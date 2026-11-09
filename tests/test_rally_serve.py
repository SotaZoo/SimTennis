import unittest

import numpy as np


class PhysicalServeTests(unittest.TestCase):
    def test_low_return_is_reachable_and_hands_service_back_to_policy(self):
        import math
        from simtennis.rally.serve import setup_serve, update_serve, serve_active
        from simtennis.rally.world import RallyWorld, rotation, hold_lift, player_action
        world = RallyWorld()
        setup_serve(world, 74000, server=0, deuce=True)
        target = np.array([3.124850443568251, 8.935053968321677])
        called = []
        def reply(observation):
            called.append(True)
            return np.zeros(4)
        for _ in range(300):
            p = world.state['players'][1]
            r = rotation(1)[::2]
            force = 100*(target-np.asarray(p['position'])[::2]*r)-90*np.asarray(p['velocity'])[::2]*r
            theta, _ = p['joint_angles']
            velocity = p['joint_velocities'][0]
            swing = 180*(-.7-theta)-12*velocity if world.state['time'] < 1.6299384730482698 else 6.108969297770358
            if world.rules.racket_hits[1]:
                swing = -12*velocity
            inertia = .224*math.cos(theta)**2+.0051*math.sin(theta)**2
            lift = hold_lift(world, 1)+inertia*144*(-.5637803902289432)/40
            receiver = np.clip(np.r_[force/700, swing/40, lift], -1, 1)
            world.step([player_action(world, 0, reply), receiver])
            update_serve(world)
            if world.rules.legal_returns[1] or world.rules.result != 'live':
                break
        self.assertEqual(world.rules.legal_returns[1], 1)
        self.assertTrue(called, 'The server must use its policy after the return')
        self.assertFalse(serve_active(world))
        impact = next(e for e in world.events if e['surface'] == 'racket' and e['player'] == 1)
        self.assertLess(impact['position'][1], .5)
        self.assertTrue(impact['complete'], 'Saved impact diagnostics must include the completed event')

    def test_lift_hold_is_stable_when_the_beam_aligns_with_its_axis(self):
        import math
        import xml.etree.ElementTree as ET
        from simtennis.rally.model import model_xml
        from simtennis.rally.world import RallyWorld, hold_lift
        from simtennis.returns.environment import native_module
        tree = ET.fromstring(model_xml())
        key = tree.find('keyframe/key')
        qpos = list(map(float, key.get('qpos').split()))
        qpos[2:4] = [.03, math.pi/2]
        key.set('qpos', ' '.join(map(str, qpos)))
        world = RallyWorld()
        world._sim = native_module().Simulation(ET.tostring(tree, encoding='unicode'))
        world._sim.feed_ball([0., 100., 0.], [0., 0., 0.])
        world.state = world._sim.snapshot()
        world.rules.begin_rally()
        peak = 0.
        for _ in range(100):
            actions = np.zeros((2, 4))
            actions[0, 3] = hold_lift(world, 0)
            world.step(actions)
            peak = max(peak, abs(world.state['players'][0]['joint_angles'][1]))
        self.assertLess(peak, .031)
        self.assertLess(abs(world.state['players'][0]['joint_angles'][1]), .003)

    def test_seeded_serves_land_diagonally_with_a_physical_impact(self):
        from simtennis.rally.serve import setup_serve, server_action, update_serve
        from simtennis.rally.world import RallyWorld
        for server in (0, 1):
            for deuce in (False, True):
                with self.subTest(server=server, deuce=deuce):
                    world = RallyWorld()
                    setup_serve(world, seed=73003, server=server, deuce=deuce)
                    initial = world.state
                    sign = -1 if server == 0 else 1
                    world._sim.set_player_start([5.*sign, 13.*sign], 1-server)
                    world.state = world._sim.snapshot()
                    landed = False
                    for _ in range(240):
                        actions = np.zeros((2, 4))
                        actions[server] = server_action(world)
                        world.step(actions)
                        update_serve(world)
                        if world.rules.last_hitter == server and not world.rules.serving:
                            landed = True
                        if world.rules.result != 'live':
                            break
                    self.assertTrue(landed)
                    self.assertEqual(world.rules.racket_hits[server], 1)
                    self.assertEqual(world.rules.result, 'point')
                    self.assertEqual(world.rules.reason, 'double_bounce')
                    self.assertEqual(world.rules.winner, server)
                    setup_serve(world, seed=73003, server=server, deuce=deuce)
                    self.assertEqual(world.state, initial)

    def test_north_and_south_serves_share_canonical_state_and_control(self):
        from simtennis.rally.serve import setup_serve, server_action, update_serve
        from simtennis.rally.world import RallyWorld
        south, north = RallyWorld(), RallyWorld()
        setup_serve(south, seed=73102, server=0, deuce=False)
        setup_serve(north, seed=73102, server=1, deuce=False)
        for _ in range(90):
            np.testing.assert_allclose(south.observation(0), north.observation(1), atol=1e-6)
            a, b = server_action(south), server_action(north)
            np.testing.assert_allclose(a, b, atol=1e-10)
            south.step([a, [0, 0, 0, 0]])
            north.step([[0, 0, 0, 0], b])
            update_serve(south)
            update_serve(north)

    def test_serve_controller_does_not_reset_the_ball_and_missed_attempt_faults(self):
        from simtennis.rally.serve import setup_serve, server_action, update_serve
        from simtennis.rally.world import RallyWorld
        world = RallyWorld()
        class CountFeeds:
            def __init__(self, sim):
                self.sim, self.feeds = sim, 0
            def __getattr__(self, name):
                return getattr(self.sim, name)
            def feed_ball(self, *args):
                self.feeds += 1
                return self.sim.feed_ball(*args)
        world._sim = CountFeeds(world._sim)
        setup_serve(world, seed=4)
        for _ in range(65):
            before = world._sim.snapshot()
            server_action(world)
            self.assertEqual(world._sim.snapshot(), before)
            world.step([[0, 0, 0, 0], [0, 0, 0, 0]])
            update_serve(world)
            if world.rules.result != 'live':
                break
        self.assertEqual(world._sim.feeds, 1)
        self.assertEqual(world.rules.result, 'fault')
        self.assertEqual(world.rules.reason, 'missed_serve')

    def test_illegal_stance_is_a_fault_before_swinging(self):
        from simtennis.rally.serve import setup_serve, update_serve
        from simtennis.rally.world import RallyWorld
        world = RallyWorld()
        setup_serve(world, seed=4)
        world._sim.set_player_start([1., 11.], 0)
        world.state = world._sim.snapshot()
        update_serve(world)
        self.assertEqual(world.rules.result, 'fault')
        self.assertEqual(world.rules.reason, 'service_stance')


if __name__ == '__main__':
    unittest.main()
