"""Native replay must substantiate point results, score changes and set completion."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class RallyEvaluationTests(unittest.TestCase):
    def replay(self, report):
        from simtennis.rally.evaluation import replay
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            path.write_text(json.dumps(report))
            result = replay(path)
            return result, hashlib.sha256(path.read_bytes()).hexdigest()

    def short_set(self):
        from simtennis.rally.evaluation import play_set
        zero = lambda obs: np.zeros(4)
        return play_set([zero, zero], 81000, max_points=1, max_seconds=.04)

    def test_full_native_set_replays_and_first_fault_does_not_award_point(self):
        from simtennis.rally.evaluation import play_set
        from simtennis.rally.world import player_action
        zero = lambda obs: np.zeros(4)
        misses_serve = lambda obs: np.zeros(4)

        def driver(world, player, policy):
            # A deliberately weak control driver never swings on service.
            # All faults, bounces and winners still come from native physics.
            if player == world.rules.server and policy is misses_serve:
                return np.zeros(4)
            return player_action(world, player, policy)

        with patch('simtennis.rally.evaluation.player_action', driver):
            report = play_set([zero, misses_serve], 81000, max_points=32, max_seconds=8.)
        self.assertTrue(report['complete'])
        self.assertEqual(report['score']['games'], [6, 0])
        first_faults = [a for a in report['attempts'] if a['result'] == 'fault' and a['attempt'] == 1]
        self.assertTrue(first_faults)
        for attempt in first_faults:
            self.assertEqual(attempt['score_before'], attempt['score_after'])
        for attempt in report['attempts']:
            self.assertEqual(attempt['rally_hits'], [0, 0])
            server = attempt['setup']['server']
            # Missed first and second attempts contribute no hit to subtract.
            expected = 0 if attempt['outcome'] == 'missed_serve' else 1
            self.assertEqual(attempt['hits'][server], expected)
        self.assertTrue(any(e['hits'] == [0, 0] and e['attempt'] == 2 for e in report['episodes']))
        self.assertTrue(all(e['rally_hits'] == [0, 0] for e in report['episodes']))
        checked, digest = self.replay(report)
        self.assertEqual(checked['report_sha256'], digest)
        self.assertTrue(checked['verified'])
        self.assertTrue(checked['set_complete'])
        self.assertEqual(checked['points'], report['points'])

    def test_replay_rejects_fabricated_completion_winners_and_score(self):
        report = self.short_set()
        self.assertFalse(report['complete'])
        mutations = {
            'completion': lambda r: r.update(complete=True),
            'set winner': lambda r: r['score'].update(winner=1, games=[0, 6]),
            'episode winner': lambda r: r['episodes'][0].update(winner=1),
            'episode result': lambda r: r['episodes'][0].update(outcome='double_fault'),
            'first score': lambda r: r['attempts'][0]['score_before'].update(games=[900, 900]),
            'last score': lambda r: r['attempts'][0]['score_after'].update(points=[1, 0]),
            'point count': lambda r: r.update(points=900),
            'attempt rally hits': lambda r: r['attempts'][0].update(rally_hits=[1, 0]),
            'episode rally hits': lambda r: r['episodes'][0].update(rally_hits=[1, 0]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaises((AssertionError, ValueError)):
                    self.replay(changed)

    def test_replay_checks_ball_velocity_player_body_pose_and_player_count(self):
        report = self.short_set()
        for field in ('ball_velocity', 'racket_pose', 'player_count'):
            with self.subTest(field=field):
                changed = copy.deepcopy(report)
                state = changed['attempts'][0]['frames'][-1]['state']
                if field == 'ball_velocity':
                    state['ball_velocity'][0] += 1
                elif field == 'racket_pose':
                    state['players'][1]['bodies']['racket']['position'][1] += .1
                else:
                    state['players'].pop()
                with self.assertRaises((AssertionError, ValueError)):
                    self.replay(changed)

    def test_empty_and_initial_frame_only_reports_are_not_verified(self):
        report = self.short_set()
        for kind in ('attempts', 'frames', 'initial_only'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(report)
                if kind == 'attempts':
                    changed['attempts'] = []
                elif kind == 'frames':
                    changed['attempts'][0]['frames'] = []
                else:
                    changed['attempts'][0]['frames'] = changed['attempts'][0]['frames'][:1]
                with self.assertRaises((AssertionError, ValueError)):
                    self.replay(changed)

    def test_point_subset_verification_does_not_claim_a_complete_set(self):
        from simtennis.rally.environment import RallyConfig
        from simtennis.rally.evaluation import evaluate_points
        zero = lambda obs: np.zeros(4)
        report = evaluate_points(zero, [zero], [81000, 81001],
                                 config=RallyConfig(feed_probability=1, max_seconds=.04), trace_count=1)
        self.assertIn('score_after', report['attempts'][0])
        checked, digest = self.replay(report)
        self.assertEqual(checked['report_sha256'], digest)
        self.assertFalse(checked['set_complete'])
        self.assertEqual(checked['points'], 1)
        for record in report['attempts'] + report['episodes']:
            self.assertEqual(record['rally_hits'], record['hits'], 'Feeds have no service contact')
        for field in ('learner_rally_hits', 'opponent_rally_hits'):
            changed = copy.deepcopy(report)
            changed[field] += 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.replay(changed)
        legacy = copy.deepcopy(report)
        for record in legacy['attempts'] + legacy['episodes']:
            del record['rally_hits']
        del legacy['learner_rally_hits'], legacy['opponent_rally_hits']
        legacy_checked, _ = self.replay(legacy)
        self.assertEqual(legacy_checked['points'], 1)
        self.assertTrue(legacy_checked['verified'])
        report['complete'] = True
        report['score'] = {'winner': 1, 'games': [0, 6]}
        with self.assertRaises((AssertionError, ValueError)):
            self.replay(report)

    def test_independent_point_scores_keep_candidate_identity_across_sides(self):
        from simtennis.rally.environment import RallyConfig
        from simtennis.rally.evaluation import evaluate_points
        zero = lambda obs: np.zeros(4)
        report = evaluate_points(zero, [zero], range(81000, 81004),
                                 config=RallyConfig(max_seconds=8.), trace_count=4)
        self.assertEqual({e['learner'] for e in report['episodes']}, {0, 1})
        self.assertEqual(report['learner_hits'], 2, 'Raw hits retain scripted serves')
        self.assertEqual(report['learner_rally_hits'], 0)
        self.assertEqual(report['opponent_rally_hits'], 0)
        for episode in report['episodes']:
            self.assertIn(episode['winner'], (0, 1))
            before, after = episode['score_before'], episode['score_after']
            self.assertEqual(before['games'], [0, 0])
            self.assertEqual(before['points'], [0, 0])
            self.assertEqual(before['ends'], episode['learner'])
            self.assertEqual(before['server'], episode['server'] ^ episode['learner'])
            expected = [1, 0] if episode['winner'] == episode['learner'] else [0, 1]
            self.assertEqual(after['points'], expected)
        checked, _ = self.replay(report)
        self.assertEqual(checked['points'], 4)
        self.assertFalse(checked['set_complete'])

    def test_native_return_contact_remains_in_rally_counts_and_replays(self):
        from simtennis.rally.environment import RallyConfig
        from simtennis.rally.evaluation import evaluate_points
        from tests.test_rally_practice import physical_test_actor
        report = evaluate_points(physical_test_actor, [physical_test_actor], [86000],
                                 config=RallyConfig(max_seconds=8.), trace_count=1)
        episode = report['episodes'][0]
        self.assertEqual(episode['hits'], [1, 1])
        self.assertEqual(episode['rally_hits'], [0, 1])
        self.assertEqual(episode['returns'], [0, 1])
        self.assertEqual(report['learner_rally_hits'], 0)
        self.assertEqual(report['opponent_rally_hits'], 1)
        checked, _ = self.replay(report)
        self.assertTrue(checked['verified'])


if __name__ == '__main__':
    unittest.main()
