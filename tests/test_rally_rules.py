"""Contact and scoring gold cases from the ITF's standard one-set format."""
import unittest

from simtennis.rally.rules import PointRules, SetScore


def contact(surface, tick, player=-1, position=(0, .033, 0), ticks=1):
    return {'surface': surface, 'first_tick': tick, 'last_tick': tick + ticks,
            'player': player, 'position': position, 'duration': ticks * .000125,
            'complete': True}


class PointRulesTests(unittest.TestCase):
    def serve(self, server=0, deuce=True):
        point = PointRules()
        point.begin(server, deuce)
        point.update([contact('racket', 0, server)])
        return point

    def rally(self):
        point = self.serve()
        point.update([contact('court', 1000, position=(-2, .033, -4)),
                      contact('racket', 2000, 1)])
        return point

    def test_each_service_box_requires_the_diagonal_first_bounce(self):
        for server in (0, 1):
            for deuce in (True, False):
                with self.subTest(server=server, deuce=deuce):
                    direction = -1 if server == 0 else 1
                    x = direction * (1 if deuce else -1) * 2
                    point = self.serve(server, deuce)
                    point.update([contact('court', 1000, position=(x, .033, direction * 4))])
                    self.assertEqual(point.result, 'live')
                    self.assertFalse(point.serving)
                    self.assertEqual(point.racket_hits[server], 1)
                    self.assertEqual(point.legal_returns, [0, 0])
                    point = self.serve(server, deuce)
                    point.update([contact('court', 1000, position=(-x, .033, direction * 4))])
                    self.assertEqual(point.result, 'fault')

    def test_line_overlap_uses_a_disk_including_corners(self):
        for position in ((-4.145, .033, -4), (.055, .033, -4), (-2, .033, -6.433)):
            point = self.serve()
            point.update([contact('court', 1000, position=position)])
            self.assertEqual(point.result, 'live')
        for position in ((-4.150, .033, -4), (.060, .033, -4), (-2, .033, -6.440),
                         (-4.145, .033, -6.430)):
            point = self.serve()
            point.update([contact('court', 1000, position=position)])
            self.assertEqual(point.result, 'fault')

    def test_service_net_touch_waits_for_correct_bounce_or_receiver(self):
        for x, outcome in ((-2, 'let'), (2, 'fault')):
            point = self.serve()
            point.update([contact('net', 100)])
            self.assertEqual(point.result, 'live')
            point.update([contact('court', 1000, position=(x, .033, -4))])
            self.assertEqual(point.result, outcome)
        point = self.serve()
        point.update([contact('net', 100), contact('racket', 1000, 1)])
        self.assertEqual(point.result, 'let')

    def test_receiver_cannot_volley_serve_but_can_volley_rally(self):
        point = self.serve()
        point.update([contact('racket', 1000, 1)])
        self.assertEqual((point.result, point.winner), ('point', 0))
        point = self.rally()
        point.update([contact('racket', 3000, 0)])
        self.assertEqual((point.result, point.last_hitter), ('live', 0))
        self.assertEqual(point.legal_returns, [0, 1])

    def test_first_good_bounce_then_second_anywhere_wins_for_hitter(self):
        point = self.rally()
        first = contact('court', 3000, position=(2, .033, 4))
        point.update([first])
        point.update([first])
        self.assertEqual(point.bounces, 1)
        self.assertEqual(point.legal_returns, [0, 1])
        point.update([contact('court', 5000, position=(20, .033, 20))])
        self.assertEqual((point.result, point.winner), ('point', 1))
        point.update([contact('racket', 6000, 0)])
        self.assertEqual((point.result, point.winner), ('point', 1))

    def test_out_or_own_half_first_bounce_loses_for_hitter(self):
        for position in ((4.150, .033, 4), (2, .033, -4), (2, .033, 12)):
            point = self.rally()
            point.update([contact('court', 3000, position=position)])
            self.assertEqual((point.result, point.winner), ('point', 0))
            self.assertEqual(point.legal_returns, [0, 0])

    def test_continuous_contact_cluster_counts_once_and_later_rehit_loses(self):
        point = self.rally()
        second = contact('racket', 2080, 1, ticks=40)
        point.update([second])
        point.update([second])
        self.assertEqual(point.racket_hits, [1, 1])
        self.assertEqual((point.first_racket_tick, point.last_racket_tick), (2000, 2120))
        point.update([contact('racket', 2400, 1)])
        self.assertEqual((point.result, point.winner), ('point', 0))

    def test_ongoing_carry_and_repeated_short_touches_cannot_extend_cluster(self):
        for records in ([contact('racket', 2000, 1, ticks=401)],
                        [contact('racket', 2120, 1, ticks=40),
                         contact('racket', 2240, 1, ticks=40),
                         contact('racket', 2360, 1, ticks=41)]):
            point = self.rally()
            point.update(records)
            self.assertEqual((point.result, point.winner), ('point', 0))
        point = self.serve()
        point.update([contact('racket', 0, 0, ticks=401)])
        self.assertEqual(point.result, 'fault')
        self.assertFalse(point.awaiting_first_bounce)
        point = PointRules()
        point.update([contact('racket', 0, 0, ticks=401)])
        self.assertEqual(point.result, 'fault')
        self.assertFalse(point.awaiting_first_bounce)

    def test_events_are_adjudicated_at_onset_in_order_not_when_completed(self):
        point = self.serve()
        net = contact('net', 100)
        net['complete'] = False
        point.update([contact('court', 1000, position=(-2, .033, -4)), net])
        self.assertEqual(point.result, 'let')

    def test_toss_without_strike_is_not_live_and_reset_clears_fault(self):
        point = PointRules()
        point.begin(0)
        point.update([contact('court', 0), contact('racket', 100, 1)])
        self.assertEqual(point.last_hitter, -1)
        point.missed_serve()
        self.assertEqual(point.result, 'fault')
        point.begin(1, False)
        point.update([contact('racket', 0, 1),
                      contact('court', 1000, position=(-2, .033, 4))])
        self.assertEqual(point.result, 'live')
        self.assertEqual(point.racket_hits, [0, 1])

    def test_rally_feed_does_not_fabricate_an_opponent_racket_hit(self):
        point = PointRules()
        point.begin_rally(last_hitter=1)
        point.update([contact('court', 0, position=(2, .033, 4)),
                      contact('racket', 1000, 0),
                      contact('court', 2000, position=(2, .033, -4))])
        self.assertEqual(point.racket_hits, [1, 0])
        self.assertEqual(point.legal_returns, [1, 0])

    def test_old_records_cannot_be_reintroduced_as_new_hits(self):
        point = self.rally()
        point.update([contact('court', 3000, position=(2, .033, 4))])
        point.update([contact('racket', 2000, 1)])
        self.assertEqual(point.result, 'live')
        self.assertEqual(point.racket_hits, [1, 1])
        self.assertEqual(point.bounces, 1)


class SetScoreTests(unittest.TestCase):
    def game(self, score, winner):
        for _ in range(4):
            self.assertTrue(score.award(winner))

    def test_deuce_advantage_service_half_and_game_switch(self):
        score = SetScore()
        for winner in (0, 1, 0, 1, 0, 1):
            score.award(winner)
        self.assertEqual([score.point_text(p) for p in (0, 1)], ['40', '40'])
        self.assertTrue(score.deuce_court())
        score.award(0)
        self.assertEqual(score.point_text(0), 'AD')
        self.assertFalse(score.deuce_court())
        score.award(1)
        self.assertEqual(score.points, [3, 3])
        score.award(1)
        score.award(1)
        self.assertEqual((score.games, score.points, score.server, score.ends),
                         ([0, 1], [0, 0], 1, 1))
        self.assertEqual((score.slot(0), score.slot(1)), (1, 0))

    def test_set_requires_six_games_and_two_game_margin(self):
        for scores, expected in (([0] * 6, [6, 0]), ([0, 1] * 5 + [0, 0], [7, 5])):
            score = SetScore()
            for winner in scores:
                self.game(score, winner)
            self.assertEqual((score.games, score.winner), (expected, 0))
            self.assertFalse(score.award(1))
            self.assertEqual(score.points_played, len(scores) * 4)

    def test_standard_tiebreak_service_order_ends_and_extended_win(self):
        score = SetScore()
        for game in range(12):
            self.game(score, game % 2)
        self.assertEqual((score.games, score.tiebreak), ([6, 6], True))
        before = score.ends
        for number, server in enumerate((0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0)):
            self.assertEqual(score.server, server)
            self.assertEqual(score.deuce_court(), number % 2 == 0)
            score.award(number % 2)
            self.assertEqual(score.ends, before ^ (((number + 1) // 6) % 2))
        self.assertEqual((score.points, score.winner), ([6, 6], -1))
        score.award(1)
        self.assertEqual(score.winner, -1)
        score.award(1)
        self.assertEqual((score.points, score.games, score.winner), ([6, 8], [6, 7], 1))

    def test_each_odd_game_changes_identity_to_slot_mapping(self):
        score = SetScore(server=1)
        for number in range(1, 6):
            old = score.ends
            self.game(score, 0)
            self.assertEqual(score.ends, old ^ (number % 2))
            self.assertEqual(score.server, 1 ^ (number % 2))


if __name__ == '__main__':
    unittest.main()
