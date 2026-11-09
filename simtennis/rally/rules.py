"""Pure contact decisions and one-set scoring for the two-car experiment.

Ported from godot/tennis_point.gd and godot/tennis_score.gd. Standard scoring,
service and bounce rules follow the ITF 2026 Rules of Tennis, rules 5–6, 10–12,
14, 17, 19–24: https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf

The car embodiment has no human feet. The caller checks its service stance and
declares an attempted serve that missed. Body/net-player/permanent-fixture and
over-net racket faults are not observable here. Landing uses a radius-0.033 m
disk, not a deformed contact patch. Continuous racket contact is approximated
by a <=20 ms gap and <=50 ms total cluster; longer contact is a carry proxy.
These mechanical thresholds do not infer deliberate double hits or carrying.
"""
from dataclasses import dataclass, field
import math


class PointRules:
    """Consume native contact snapshots without changing physical state.

    Contacts may be repeated and arrive in any order within a snapshot. Native
    snapshots expose onsets immediately; a high-water mark deduplicates history
    with constant memory. Updating an existing racket record can still expose
    sustained contact. `legal_returns` counts a real rally shot once, when it
    lands in or is volleyed by the opponent. It excludes serves and feed setup.
    """
    radius = .033
    half_width = 4.115
    half_length = 11.885
    service_length = 6.4
    centre_line_half_width = .025

    def __init__(self, timestep=.000125):
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError('The physical timestep must be positive and finite')
        self.timestep = timestep
        self.begin(0)

    def begin(self, server, deuce=True):
        if server not in (0, 1):
            raise ValueError('The server must be physical slot 0 or 1')
        self.server = server
        self.deuce = deuce
        self.result = 'live'
        self.winner = -1
        self.reason = ''
        self.last_hitter = -1
        self.bounces = 0
        self.serving = True
        self.awaiting_first_bounce = False
        self.legal_returns = [0, 0]
        self.racket_hits = [0, 0]
        self.first_racket_tick = -1
        self.last_racket_tick = -1
        self._serve_touched_net = False
        self._return_candidate = False
        self._return_recorded = False
        self._onset_tick = -1
        self._onsets_at_tick = set()

    def begin_rally(self, last_hitter=1):
        """Begin a declared feed without inventing an opponent racket impact."""
        if last_hitter not in (0, 1):
            raise ValueError('The previous hitter must be physical slot 0 or 1')
        self.begin(last_hitter)
        self.serving = False
        self.last_hitter = last_hitter

    def update(self, contacts):
        for event in sorted(contacts, key=lambda e: (e['first_tick'], e['surface'], e['player'])):
            if self.result != 'live':
                break
            tick = event['first_tick']
            identity = (event['surface'], event['player'])
            new = tick > self._onset_tick or (tick == self._onset_tick and identity not in self._onsets_at_tick)
            if new:
                if tick > self._onset_tick:
                    self._onset_tick = tick
                    self._onsets_at_tick.clear()
                self._onsets_at_tick.add(identity)
                if event['surface'] == 'racket':
                    self._racket(event)
                elif event['surface'] == 'court':
                    self._bounce(event['position'])
                elif event['surface'] == 'net' and self.serving and self.awaiting_first_bounce:
                    self._serve_touched_net = True
            elif (event['surface'] == 'racket' and event['player'] == self.last_hitter
                  and self.first_racket_tick >= 0 and tick >= self.first_racket_tick):
                self._extend_cluster(event)
        return self.result

    def _extend_cluster(self, event):
        self.last_racket_tick = max(self.last_racket_tick, event['last_tick'])
        span = (self.last_racket_tick - self.first_racket_tick) * self.timestep
        if span > .05 + 1e-9 or event['duration'] > .05 + 1e-9:
            self._illegal_hit('sustained_contact')

    def _illegal_hit(self, reason):
        if self.serving:
            self._finish('fault', reason)
        else:
            self._finish('point', reason, 1 - self.last_hitter)

    def _racket(self, event):
        player = event['player']
        if player not in (0, 1):
            return
        if self.last_hitter == -1:
            if player == self.server:
                self.awaiting_first_bounce = True
                self._accept_hit(event)
            return
        if player == self.last_hitter:
            gap = (event['first_tick'] - self.last_racket_tick) * self.timestep
            if self.first_racket_tick < 0 or self.bounces or gap > .02 + 1e-9:
                self._illegal_hit('racket_rehit')
            else:
                self._extend_cluster(event)
            return
        if self.serving:
            if self._serve_touched_net:
                self._finish('let', 'service_let')
            else:
                self._finish('point', 'receiver_volley', self.server)
            return
        self._record_return()
        self._accept_hit(event)

    def _accept_hit(self, event):
        self.last_hitter = event['player']
        self.bounces = 0
        self.first_racket_tick = event['first_tick']
        self.last_racket_tick = event['last_tick']
        self.racket_hits[self.last_hitter] += 1
        self._return_candidate = not self.serving
        self._return_recorded = False
        self._extend_cluster(event)

    def _record_return(self):
        if self._return_candidate and not self._return_recorded:
            self.legal_returns[self.last_hitter] += 1
            self._return_recorded = True

    def _bounce(self, position):
        if self.last_hitter == -1:
            return
        self.bounces += 1
        if self.bounces >= 2:
            self._finish('point', 'double_bounce', self.last_hitter)
        elif self.serving:
            if not self._inside_service_box(position):
                self._finish('fault', 'serve_out')
            elif self._serve_touched_net:
                self._finish('let', 'service_let')
            else:
                self.serving = False
                self.awaiting_first_bounce = False
        elif not self._inside_opponent_court(position):
            self._finish('point', 'rally_out', 1 - self.last_hitter)
        else:
            self._record_return()

    def missed_serve(self):
        if self.result == 'live' and self.serving and self.last_hitter == -1:
            self._finish('fault', 'missed_serve')

    def service_fault(self, reason='service_stance'):
        if self.result == 'live' and self.serving:
            self._finish('fault', reason)

    def _finish(self, result, reason, winner=-1):
        self.result, self.reason, self.winner = result, reason, winner
        self.serving = False
        self.awaiting_first_bounce = False

    def _inside_service_box(self, position):
        direction = -1 if self.server == 0 else 1
        xside = direction * (1 if self.deuce else -1)
        return self._disk_in_rectangle(xside * position[0], direction * position[2],
                                       -self.centre_line_half_width, self.half_width, self.service_length)

    def _inside_opponent_court(self, position):
        direction = -1 if self.last_hitter == 0 else 1
        return self._disk_in_rectangle(position[0], direction * position[2],
                                       -self.half_width, self.half_width, self.half_length)

    def _disk_in_rectangle(self, x, z, left, right, depth):
        dx = x - min(right, max(left, x))
        dz = z - min(depth, max(0., z))
        return dx * dx + dz * dz <= self.radius * self.radius + 1e-12


@dataclass
class SetScore:
    """Standard advantage games and a seven-point tiebreak at six games all.

    Scores and server identify agents; `slot` maps identities to physical ends.
    A completed set holds its final mapping because no next point is prepared.
    Service attempts and physical point decisions belong to the caller.
    """
    points: list = field(default_factory=lambda: [0, 0])
    games: list = field(default_factory=lambda: [0, 0])
    server: int = 0
    winner: int = -1
    tiebreak: bool = False
    ends: int = 0
    points_played: int = 0
    _tiebreak_first_server: int = 0

    def slot(self, identity):
        return identity ^ self.ends

    def deuce_court(self):
        return sum(self.points) % 2 == 0

    def point_text(self, player):
        if self.tiebreak:
            return str(self.points[player])
        return 'AD' if self.points[player] >= 4 else ('0', '15', '30', '40')[self.points[player]]

    def award(self, player):
        if self.winner != -1 or player not in (0, 1):
            return False
        old_games = sum(self.games)
        self.points[player] += 1
        self.points_played += 1
        other = 1 - player
        if self.tiebreak:
            if self.points[player] >= 7 and self.points[player] - self.points[other] >= 2:
                self.games[player] += 1
                self.winner = player
            else:
                self.server = (self._tiebreak_first_server + (sum(self.points) + 1) // 2) % 2
        elif self.points[player] >= 4 and self.points[player] - self.points[other] >= 2:
            self.games[player] += 1
            self.points = [0, 0]
            self.server = 1 - self.server
            if self.games[player] >= 6 and self.games[player] - self.games[other] >= 2:
                self.winner = player
            elif self.games == [6, 6]:
                self.tiebreak = True
                self._tiebreak_first_server = self.server
        elif self.points[0] >= 3 and self.points[0] == self.points[1]:
            self.points = [3, 3]
        if self.winner == -1:
            if sum(self.games) != old_games and sum(self.games) % 2:
                self.ends ^= 1
            elif self.tiebreak and sum(self.points) > 0 and sum(self.points) % 6 == 0:
                self.ends ^= 1
        return True
