"""Verify recorded physics, then present the scored set in Godot."""
import argparse
from pathlib import Path
import shutil
import subprocess


def view_replay(path, check=False, capture=None):
    from .evaluation import replay
    from tools.build import ROOT, GODOT, check_godot_output

    path = Path(path).resolve()
    replay(path)
    folder = ROOT / '.scratch/rally/viewer'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('court.gd', 'car_visual.gd', 'player_appearance.gd', 'ball_visual.gd',
                 'court_materials.gd', 'return_replay.gd', 'rally_replay.gd',
                 'tests/rally_replay.gd'):
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'godot' / name, destination)
    (folder / 'models').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'models/car_match.xml', folder / 'models/car_match.xml')
    config = (ROOT / 'godot/project.godot').read_text().replace('run/main_scene="res://main.tscn"\n', '')
    (folder / 'project.godot').write_text(config)
    command = [str(GODOT), '--path', str(folder), '--script',
               'res://tests/rally_replay.gd' if check else 'res://rally_replay.gd', '--', str(path)]
    if capture is not None:
        capture = Path(capture).resolve()
        capture.parent.mkdir(parents=True, exist_ok=True)
        command.append('--capture=' + str(capture))
    if check:
        command.insert(1, '--headless')
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=90)
        print(completed.stdout, end='')
        check_godot_output(completed.stdout + completed.stderr, (
            'PASS: both recorded players, identity colors and score transitions',
            'PASS: replay pause, point navigation, speed, help, camera, zoom and final-frame hold'))
    else:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--check', action='store_true', help='Check poses, scoring and controls without rendering')
    parser.add_argument('--capture', type=Path, help='Save a rendered frame, then close')
    args = parser.parse_args()
    view_replay(args.path, args.check, args.capture)


if __name__ == '__main__':
    main()
