"""Verify or view a recorded SimTennis contest."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['replay', 'view'])
    parser.add_argument('recording', type=Path)
    parser.add_argument('--check', action='store_true', help='Validate viewer state without rendering')
    args = parser.parse_args()
    if args.command == 'replay':
        from .rally.evaluation import replay
        print(json.dumps(replay(args.recording)))
    else:
        from .rally.view import view_replay
        view_replay(args.recording, check=args.check)


if __name__ == '__main__':
    main()
