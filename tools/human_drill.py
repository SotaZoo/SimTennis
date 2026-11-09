"""Build the native arm bridge and open the human-model drill."""
from pathlib import Path
import argparse
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build import GODOT, check_godot_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    build = ROOT / '.scratch/human-build'
    command = ['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
               '-DTENNIS_BUILD_GODOT=ON', '-DTENNIS_BUILD_PYTHON=OFF', '-DGODOTCPP_TARGET=template_debug']
    if platform.system() == 'Darwin':
        command += [f'-DCMAKE_OSX_ARCHITECTURES={platform.machine()}']
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'tennis', '--parallel', '4'], check=True)
    for name in ['models', 'assets']:
        shutil.copytree(ROOT/name, ROOT/'godot'/name, dirs_exist_ok=True)
    result = subprocess.run([str(GODOT), '--headless', '--path', str(ROOT/'godot'), '--editor', '--import'],
                            text=True, capture_output=True, check=True)
    check_godot_output(result.stdout + result.stderr)
    command = [str(GODOT), '--path', str(ROOT/'godot'), '--script', 'res://human_drill.gd']
    if args.check:
        command.insert(1, '--headless')
        command += ['--quit-after', '120', '--', '--check']
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=60)
        check_godot_output(result.stdout + result.stderr, ('PASS: human character loaded;',))
        print(result.stdout)
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
