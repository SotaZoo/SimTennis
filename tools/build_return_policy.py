"""Build the Python connection and native checks without a Godot dependency."""
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    deps = ROOT / '.scratch/deps'
    if not (deps / 'mujoco/include/mujoco/mujoco.h').exists():
        raise SystemExit('MuJoCo dependencies missing. Run tools/build.py deps first.')
    build = ROOT / '.scratch/rl-build'
    configure = ['cmake', '-S', str(ROOT), '-B', str(build),
                 '-DCMAKE_BUILD_TYPE=Release', '-DTENNIS_BUILD_GODOT=OFF',
                 '-DTENNIS_BUILD_PYTHON=ON', f'-DPython_EXECUTABLE={sys.executable}']
    if platform.system() == 'Darwin':
        configure.append(f'-DCMAKE_OSX_ARCHITECTURES={platform.machine()}')
    subprocess.run(configure, check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', '4'], check=True)


if __name__ == '__main__':
    main()
