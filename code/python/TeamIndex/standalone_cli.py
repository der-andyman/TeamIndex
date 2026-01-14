from pathlib import Path
import subprocess
import sys

def main():
    exe = Path(__file__).parent / "bin" / "standalone_runtime"
    if not exe.exists():
        raise RuntimeError(
            "Standalone executable not available. "
            "Reinstall with ENABLE_STANDALONE=ON, e.g., via\n"
            "pip install ./code --config-settings=cmake.define.ENABLE_STANDALONE=ON"
        )
    raise SystemExit(subprocess.call([str(exe)] + sys.argv[1:]))