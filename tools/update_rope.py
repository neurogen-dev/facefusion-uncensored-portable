import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_cu124.txt", "--default-timeout", "100"])

install_requirements()
pass
