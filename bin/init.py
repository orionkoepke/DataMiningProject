#!/usr/bin/env python3
"""
Initialize script for algorithmic trading strategies project.
Creates or recreates a Python virtual environment in the project root
and installs required dependencies.
"""

import sys
import subprocess
import shutil
from pathlib import Path


def get_venv_pip(venv_path: Path) -> Path:
    """Get the path to pip in the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def create_virtual_environment():
    """Create or recreate a Python virtual environment in the project root."""
    # Get the project root directory (parent of bin folder)
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "venv"
    
    # Remove existing virtual environment if it exists
    if venv_path.exists():
        print(f"Removing existing virtual environment at: {venv_path}")
        try:
            shutil.rmtree(venv_path)
            print("✓ Existing virtual environment removed")
        except Exception as e:
            print(f"Error removing existing virtual environment: {e}")
            sys.exit(1)
    
    # Create virtual environment
    print(f"Creating virtual environment at: {venv_path}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("✓ Virtual environment created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    
    # Install dependencies
    print("\nInstalling dependencies...")
    pip_path = get_venv_pip(venv_path)
    dependencies = ["pytz", "alpaca-py", "pandas", "pandas_market_calendars", "pylint"]
    
    try:
        for dep in dependencies:
            print(f"  Installing {dep}...")
            subprocess.run([str(pip_path), "install", dep], check=True)
        print("✓ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error installing dependencies: {e}")
        sys.exit(1)
    
    # Print activation instructions
    print(f"\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")


if __name__ == "__main__":
    create_virtual_environment()

