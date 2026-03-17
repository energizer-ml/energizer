#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Update the package version across various files.")
    parser.add_argument("new_version", help="The new version string, e.g., 0.1.5.1")
    args = parser.parse_args()
    
    new_version = args.new_version
    
    # 1. Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        new_content = re.sub(r'(?m)^version\s*=\s*".*"$', f'version = "{new_version}"', content, count=1)
        if content == new_content:
            print("Warning: Could not find version string in pyproject.toml to update.")
        else:
            pyproject_path.write_text(new_content)
            print(f"Updated pyproject.toml to version {new_version}")
    else:
        print("Error: pyproject.toml not found.")
        sys.exit(1)
        
    # 2. Update energizer/__init__.py
    init_path = Path("energizer/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        new_content = re.sub(
            r'(?m)^__version__\s*=\s*".*"$', 
            f'__version__ = "{new_version}"', 
            content, 
            count=1
        )
        if content == new_content:
            print("Warning: Could not find __version__ in energizer/__init__.py to update.")
        else:
            init_path.write_text(new_content)
            print(f"Updated energizer/__init__.py to version {new_version}")
    else:
        print("Error: energizer/__init__.py not found.")
        sys.exit(1)

    # 3. Update uv.lock
    print("Running `uv lock` to update uv.lock...")
    try:
        subprocess.run(["uv", "lock"], check=True)
        print("uv.lock updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running `uv lock`: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: `uv` command not found. Please ensure uv is installed.")
        sys.exit(1)
        
    print(f"\nVersion successfully updated to {new_version}!")

if __name__ == "__main__":
    main()
