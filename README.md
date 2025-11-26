# 1. Make sure you're in the right directory
pwd

# 2. Clean old build artifacts
rm -rf build dist *.egg-info

# 3. Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4. Editable install
python -m pip install -e .