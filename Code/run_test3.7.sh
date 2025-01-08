#!/bin/bash
if [ ! -d venv ]; then
  echo "venv hasn't been initialized. Running setup.sh"
  ./setup.sh
fi
source venv/bin/activate
echo "Running muzero"
venv3.7/bin/python3 muzero.py
echo "End of muzero"
deactivate
