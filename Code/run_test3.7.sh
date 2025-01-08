#!/bin/bash
if [ ! -d venv3.7 ]; then
  echo "venv hasn't been initialized. Running setup.sh"
  ./setup.sh
fi
source venv3.7/bin/activate
echo "Running muzero"
venv3.7/bin/python3 muzero.py
echo "End of muzero"
deactivate
