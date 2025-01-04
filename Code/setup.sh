#!/bin/bash
if [ ! -d venv ]; then
	echo "Start of creating venv"
	python3.10 -m venv venv
	echo "venv created"
	echo "Start of installing python packages"
	source venv/bin/activate
	pip install -r requirements_3.10.txt
	deactivate
	echo "End of installing python packages"
else
	echo "venv already installed!"
fi
