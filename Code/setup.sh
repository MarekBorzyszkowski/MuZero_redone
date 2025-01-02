#!/bin/bash
if [ ! -d venv ]; then
	echo "Start of creating venv"
	python3.7 -m venv venv
	echo "venv created"
	echo "Start of installing python packages"
	source venv/bin/activate
	pip install -r requirements_guzw.txt
	deactivate
	echo "End of installing python packages"
else
	echo "venv already installed!"
fi