#!/bin/bash
if [ ! -d venv3.7 ]; then
	echo "Start of creating venv3.7"
	python3.10 -m venv venv3.7
	echo "venv created"
	echo "Start of installing python packages"
	source venv3.7/bin/activate
	pip install -r requirements_3.7.txt
	deactivate
	echo "End of installing python packages"
else
	echo "venv3.7 already installed!"
fi
