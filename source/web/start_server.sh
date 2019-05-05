#!/bin/bash
rm -rf ./images/*.jpg
./open_browser.sh &
PYTHONPATH=$PYTHONPATH:../ /usr/bin/python3 ./app.py
