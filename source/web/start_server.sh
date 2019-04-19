#!/bin/bash
rm -rf ./images/*
./open_browser.sh &
PYTHONPATH=$PYTHONPATH:../ /usr/bin/python3 ./app.py
