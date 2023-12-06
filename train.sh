#!/usr/bin/env bash
set -e
set -u


git add */*.py
git add */*/*.py
git add *.sh
git commit -m "new run `date +'%Y-%m-%d %H:%M:%S'`"
python RL/RL.py