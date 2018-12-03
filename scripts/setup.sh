#!/bin/bash

. "scripts/create-virtenv.sh"

# this deals with some pita sklearn issue
export LC_ALL=C


# todo: use conda instead
pip install -e .
