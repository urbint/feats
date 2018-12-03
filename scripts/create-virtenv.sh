#!/bin/bash

export VE=".virtualenv"

if [ ! -d "$VE" ]; then
    echo "creating virtual env"
    py=`which python`
    echo "using python $py"
    virtualenv "$VE" -p $py
else
    echo "virutalenv folder already exists"
fi

source "$VE/bin/activate"

# todo: use conda instead
pip install wheel
