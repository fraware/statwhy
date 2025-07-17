#!/bin/bash

sudo apt-get update

# Install opam
sudo apt-get install opam
opam init --compiler=5.0.0 -y --shell=bash --shell-setup
eval $(opam env --switch=5.0.0)

# Install StatWhy
cd cameleer
sudo apt-get install autoconf libcairo2-dev libgtk-3-dev libgtksourceview-3.0-dev
opam pin add . -y

# Install CVC5
sudo apt-get install cvc5

# Enable CVC5 on Why3
why3 config detect
source "$HOME/.profile"
