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
cd ../
mkdir ./tmp
cd ./tmp
wget https://github.com/cvc5/cvc5/releases/download/cvc5-1.2.0/cvc5-Linux-x86_64-static.zip
unzip cvc5-Linux-x86_64-static.zip
sudo cp ./cvc5-Linux-x86_64-static/bin/cvc5 /usr/local/bin
cd ../
rm -rf tmp
why3 config detect
source "$HOME/.profile"
