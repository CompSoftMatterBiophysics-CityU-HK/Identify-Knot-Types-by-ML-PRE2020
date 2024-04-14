#!/usr/bin/env bash

# download to ./data folder for the jupyter docker to mount
cd ./data

# -nc to download if not exist
wget -nc https://zenodo.org/records/10946638/files/L100_Lp2_D11_circular_knot0-31-41-52-51.tar.gz
wget -nc https://zenodo.org/records/10946638/files/1M_L60_Lp4_D9_circular_knot0-31-41-52-51.tar.gz
