#!/bin/bash

python3 $(dirname $0)/compute_all_correlations.py $1

zip -r $(dirname $0)/../../corr.zip $(dirname $0)/correlations
sudo halt