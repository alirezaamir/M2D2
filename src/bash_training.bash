#!/bin/bash
for l in 2 4 8 16 32 64 128
do
	python training.py vae_unsup_chb 0.00001 $l 0.0001 0.02 0 -1
done
