#!/bin/bash
for l in {0..30}
do
  echo $l
	python training.py vae_epil 0.0001 32 0.0001 0.5 0 $l
done
