#!/bin/bash
for i in {1..10}
do
	echo $i
	python training.py vae_supervised 0.001 16 0.0001 0.5 0 $i
done
