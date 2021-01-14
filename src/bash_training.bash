#!/bin/bash
for i in {1..10}
do
	echo $i
	python training.py vae_unsupervised 0.00001 16 0.0001 0.5 0 $i
done
