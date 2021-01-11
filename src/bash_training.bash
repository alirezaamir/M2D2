#!/bin/bash
for i in {1..24}
do
	echo $i
	python training.py ae_unsupervised 0.001 16 0.0001 0.5 0 $i
done
