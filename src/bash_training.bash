#!/bin/bash
for i in {1..10}
do
	echo $i
	python training.py epilepsiae 0.00001 16 0.00001 0.5 0 $i
done
