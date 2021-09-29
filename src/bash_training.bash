#!/bin/bash
for pat in {8..23}
do
  echo $pat
	python mmd_train.py $pat
done
