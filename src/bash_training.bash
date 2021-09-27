#!/bin/bash
for pat in {1..23}
do
  ech $pat
	python mmd_train.py $pat
done
