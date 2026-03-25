#!/bin/bash

for i in {0..15}; do temper.lmp_regroup_xyz -s 1 dump.temper.$i; done
