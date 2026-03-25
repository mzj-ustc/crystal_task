#!/bin/bash 

cwd=`pwd`

n_parallel=48

for dd in `ls -d RUN*/`;
do
    cd $dd
    
    $cwd/loop_unit.sh &

    cd -;

    while true
    do
	unit_num=`ps -ejf | grep loop_unit | wc -l`

	if [ $unit_num -le $n_parallel ];
	then
	    break;
	fi

	sleep 10
    done

    
done
