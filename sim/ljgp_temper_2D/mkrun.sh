#!/bin/bash

cwd=`pwd`
fstate='states.dat'

arr=(`cat states.dat `)
idx=0

while true
do
    if [ -f STOP ];             # use "touch STOP" to stop the run
    then
        break
    fi

    sid=${arr[$idx]}
    idx=$(( $idx +1 ))
    eps=${arr[$idx]}
    idx=$(( $idx +1 ))
    r0=${arr[$idx]}
    idx=$(( $idx +1 ))

    #echo $id, $eps, $r0
    
    dname=RUN_$sid
    #echo $dname
    if [ -d $dname ]
    then
	continue
    fi
    
    mkdir $dname                        # make a new directory for the new structure, SID for structure ID
    cd $dname
    echo $eps > EPSILON
    echo $r0 > R0
    ../pair_ljgp_tabulate.py -n 301 -i 0.01 -o 3.01 > ljgp.tab
    cp ../setup/in.melt .
    cp ../setup/qlammps .
    cp ../setup/lammp.xyz.init .
    
    ./qlammps -n 64 -t 1
    wait
    cd $cwd

    echo $sid > Nrun
done
