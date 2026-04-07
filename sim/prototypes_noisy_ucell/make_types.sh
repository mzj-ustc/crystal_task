#!/bin/bash

# This file contains all prototypes of known uniary structures
# Copied from Encyclopedia of Crystallographic Prototypes" at
# https://www.aflowlib.org/prototype-encyclopedia/prototype_index.html
# Remeber to cite their referenes if using this data file
fname='prototypes_list.txt'

num=`cat $fname | tail -n +3 | wc -l`
dnames=(`cat $fname | awk '{print $4}' | tail -n +2 | tr '\n' ' '`)
keywords=(`cat $fname | awk '{print $5}' | tail -n +2 | tr '\n' ' '`)

is_wget=false

for i in `seq 1 $num`;
do
    echo ${dnames[$i]} ${keywords[$i]}
    keyword=${keywords[$i]}
    dname=$keyword
    
    if $is_wget
    then  
	mkdir $keyword
	cd $keyword
	wget https://www.aflowlib.org/prototype-encyclopedia/POSCAR/$keyword.poscar 
	cd -
    fi

    # cd $dname
    # poscar_name=$keyword.poscar
    # pos2xyz $poscar_name > xyz.ini
    # cd -;

    cd $dname
    if ! [ -f xyz.cluster ];
    then 
	../make_structs.py xyz.ini -a 70 -n 2744
    fi
    cd -;

done

