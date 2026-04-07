#!/bin/bash

name=`echo $1 | tr '.' '\ ' | awk '{print $1}'`
new_name=$2

ln -s $name.out $new_name.out
