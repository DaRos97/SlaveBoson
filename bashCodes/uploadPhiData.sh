#!/bin/bash

# Copy data to git folder
cp -r Data/phi* git/SlaveBoson/Data/

# go to git folder
cdd

# add changes to commit
git add Data/phi*
# commit changes with parameter
git commit -m $1
# push changes
git push
