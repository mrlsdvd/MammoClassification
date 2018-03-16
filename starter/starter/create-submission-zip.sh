#!/bin/bash
if [ -d deps/ ]
then
  zip -r gradescope-submission.zip *.py deps/
else
  zip -r gradescope-submission.zip *.py
fi
