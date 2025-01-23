#!/bin/sh

set -e

about="
This script succeeds if all files have been committed to git,
and fails if any files have not been committed.
Stdout will contain the hash of the commit.
This scripts should be called at the beginning of every training run for reproducibility.
I have unfortunately had a major regression that I couldn't debug because I'm not sure exactly what the code was run for several training runs because the code wasn't being committed.
"

if ! [ -z "$(git status --porcelain)" ]; then
    echo "ERROR: there are uncommitted files"
    git status
    exit 1
fi

git rev-parse --short=8 HEAD
