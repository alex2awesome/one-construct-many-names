#!/bin/bash
# PREREG-28 helper: pull one task's judge outputs from sk2 into its outputs dir (rsync, no deletes).
# usage: prereg28_sync.sh <task>
T=$1
R=ROOT
O=$R/outputs/lexicon/prereg28_${T}_20261012
timeout 120 rsync -a sk2:/remote/anon/p28/$T/ $O/ 2>&1 | grep -v -E 'chdir|bashrc'
timeout 60 rsync -a sk2:/remote/anon/p28/runlog_gpt56.tsv sk2:/remote/anon/p28/runlog_claude.tsv sk2:/remote/anon/p28/runner_gpt56.log sk2:/remote/anon/p28/runner_claude.log $O/ 2>&1 | grep -v -E 'chdir|bashrc'
