#!/bin/bash
# usage: run_extract.sh "<model1> <model2> ..." <gpus> <conds>
export HF_HOME=/mount/studenten-temp1/users/shahidmm/.hfcache
PY=/mount/studenten-temp1/users/shahidmm/virtualenvs/td/bin/python
LOG=/mount/studenten-temp1/users/shahidmm/research_n/logs
cd "$(dirname "$0")"
for m in $1; do
  $PY extract.py --model $m --conds $3 --gpus $2 2>&1 | grep -v -E "Loading weights|Warning|warn" >> $LOG/extract_$m.log
done
