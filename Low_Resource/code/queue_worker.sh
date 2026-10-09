#!/bin/bash
# usage: queue_worker.sh "<models>" <gpus> <pid-to-wait-for>
# Repeatedly extracts every condition that has data/<cond>.json but no cache yet.
# Stops when the file STOP_QUEUE exists in this directory.
export HF_HOME=/mount/studenten-temp1/users/shahidmm/.hfcache
PY=/mount/studenten-temp1/users/shahidmm/virtualenvs/td/bin/python
LOG=/mount/studenten-temp1/users/shahidmm/research_n/logs
cd "$(dirname "$0")"
while kill -0 "$3" 2>/dev/null; do sleep 30; done
ALL="en de ar hi fr es ur ur_Deva ur_Latn mr ne gu pa pa_Arab pa_Deva gu_Deva mr_Gujr hi_Latn"
while [ ! -f STOP_QUEUE ]; do
  for m in $1; do
    todo=""
    for c in $ALL; do
      [ -f ../data/$c.json ] && [ ! -f ../acts/$m/$c.npy ] && todo="$todo,$c"
    done
    todo=${todo#,}
    if [ -n "$todo" ]; then
      echo "$(date) $m $todo" >> $LOG/queue_$2.log
      $PY -u extract.py --model $m --conds $todo --gpus $2 >> $LOG/extract_$m.log 2>&1
    fi
  done
  sleep 60
done
