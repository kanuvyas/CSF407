#!/usr/bin/env bash
# Runs every Prolog query used in the report against planner.pl
cd "$(dirname "$0")"
for q in "animal(polly)" "can_move(a,b)" "can_move(a,c)" "valid_move(a,b)" "valid_move(b,c)" \
         "valid_move(a,c)" "valid_path([a,b,c])" "valid_path([a,c])" "reduce_speed"; do
  printf '?- %-22s' "$q."
  swipl -q -g "( $q -> writeln(true) ; writeln(false) )" -t halt planner.pl
done
