# Assuming there is a data file with centers in it
# iterate length of ESQs
for x in $(cat Vsets.txt);
  do
    python analytic_rf_bucket.py --Vset $x
  done
