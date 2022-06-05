# Assuming there is a data file with centers in it
for x in $(cat fraction_R.txt);
  do
    for y in $(cat chop.txt);
      do
        python conductor_fieldsolve.py --scale_pole $x --rod_fraction $y
      done
  done
