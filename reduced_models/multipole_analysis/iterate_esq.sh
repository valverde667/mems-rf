# Assuming there is a data file with centers in it
# for x in $(cat fraction_R.txt);
#   do
#     for y in $(cat chop.txt);
#       do
#         python esq_field_analysis.py --scale_pole $x --rod_fraction $y
#       done
#   done

# iterate length of ESQs
# for x in $(cat length_scales.txt);
#   do
#     python esq_field_analysis.py --scale_length $x
#   done

# iterate Rod Radius
for x in $(cat rod_ratios.txt);
  do
    python esq_field_analysis.py --scale_pole_rad $x
  done

# iterate Rod Fraction
# for x in $(cat chop.txt);
#   do
#     python esq_field_analysis.py --rod_fraction $x --scale_pole_rad 1.145
#   done

# iterate Rod Voltage
# for x in $(cat Vsets.txt);
#   do
#     python esq_field_analysis.py --voltage $x --rod_fraction 1.0
#   done
