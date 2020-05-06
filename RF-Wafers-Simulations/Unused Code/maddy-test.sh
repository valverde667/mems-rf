#File to quickly run single-species-simulation on a node on Lawrencium Cluster
for i in {1..1}
do
print(f"Simulation {i} activated")
python single-species-simulation.py --bunch_length=1e-9
sleep 1
print("Simulation Complete!")
done

# wait until all jobs are done
for job in `jobs -p`
do
echo $job
wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi
