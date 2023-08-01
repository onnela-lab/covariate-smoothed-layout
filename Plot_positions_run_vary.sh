#
echo “submitted”
#
sbatch -o out.txt \
-e e.txt \
--job-name=my_analysis.txt \
Plot_positions_base_shell.sh
#exit
sleep 0.4 # pause to be kind to the scheduler

