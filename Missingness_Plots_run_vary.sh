for iter in 1; do
for total_nodes in 100; do
for groups in ${total_nodes}; do
#for groups in 2 5; do
for p_out in 5; do
for p_in in 0 0.50 0.75 1.0 1.5; do
for cat_cont in 2; do

#
echo “${iter}, ${total_nodes}, ${groups}, ${p_out}, ${p_in}, ${cat_cont},”
#
sbatch -o out_iter_${iter}_TN_${total_nodes}_groups_${groups}_PI_${p_in}_PO_${p_out}_CC_${cat_cont}.txt \
-e e_iter_${iter}_TN_${total_nodes}_groups_${groups}_PI_${p_in}_PO_${p_out}_CC_${cat_cont}.txt \
--job-name=my_analysis_iter_${iter}_TN_${total_nodes}_groups_${groups}_PI_${p_in}_PO_${p_out}_CC_${cat_cont}.txt \
Missingness_Plots_base_shell.sh $iter $p_in $p_out $groups $total_nodes $cat_cont
#exit
sleep 0.4 # pause to be kind to the scheduler
done
done
done
done
done
done

