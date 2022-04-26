EXPERIMENTS="experiments/data_sharing_changing_iid/experiments.csv"

{
    read
    while IFS=, read -r iid_score topology exp_name
    do 
        echo "Experiment with iid_score $iid_score, has topology $topology and named $exp_name"
        dvc exp run -n "$exp_name" -S iid_score="$iid_score" -S optimal_data_sharing.topology="$topology"
    done
} < $EXPERIMENTS 
