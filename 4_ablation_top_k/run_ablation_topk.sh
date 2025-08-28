SEEDS=(0 1 2)

TOPKS=(
	1
	2
	3
	4
)


echo 'running quadratic top-k ablation experiments...'
for topk in "${TOPKS[@]}"; do
	echo 'start running top-'${topk}'...'
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running quadratic gate top-${topk} seed ${seed} with num_adversaries=4, num_good_agents=3"
		echo "=========================================================="
		python main.py \
		    --algorithm "pimore" \
		    --seed "${seed}" \
		    --num_adversaries 4 \
		    --num_good_agents 3 \
		    --critic_gate_type "quadratic" \
		    --critic_moe_type "residual" \
		    --critic_local_nn_num_cells 96 64 64 64 64 64 64 64 64 64 \
		    --critic_global_nn_num_cells 64 64 \
		    --critic_num_experts 4 \
		    --critic_top_k "${topk}"
	done
done

echo 'running linear top-k ablation experiments...'
for topk in "${TOPKS[@]}"; do
	echo 'start running top-'${topk}'...'
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running linear gate top-${topk} seed ${seed} with num_adversaries=4, num_good_agents=3"
		echo "=========================================================="
		python main.py \
		    --algorithm "pimore" \
		    --seed "${seed}" \
		    --num_adversaries 4 \
		    --num_good_agents 3 \
		    --critic_gate_type "linear" \
		    --critic_moe_type "residual" \
		    --critic_local_nn_num_cells 96 64 64 64 64 64 64 64 64 64 \
		    --critic_global_nn_num_cells 64 64 \
		    --critic_num_experts 4 \
		    --critic_top_k "${topk}"
	done
done






