SEEDS=(0 1 2)

COMBINATIONS=(
  "5 4"
  "4 3"
  "3 2"
)


echo 'start running residual structure...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running residual structure seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "pimore" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_gate_type "quadratic" \
		    --critic_moe_type "residual" \
		    --critic_local_nn_num_cells 96 64 56 48 56 48 56 48 56 48 \
		    --critic_global_nn_num_cells 64 64 \
		    --critic_num_experts 4 \
		    --critic_top_k 2
	done
done

echo 'start running vanilla-sparse structure...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running vanilla-sparse structure seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "pimore" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_gate_type "quadratic" \
		    --critic_moe_type "vanilla_sparse" \
		    --critic_local_nn_num_cells 96 64 96 64 96 64 96 64 \
		    --critic_global_nn_num_cells 64 64 \
		    --critic_num_experts 4 \
		    --critic_top_k 2
	done
done

echo 'start running vanilla-dense structure...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running vanilla-dense structure seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "pimore" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_gate_type "quadratic" \
		    --critic_moe_type "vanilla" \
		    --critic_local_nn_num_cells 96 72 96 72 \
		    --critic_global_nn_num_cells 64 64 \
		    --critic_num_experts 2 \
		    --critic_top_k 2
	done
done







