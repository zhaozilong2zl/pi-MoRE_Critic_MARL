SEEDS=(0 1 2)


GATE_TYPES=(
  "quadratic"
  "linear"
  "mlp"
  "one_hot"
)

COMBINATIONS=(
  "5 4"
  "4 3"
  "3 2"
)



for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for gate_type in "${GATE_TYPES[@]}"; do
		echo 'start running '${gate_type}' gate...'
		for seed in "${SEEDS[@]}"; do
			echo "=========================================================="
			echo "start running "${gate_type}" gate seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
			echo "=========================================================="
			python main.py \
			    --algorithm "pimore" \
			    --seed "${seed}" \
			    --num_adversaries "${num_adversaries}" \
			    --num_good_agents "${num_good_agents}" \
			    --critic_gate_type "${gate_type}" \
			    --critic_moe_type "residual" \
			    --critic_local_nn_num_cells 96 64 64 64 64 64 64 64 64 64 \
			    --critic_global_nn_num_cells 64 64 \
			    --critic_num_experts 4 \
			    --critic_top_k 2
		done
	done
done








