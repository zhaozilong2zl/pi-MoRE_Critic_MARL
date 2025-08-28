SEEDS=(0 1 2)

COMBINATIONS=(
  "5 4"
  "4 3"
  "3 2"
)


echo 'start running pi-MoRE...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running pi-MoRE seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
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

echo 'start running pi-MAPPO...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running pi-MAPPO seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "pimappo" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_local_nn_num_cells 128 112 \
		    --critic_global_nn_num_cells 64 64
	done
done

echo 'start running MAAC...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running MAAC seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "maac" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_num_cells_e 64 64 \
		    --critic_num_cells_q 64 64 \
		    --critic_num_heads 4
	done
done

echo 'start running MA-POCA...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running MA-POCA seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "mapoca" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_num_cells_e 64 64 \
		    --critic_num_cells_v 64 64 \
		    --critic_num_heads 4 \
		    --critic_embedding_dim 64
	done
done

echo 'start running MAPPO...'
for combo in "${COMBINATIONS[@]}"; do
	read num_adversaries num_good_agents <<< "$combo"
	for seed in "${SEEDS[@]}"; do
		echo "=========================================================="
		echo "start running MAPPO seed ${seed} with num_adversaries=${num_adversaries}, num_good_agents=${num_good_agents}"
		echo "=========================================================="
		python main.py \
		    --algorithm "mappo" \
		    --seed "${seed}" \
		    --num_adversaries "${num_adversaries}" \
		    --num_good_agents "${num_good_agents}" \
		    --critic_num_cells 128 128
	done
done










