rm -rf runs/
STEPS=5000
for DATASET in bishop_toy, abalone, concrete, energy, power, parkinsons, liver
do
    python run.py ce --dataset_name=$DATASET --steps=$STEPS --method_kwargs="{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}"
    python run.py pinball --dataset_name=$DATASET --steps=$STEPS --method_kwargs="{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (-0.1, 1.1)}" 
    python run.py crpshist --dataset_name=$DATASET --steps=$STEPS --method_kwargs="{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}"
    python run.py crpsqr --dataset_name=$DATASET --steps=$STEPS --method_kwargs="{quantile_levels: [0.1, 0.25, 0.5, 0.75, 0.9], bounds: (-0.1, 1.1)}"
    python run.py laplacewb --dataset_name=$DATASET --method_kwargs="{train_width: False}" --steps=$STEPS
    python run.py laplacewb --dataset_name=$DATASET --method_kwargs="{train_width: True}" --steps=$STEPS
    python run.py laplacescore --dataset_name=$DATASET --steps=$STEPS
    python run.py mdn --dataset_name=$DATASET --steps=$STEPS --method_kwargs="{n_components: 3}"
done



