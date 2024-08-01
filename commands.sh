rm -rf runs/
STEPS=100
for DATASET in abalone concrete energy power parkinsons liver bishop_toy
do
    python run.py ce --dataset_name=$DATASET --method_kwargss="[{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, {bin_borders: [0, 0.25, 0.5, 0.75, 1.0]}]"
    python run.py pinball --dataset_name=$DATASET --method_kwargss="[{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.)}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.)}]" 
    python run.py crpshist --dataset_name=$DATASET --method_kwargss="[{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, {bin_borders: [0, 0.25, 0.5, 0.75, 1.0]}]"
    python run.py crpsqr --dataset_name=$DATASET --method_kwargss="[{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.)}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.)}, {quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.), predict_residuals=True}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.), predict_residuals=True}]" 
    python run.py laplacewb --dataset_name=$DATASET --method_kwargss="[{train_width: False}, {train_width: True}]" 
    python run.py laplacescore --dataset_name=$DATASET 
    python run.py mdn --dataset_name=$DATASET --method_kwargss="[{n_components: 3}, {n_components: 10}]"
done



