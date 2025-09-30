# commands
# rm -rf runs
SEED=0

datasets=("bishop_toy" "Abalone" "Concrete Compressive Strength" "Energy Efficiency" "Combined Cycle Power Plant" "Parkinsons Telemonitoring" "Liver Disorders")
for dataset in "${datasets[@]}"
do
    python runsimple.py mdn --method_kwargs="{n_components: 8}" --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py laplacescore --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py laplacecrps --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py crpshist --method_kwargs="{n_bins: 32, bounds: (0,1)}" --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py ce --method_kwargs="{n_bins: 32, bounds: (0,1)}" --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py logscoreqr --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py pinball --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=final --dataset_name="$dataset" --seed=$SEED &
    python runsimple.py laplacewb --tag=final --dataset_name="$dataset" --seed=$SEED &
    wait
done
echo "DONE SEED $SEED"

# python runsimple.py logscoreqr --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=last --dataset_name="Concrete Compressive Strength" &
# python runsimple.py logscoreqr --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=last --dataset_name="Energy Efficiency" &
# python runsimple.py logscoreqr --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=last --dataset_name="Combined Cycle Power Plant" &
# python runsimple.py logscoreqr --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=last --dataset_name="Parkinsons Telemonitoring" &
# python runsimple.py logscoreqr --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=last --dataset_name="Liver Disorders" &
# wait
# echo DONEEEE

# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name Abalone 
# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name Abalone &
# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name="Concrete Compressive Strength" &
# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name="Energy Efficiency" &
# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name="Combined Cycle Power Plant" &
# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name="Parkinsons Telemonitoring" &
# python runsimple.py mcd --weight_decay 0.005 --method_kwargs "{n_bins: 32, bounds: (0, 1), dropout_p: 0.15, n_preds: 1000}" --max_seconds 120 --val_every None --dataset_name="Liver Disorders" &
# wait
# echo DONEEEE


# python runsimple.py ce --method_kwargs="{n_bins: 32, bounds: (0,1)}" --tag=new_32 &
# python runsimple.py ce --method_kwargs="{n_bins: 5, bounds: (0,1)}" --tag=new_5
# wait
# python runsimple.py crpshist --method_kwargs="{n_bins: 32, bounds: (0,1)}" --tag=new_32 &
# python runsimple.py crpshist --method_kwargs="{n_bins: 5, bounds: (0,1)}" --tag=new_5 &
# python runsimple.py crpshist --method_kwargs="{n_bins: 32, bounds: (0,1)}" --lr=0.001 --tag=new_32_001
# wait
# python runsimple.py crpshist --method_kwargs="{n_bins: 5, bounds: (0,1)}" --lr=0.001 --tag=new_5_001 &
# python runsimple.py pinball --method_kwargs="{n_quantile_levels: 32, bounds: (0,1)}" --tag=new_32 &
# python runsimple.py pinball --method_kwargs="{n_quantile_levels: 5, bounds: (0,1)}" --tag=new_5 &
# python runsimple.py crpsqr --method_kwargs="{n_quantile_levels: 5, bounds: (0,1)}" --tag=new_5
# wait


# rm -rf runs
# python monocular.py --method_name=laplacewb --device="cuda:0" &
# python monocular.py --method_name=laplacescore --device="cuda:1" 
# wait
# python monocular.py --method_name=mdn --method_kwargs="{n_components: 3}" --device="cuda:0" &
# python monocular.py --method_name=pinball --method_kwargs="{n_quantile_levels: 128, bounds: (-5., 5.)}" --device="cuda:1" 
# wait
# python monocular.py --method_name=crpsqr --method_kwargs="{n_quantile_levels: 128, bounds: (-5., 5.)}" --device="cuda:0" &
# python monocular.py --method_name=ce --method_kwargs="{n_bins: 128, bounds: (-5., 5.)}" --device="cuda:1" 
# wait
# python monocular.py --method_name=crpshist --method_kwargs="{n_bins: 128, bounds: (-5., 5.)}" --device="cuda:1" &


# rm -rf runs/
# for SEED in 0 1 2
# do
#     for DATASET in abalone concrete energy power parkinsons liver bishop_toy
#     do
#         python run.py ce --dataset_name=$DATASET --method_kwargss="[{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, {bin_borders: [0, 0.25, 0.5, 0.75, 1.0]}]" --seeds=$SEED  &
#         python run.py pinball --dataset_name=$DATASET --method_kwargss="[{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.)}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.)}]" --seeds=$SEED &
#         python run.py crpshist --dataset_name=$DATASET --method_kwargss="[{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, {bin_borders: [0, 0.25, 0.5, 0.75, 1.0]}]" --seeds=$SEED &
#         python run.py crpsqr --dataset_name=$DATASET --method_kwargss="[{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.)}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.)}, {quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.), predict_residuals: True}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.), predict_residuals: True}]" --seeds=$SEED &
#         python run.py laplacewb --dataset_name=$DATASET --method_kwargss="[{train_width: False}, {train_width: True}]" --seeds=$SEED &
#         python run.py laplacescore --dataset_name=$DATASET --seeds=$SEED &
#         python run.py mdn --dataset_name=$DATASET --method_kwargss="[{n_components: 3}, {n_components: 10}]" --seeds=$SEED &
#         python run.py iqn --dataset_name=$DATASET --seeds=$SEED &
#         wait
#     done
#     echo "Finished seed $SEED"
# done

# #  FOR TESTING

# # set -e
# # for SEED in 0 1 2
# # do
# #     for DATASET in abalone concrete energy power parkinsons liver bishop_toy
# #     do
# #         python run.py ce --dataset_name=$DATASET --method_kwargss="[{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, {bin_borders: [0, 0.25, 0.5, 0.75, 1.0]}]" --seeds=$SEED  
# #         python run.py pinball --dataset_name=$DATASET --method_kwargss="[{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.)}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.)}]" --seeds=$SEED 
# #         python run.py crpshist --dataset_name=$DATASET --method_kwargss="[{bin_borders: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, {bin_borders: [0, 0.25, 0.5, 0.75, 1.0]}]" --seeds=$SEED 
# #         python run.py crpsqr --dataset_name=$DATASET --method_kwargss="[{quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.)}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.)}, {quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], bounds: (0., 1.), predict_residuals: True}, {quantile_levels: [0.05, 0.25, 0.5, 0.75, 0.95], bounds: (0., 1.), predict_residuals: True}]" --seeds=$SEED 
# #         python run.py laplacewb --dataset_name=$DATASET --method_kwargss="[{train_width: False}, {train_width: True}]" --seeds=$SEED 
# #         python run.py laplacescore --dataset_name=$DATASET --seeds=$SEED 
# #         python run.py mdn --dataset_name=$DATASET --method_kwargss="[{n_components: 3}, {n_components: 10}]" --seeds=$SEED 
# #         python run.py iqn --dataset_name=$DATASET --seeds=$SEED 
# #     done
# #     wait
# #     echo "Finished seed $SEED"
# #     exit
# # done



