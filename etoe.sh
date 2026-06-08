# run_all_exchange_rate.sh
#!/bin/bash

echo "=== Step 1: Preprocessing Exchange Rate Data ==="
python preprocess_exchange_rate.py

echo ""
echo "=== Step 2: Training TimeDiff Model ==="
python etdiff_train.py \
  --load_path data/exchange_rate_processed/TRAIN_exchange_rate.pt \
  --data_name exchange_rate \
  --check_point_path models/exchange_rate_model.pt \
  --cut_length 96 \
  --hidden 128 \
  --num_layers 3 \
  --model lstm \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --num_steps 100000 \
  --timesteps 1000 \
  --diff_type gaussian \
  --seed 2023

echo ""
echo "=== Step 3: Evaluation ==="
for metric in discriminative predictive; do
  echo "Running $metric evaluation..."
  python eval_samples.py \
    --data_name exchange_rate \
    --sync_path samples/exchange_rate_model_samples.npy \
    --train_path data/exchange_rate_processed/TRAIN_exchange_rate.pt \
    --test_path data/exchange_rate_processed/TEST_exchange_rate.pt \
    --metric $metric \
    --runs 10 \
    --seed 2023
done

echo ""
echo "=== Step 4: Visualization ==="
python eval_samples.py \
  --data_name exchange_rate \
  --sync_path samples/exchange_rate_model_samples.npy \
  --train_path data/exchange_rate_processed/TRAIN_exchange_rate.pt \
  --test_path data/exchange_rate_processed/TEST_exchange_rate.pt \
  --metric t-sne \
  --t_sne_num 2000 \
  --img_name exchange_rate_tsne.png \
  --seed 2023

echo ""
echo "=== All steps complete! ==="