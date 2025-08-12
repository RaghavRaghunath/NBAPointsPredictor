#!/bin/bash

# Train models for each statistic
STATS=("PTS" "AST" "REB" "STL" "BLK")

for stat in "${STATS[@]}"; do
  echo "\n=== Training $stat model ==="
  python src/models/train_rnn_model.py --target_stat=$stat
  
  # Check if training was successful
  if [ $? -eq 0 ]; then
    echo "✅ Successfully trained $stat model"
  else
    echo "❌ Failed to train $stat model"
    exit 1
  fi
done

echo "\n=== All models trained successfully! ==="
