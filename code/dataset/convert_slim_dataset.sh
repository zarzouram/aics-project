# refrence: https://github.com/rnagumo/gqnlib/blob/master/bin/download_slim.sh

# Dataset name mast be one of the following
# * turk_data
# * synthetic_data

# Kwargs
export DATASET_DIR=${1:-$HOME/Corpora/slim/original_data}
export VOCAB_PATH=${2:-$HOME/LT2318/aics-project/data/vocab.json}

# Convert tfrecords to gzip files
python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "turk_data" \
                                  --mode "train" \
                                  --vocab_path ${VOCAB_PATH} \
                                  --first-n -1 \
                                  --batch-size 500

echo "turk_data/train Completed"

python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "turk_data" \
                                  --mode "valid" \
                                  --vocab_path ${VOCAB_PATH} \
                                  --first-n -1 \
                                  --batch-size 500

echo "turk_data/valid Completed"

python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "turk_data" \
                                  --mode "test" \
                                  --vocab_path ${VOCAB_PATH} \
                                  --first-n -1 \
                                  --batch-size 500

echo "turk_data/test Completed"

echo "All Completed"