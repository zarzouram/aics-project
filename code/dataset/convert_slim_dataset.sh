# refrence: https://github.com/rnagumo/gqnlib/blob/master/bin/download_slim.sh

# Dataset name mast be one of the following
# * turk_data
# * synthetic_data

# Kwargs
export DATASET_DIR=${1:-$HOME/Corpora/slim/original_data}

# Convert tfrecords to gzip files
python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "turk_data" \
                                  --mode "train" \
                                  --first-n -1 \
                                  --batch-size 64

echo "turk_data/train Completed"

python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "turk_data" \
                                  --mode "valid" \
                                  --first-n -1 \
                                  --batch-size 64

echo "turk_data/valid Completed"

python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "turk_data" \
                                  --mode "test" \
                                  --first-n -1 \
                                  --batch-size 64

echo "turk_data/test Completed"

echo "All Completed"