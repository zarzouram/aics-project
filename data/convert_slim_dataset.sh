# Dataset name mast be one of the following
# * turk_data
# * synthetic_data

# Kwargs
export DATASET_DIR=${1:-$HOME/resources/slim}


# Convert tfrecords to gzip files
# python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
#                                   --dataset "turk_data" \
#                                   --mode "train" \
#                                   --first-n -1 \
#                                   --batch-size 200

# echo "turk_data/train Completed"

# python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
#                                   --dataset "turk_data" \
#                                   --mode "valid" \
#                                   --first-n -1 \
#                                   --batch-size 200

# echo "turk_data/valid Completed"

# python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
#                                   --dataset "turk_data" \
#                                   --mode "test" \
#                                   --first-n -1 \
#                                   --batch-size 200

# echo "turk_data/test Completed"

# python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
#                                   --dataset "synthetic_data" \
#                                   --mode "valid" \
#                                   --first-n -1 \
#                                   --batch-size 200

echo "turk_data/valid Completed"

python3 ./convert_slim_dataset.py --dataset_path ${DATASET_DIR} \
                                  --dataset "synthetic_data" \
                                  --mode "test" \
                                  --first-n -1 \
                                  --batch-size 200
echo "All Completed"