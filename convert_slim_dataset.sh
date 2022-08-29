# refrence: https://github.com/rnagumo/gqnlib/blob/master/bin/download_slim.sh

# Dataset name mast be one of the following
# * turk_data
# * synthetic_data

# Kwargs
ds_dir="/srv/data/zarzouram/lt2318/slim/original/"
ds="turk"

while [ "$1" != "" ]; do
    case $1 in
    --ds_dir )  shift
                ds_dir=$1
                ;;
    --ds )      shift
                ds=$1
                ;;
    *) break;
    esac;
    shift
done

if [ $ds != "turk" ] && [ $ds != "synth" ]; then
    echo "Error: DATASET shoudld be either 'turk' or 'synth'"
    exit 1
fi

# Convert tfrecords
echo "Start processing $ds train dataset"
python3 ./codes/dataset/convert_slim_dataset.py \
                                --dataset_path ${ds_dir} \
                                --dataset ${ds} \
                                --mode "train"

echo -e "\nStart processing $ds test dataset"
python3 ./codes/dataset/convert_slim_dataset.py \
                                --dataset_path ${ds_dir} \
                                --dataset ${ds} \
                                --mode "test"

if [[ "$ds" == "turk" ]]; then
    echo -e "\nStart processing $ds val dataset"
    python3 ./codes/dataset/convert_slim_dataset.py \
                                    --dataset_path ${ds_dir} \
                                    --dataset ${ds} \
                                    --mode "valid"

fi

echo -e "\nAll tasks are completed."