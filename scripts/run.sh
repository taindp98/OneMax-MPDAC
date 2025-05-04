# Check if the .env file exists
if [ -f .env ]; then
    # Load environment variables from the .env file
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
fi
# Set the working directory to the current directory
export WORKDIR=$(pwd)
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# python onemax_mpdac/train.py -c onemax_mpdac/configs/onemax_n100_cmp.yml -o outputs -s 1 --n-cpus 4 --gamma 0.99
python onemax_mpdac/derive_mp_policy.py --n 500 --type lbd1_alpha_lbd2 --is-discrete