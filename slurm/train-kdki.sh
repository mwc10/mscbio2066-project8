#!/bin/bash
### Slurm Construction

#SBATCH --job-name=mscbio2066
#SBATCH -t 01:00:00
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16

### User Config
CONFIG=configs/hu-b2048-r2-kikd.json
SOURCE_DATA=data/
OUTPUT=models/hu-b20480-r2-kikd/

### Job Local Directory Setup
WD=${SLURM_SUBMIT_DIR}
USER=$(whoami)
JOB_DIR=/scr/${USER}/${SLURM_JOB_ID}
# copy data over
mkdir -p ${JOB_DIR}
cd ${JOB_DIR}
rsync -av ${WD}/sync/* .

# copy back files on exit 
trap "echo 'copying back on exit'; rsync -av * ${WD}/sync" EXIT

# finally, start the script



echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}

# activate modules
module load anaconda
eval "$(conda shell.bash hook)"
conda activate p8dev

# run script
python3 ${WD}/train-model.py -i ${SLURM_ARRAY_TASK_ID} -j ${CONFIG} -d ${SOURCE_DATA} -o ${OUTPUT}

exit 0
