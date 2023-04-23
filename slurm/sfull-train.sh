#!/bin/bash
### Slurm Construction

#SBATCH --array=0-745%10
#SBATCH --job-name=mscbio2066
#SBATCH -t 04:00:00
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16

### User Config
DATASETS=(KDKI KDKIEC50)
RADII=(1 2 3 5 7 9 11)

### Job Local Directory Setup
WD=${SLURM_SUBMIT_DIR}
USER=$(whoami)
JOB_DIR=/scr/${USER}/p8
# copy data over
mkdir -p ${JOB_DIR}
cd ${JOB_DIR}
rsync -av ${WD}/sync/* .

# copy back files on exit 
trap "echo 'copying back on exit'; rsync -av * ${WD}/sync" EXIT

### MAIN ####
echo ${SLURM_JOB_NAME}  allocated to ${SLURM_NODELIST}


# activate modules
module load anaconda
eval "$(conda shell.bash hook)"
conda activate p8dev

echo starting run
# run script
for DS in "${DATASETS[@]}"; do
  for RADIUS in "${RADII[@]}"; do
    NAME="hu-fp2048r${RADIUS}-${DS}"
    CONFIG=configs/${NAME}.json
    DATA=data/
    OUTPUT=models/${NAME}

    echo -------------------
    echo Running $DS for FP $RADIUS over $N_CMPDS
    
    python3 ${WD}/train-model.py -i ${SLURM_ARRAY_TASK_ID} -j ${CONFIG} -d ${DATA} -o ${OUTPUT}
  done
done

exit 0
