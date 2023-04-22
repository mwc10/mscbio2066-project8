#!/bin/bash
### Slurm Construction

#SBATCH --array=0-745%10
#SBATCH --job-name=mscbio2066
#SBATCH -t 02:00:00
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=16

### User Config
DATASETS=(KiKd KiKdEC50)
RADII=(1 2 3 5 7 9 11)
MIN_CMPDS=(30 60 120)

CNAME=hu-fp2048r11-KiKdEC50-60
CONFIG=configs/${CNAME}.json
SOURCE_DATA=data/
OUTPUT=models/${CNAME}

### Job Local Directory Setup
#WD=${SLURM_SUBMIT_DIR}
#USER=$(whoami)
#JOB_DIR=/scr/${USER}/p8
# copy data over
#mkdir -p ${JOB_DIR}
#cd ${JOB_DIR}
#rsync -av ${WD}/sync/* .

# copy back files on exit 
#trap "echo 'copying back on exit'; rsync -av * ${WD}/sync" EXIT

### MAIN ####
echo ${SLURM_JOB_NAME}  allocated to ${SLURM_NODELIST}

# activate modules
#module load anaconda
#eval "$(conda shell.bash hook)"
#conda activate p8dev

# run script
for DS in "${DATASETS[@]}"; do
  for RADIUS in "${RADII[@]}"; do
    for N_CMPDS in "${MIN_CMPDS[@]}"; do
      NAME="hu-fp2048r${RADIUS}-${DS}-${N_CMPDS}"
      CONFIG=configs/${NAME}.json
      DATA=data/
      OUTPUT=models/${NAME}
      echo $NAME
    done
  done
done
 
#python3 ${WD}/train-model.py -i ${SLURM_ARRAY_TASK_ID} -j ${CONFIG} -d ${SOURCE_DATA} -o ${OUTPUT}

exit 0
