#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o /u/home/s/skikuchi/scratch/MedAI6/joblog/ResUnetAtt_tmp.$JOB_ID
#$ -j y
# Edit the line below to request the appropriate runtime and memory
# (or to add any other resource) as needed:
#$ -l h_rt=5:00:00,h_data=32G,gpu,RTX2080Ti,cuda=1
# Add multiple cores/nodes as needed:
#$ -pe shared 1

# ex. qsub sub_valid.sh normal

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "


# load the job environment:
. /u/local/Modules/default/init/modules.sh
module load anaconda3
module load cuda/11.8
module load python
# To see which versions of anaconda are available use: module av anaconda
# activate an already existing conda environment (CHANGE THE NAME OF THE ENVIRONMENT):
conda activate /u/home/s/skikuchi/miniconda3/envs/kaggle

# in the following two lines substitute the command with the
# needed command below:
echo "python Transfer_AttUnet.py"
# call python code with -out only if second arguments is given:

python "/u/home/s/skikuchi/scratch/MedAI6/Transfer_Unets/UnetAtt/Transfer_AttUnet.py"

# echo job info on joblog:
echo "Job $JOB_ID.$SGE_TASK_ID ended on:   " `hostname -s`
echo "Job $JOB_ID.$SGE_TASK_ID ended on:   " `date `
echo " "
### anaconda_python_submit.sh STOP ###