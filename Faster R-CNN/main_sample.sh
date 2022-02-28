#!/bin/bash
#SBATCH --job-name=MobNet_trn(OOD_dt3)
#SBATCH --partition=any #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=60GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/pbenga2s/RnD/crowd-counting-using-tensorflow/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/pbenga2s/RnD/crowd-counting-using-tensorflow/job_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/RnD

# locate to your root directory
cd /home/pbenga2s/RnD/crowd-counting-using-tensorflow
# run the script
python main_sample.py

#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml

