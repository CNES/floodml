#!/bin/bash
#SBATCH --job-name=FloodDAMDT    # job's name
# --output=/work/scrath/data/username/output  --error=/work/scrath/data/username/error (%j = jobID )
#SBATCH --output=./LOG/outputfile-%j.out
#SBATCH --error=./LOG/errorfile-%j.err
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH -n 32                   # number of tasks ( or --tasks=32)
#SBATCH --gpus=a100_40g:1    # for A100 device
#SBATCH --qos=gpu_all               # QoS
#SBATCH --partition=gpu_std         # partition
#SBATCH --time=00:59:00             # Walltime put to an hour. Rise that number for large dataset processing
#SBATCH --mem-per-cpu=4000M         # memory per core 
#SBATCH --account=floodml         # MANDATORY : account (launch myaccounts to list your accounts) 
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc
#  Launch job with command : sbatch job_array.slurm
#  Check job with : squeue -u $USER
#Statut détaillé d'un job: scontrol show job <jobid>

echo ""
echo "###############################################################"
echo "                  Hello from Inference script!                 "
echo "###############################################################"
echo ""

module load conda
conda activate rapids-0.21.08

echo "Environment rapids-0.21.08 loaded"
cd Path2FLDMLproject
echo "Executing python Inference script..."

# Configuration
cedir=' /work/datalake/static_aux/MNT/Copernicus_DSM/'
medir='/work/FLOODML/data/deliveries/floodml/phase-1-cls/MERIT_S2/'
gswdir='/work/datalake/static_aux/MASQUES/PEKEL/2018/occurrence/'
wc_dir='/work/FLOODML/data/deliveries/flooddamdt/ESA_worldcover/2021_v200'

dbpath='/work/FLOODML/data/deliveries/flooddam/trained_models/DB_S2_GPU_R02108.sav'

infold='/work/FLOODML/data/deliveries/flooddam/Study_cases/Soudan/S2L2A'
oufold='/work/scratch/data/username/FLDML_OUT/'
tmp_dir='/work/scratch/data/username/FLDML_OUT/'

type='s2'

# Lancement
python RDF-3-inference.py -i $infold -o $oufold -m $medir -wc $wc_dir --satellite $type -db $dbpath -g $gswdir -r 2 -tmp $tmp_dir

echo "    Python Inference script execution over" 

exit 1


