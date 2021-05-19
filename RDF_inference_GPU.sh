#!/bin/bash

#PBS -N FloodDAM
#PBS -q qgpgpu
#PBS -l select=1:ncpus=8:mem=40000mb:ngpus=1
#PBS -l walltime=00:05:00
#PBS -o /work/scratch/fatrasc/FLOODDAM/FDAM_TSX/LOG
#PBS -e /work/scratch/fatrasc/FLOODDAM/FDAM_TSX/LOG

echo ""
echo "###############################################################"
echo "    Hello from Inference script!    "
echo "###############################################################"
echo ""

module load conda
conda activate rapids-0.16
echo "Environment rapids-0.16 loaded"

cd /home/eh/fatrasc/scratch/FLOODDAM/FDAM_TSX/wp3-rapid-mapping-4-terrasarx-compatibility

echo "Executing python Inference script..."

# Configuration
infold='/work/OT/floodml/data/deliveries/phase-1-cls/'
oufold='/work/scratch/fatrasc/FLOODDAM/FDAM_TSX/Inferences/'
medir='/work/OT/floodml/data/deliveries/phase-1-cls/MERIT_S2/'
cedir='/work/datalake/static_aux/MNT/Copernicus_DSM/'
type=1
dbpath='/work/scratch/fatrasc/FLOODDAM/DB_RDF/DB_RDF_global_S1_VVseul.sav'
gswdir='/work/OT/floodml/data/deliveries/phase-1-cls/GSW_Tiled/'


#infold='/work/OT/floodml/data/inputs_tsx/TSX1_SAR__EEC_RE___SL_D_SRA_20160110T063435_20160110T063436' #Irlande
#infold='/work/OT/floodml/data/inputs_tsx/TSX1_SAR__EEC_RE___SM_D_SRA_20171202T155847_20171202T155855' #Albanie
#infold='/work/OT/floodml/data/inputs_tsx/TSX1_SAR__EEC_RE___SM_S_SRA_20180204T055255_20180204T055300' #Paris
#infold='/work/OT/floodml/data/inputs_tsx/TSX1_SAR__EEC_RE___SM_S_SRA_20180204T163537_20180204T163543' #Paris
infold='/work/OT/floodml/data/inputs_tsx/TDX1_SAR__EEC_RE___SM_S_SRA_20180131T174358_20180131T174404' #Paris

oufold='/work/scratch/fatrasc/FLOODDAM/FDAM_TSX/Inferences/Paris'
dbpath='/work/scratch/fatrasc/FLOODDAM/DB_RDF/DB_RDF_global_S1_VVseul.sav'
python RDF-3-inference.py -i $infold -o $oufold -c $cedir --satellite tsx -db $dbpath --gsw $gswdir 


echo "    Python Inference script execution over"

exit 1

#parser.add_argument('-i', '--input', help='Input EMSR folder', type=str, required=True)
#parser.add_argument('-o', '--Inf_ouput', help='Output folder', type=str, required=True)
#parser.add_argument('-m', '--meritdir', help='MERIT DEM folder.'
#                                                 'Either this or --copdemdir has to be set for sentinel 1.',
#                        type=str, required=False)
#parser.add_argument('-c', '--copdemdir', help='Copernicus DEM folder.'
#                                                  'Either this or --meritdir has to be set for sentinel 1.',
#                        type=str, required=False)
#parser.add_argument('--sentinel', help='S1 or S2', type=int, required=True, choices=[1, 2])
#parser.add_argument('-db', '--db_path', help='Learning database filepath', type=str, required=True)
#parser.add_argument('-tmp', '--tmp_dir', help='Global DB output folder ', type=str, required=False, default="tmp")
#parser.add_argument('-g', '--gsw', help='Tiled GSW folder', type=str, required=True)
#parser.add_argument('-ti', '--tile_ref', help='Input tile ref', type=str, required=False)
#parser.add_argument('-orb', '--orbit', help='Input orbit number', type=str, required=False)
#parser.add_argument('-d', '--date', help='Input date tag ', type=str, required=False)
#parser.add_argument('-ot', '--outag', help='Output suffix tag ', type=str, required=False)





