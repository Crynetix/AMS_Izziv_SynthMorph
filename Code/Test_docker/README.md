# Deformable image registration
[Datasets](#datasets) | [Evaluation](#evaluation) | [Environment](#environment-setup)


## Datasets

### Oncoreg

For evaluation purposes we are using the [OncoReg](https://learn2reg.grand-challenge.org/oncoreg/) challenge dataset. 
You can download the data from [here](https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg). 

The dataset has the following data structure:

├── [imagesTr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/imagesTr): CT and CBCT data (CT is 0000, CBCTs are 0001 and 0002)

├── [keypoints01Tr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/keypoints01Tr): keypoints for CT-CBCT01, can be used for registration

├── [keypoints02Tr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/keypoints02Tr): keypoints for CT-CBCT02, can be used for registration

├── [labelsTr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/labelsTr): segmentations

├── [landmarks01Tr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/landmarks01Tr): landmarks for CT-CBCT01

├── [landmarks02Tr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/landmarks02Tr): landmarks for CT-CBCT02

├── [masksTr](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/masksTr): masks

├── [info.txt](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/info.txt): read more about the dataset here

└── [ThoraxCBCT_dataset.json](../../mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23/ThoraxCBCT_dataset.json): specifies data structure and the **train/validation/test split!**

## Evaluation

According to the evaluation protocol and train/validation/test splits of the
[OncoReg](https://learn2reg.grand-challenge.org/oncoreg/) challenge
we computed the deformation fields for the following CT-to-CBCT registration pairs (cf. [imagesTr](#oncoreg)) that
form the validation split:
```bash
0011_0001<--0011_0000
0012_0001<--0012_0000
0013_0001<--0013_0000
0011_0002<--0011_0000
0012_0002<--0012_0000
0013_0002<--0013_0000
```

Evaluation input must include files in format like: `disp_0011_0001_0011_0000.nii.gz`, where `0011` is *case*, `0011_0000` *(CT)* registered to `0011_0001` *(CBCT)*.

**NOTE**: Sample registration files are located on `ZigaPublic Worker (192.168.32.141)` at: `/shared/dart/deformable_registration/vxmpp_results/input` and can be used as example.

## Evaluation using docker - *EASY*

To obtain results, run the following `docker run` command, with proper `./input` and `./output` directories:

```bash
docker run \
    --rm \
    -u $UID:$UID \
    -v ./input:/input \
    -v ./output:/output/ \
    gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration \
    python evaluation.py -v
```

In this case, our results are located in `./input`, and metrics will be output in `./output/metrics.json`.

For more adjustment options run the command with `python evaluation.py --help`.

**DEMO**

```bash
docker run \
    --rm \
    -u $UID:$UID \
    -v /shared/dart/deformable_registration/vxmpp_results/input:/input \
    -v ./output:/output/ \
    gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration \
    python evaluation.py -v
```

Output data will be located at `./output`.

**BUILD IMAGE**

To build your own image, use:

```bash
docker build -t my-deformable-image .
```

But make sure to include data:

```bash
cd data
wget https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg/download/ThoraxCBCT_OncoRegRelease_06_12_23.zip
unzip ThoraxCBCT_OncoRegRelease_06_12_23.zip
rm -r __MACOSX/
rm ThoraxCBCT_OncoRegRelease_06_12_23.zip
```
Data will be in: *Release_06_12_23* folder.


## Evaluation using pyenv - *COMPLEX*

To obtain the results for the `vxmpp` run the following script:

```bash
export PYTHONPATH=<path-to-repo>:<path-to-repo/modules/L2R/evaluation>
python utils/evaluation.py 
  -i <path-to-deformation-fields> 
  -d <path-to-oncoreg-data-folder> 
  -o <path-to-output-metrics-json> 
  -c data/configs/ThoraxCBCT_VAL_evaluation_config.json
  -v
```

### My example (*see paths to get the example data*)

Example evaluation is demonstrated for the 
 [VoxelmorphPlusPlus](https://github.com/mattiaspaul/VoxelMorphPlusPlus) 
registration method (abbreviated hereafter as `vxmpp`). 

The following command was used to perform evaluation:
```bash
export PYTHONPATH=/home/zigaso/dev/deformable-registration:/home/zigaso/dev/deformable-registration/modules/L2R/evaluation
python utils/evaluation.py 
  -i /home/zigaso/mountpoints/nasx/medical/projects/dart/models/oncoreg/vxmpp/results 
  -d /home/zigaso/mountpoints/nasx/medical/projects/dart/data/oncoreg/Release_06_12_23 
  -o /home/zigaso/dev/deformable-registration/data/evaluation/vxmpp/metrics.json 
  -c /home/zigaso/dev/deformable-registration/data/configs/ThoraxCBCT_VAL_evaluation_config.json
  -v
```

The console output is here:
```bash
 aggregated results:
	LogJacDetStd        : 0.10320 +- 0.00834 | 30%: 0.10965
	TRE_kp              : 7.69219 +- 2.03832 | 30%: 7.36717
	TRE_lm              : 8.51688 +- 3.86951 | 30%: 8.24852
	DSC                 : 0.55044 +- 0.13053 | 30%: 0.49095
	HD95                : 33.87484 +- 10.90554 | 30%: 23.67392
```

You can find the resulting `.json` in this repo [metrics.json](data/evaluation/vxmpp/metrics.json).

## Environment setup

See tutotial on setting up Python environment using `pyenv`:
[Medium pyenv tutorial](https://medium.com/@adocquin/mastering-python-virtual-environments-with-pyenv-and-pyenv-virtualenv-c4e017c0b173)

When the `pyenv install` yield errors, there are missing deps. See solution with the highest ranking: 
[stack overflow post](https://stackoverflow.com/questions/38701203/error-the-python-ssl-extension-was-not-compiled-missing-the-openssl-lib).

This worked:
```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git 
```

The python env was setup using:
```bash
pyenv install 3.10.15
pyenv virtualenv 3.10.15 l2r
pyenv virtualenvs # to see installed virtual envs 
pyenv activate l2r
```
To finalize installation go to `modules/L2R/evaluation` and run:
```bash
pip install -r requirements.txt
```

*Note*: you might need to update the `pip` to newest version.

## Authors and acknowledgment

Thanks to Wiebke and Mattias from the [OncoReg](https://learn2reg.grand-challenge.org/oncoreg/) challenge team 
for their help in running the evaluation.
