# AMS Izziv VoxelMorph

Pozdravljeni v repozitoriju za **VoxelMorph** registracijo CT in CBCT slik!  
V nadaljevanju so navodila za gradnjo Docker slike, zagon okolja in pridobitev deformacijskih polj.



## Pregled

Ta projekt uporablja [VoxelMorph](https://github.com/voxelmorph/voxelmorph) za deformabilno registracijo medicinskih slik (C (\leftrightarrow\) CBCT). Vse zasluge gredo originalnim avtorjem. 


## Struktura Repozitorija

```plaintext
AMS_Izziv_VoxelMorph/
├── Code/
│   └── Docker/
│       ├── Dockerfile
│       ├── train.py
│       ├── test.py
│       ├── requirements.txt
│       ...
├── .gitignore
└── README.md  (ta datoteka)
```

V mapi ```Code/Docker/``` se nahaja vsa glavna koda:

- ```dockerfile``` (gradnja slike)
- ```train.py``` (skripta za treniranje modela VoxelMorph)
- ```test.py``` (skripta za generiranje deformacijskih polj)
- ```requirements.txt``` (zahtevane Python knjižnice)

## 1. Gradnja Docker image-a
Poženemo ukaz:
```
git clone https://github.com/Crynetix/AMS_Izziv_VoxelMorph.git
```
Najprej se premaknemo v mapo kjer je ```dockerfile```. Npr:
```
cd AMS_Izziv_VoxelMorph/Code/Docker
```
tu zaženemo
```
docker build -t voxelmorph-izziv .
```

## 2. Zagon docker containerja

Ko je slika zgrajena, lahko ustvarimo interaktivni kontejner:
```
docker run --runtime=nvidia -it \
    -v /pot/do/podatkov:/app/data \
    -v /pot/do/nekega_output_folderja:/app/output \
    --name voxelmorph_container \
    voxelmorph-izziv /bin/bash
```

- ```--runtime=nvidia``` Omogoča dostop do GPU (če je na voljo).
- ```-v /path/do/podatkov:/app/data``` Montira lokalno mapo s podatki (npr. slike in .json) v /app/data znotraj kontejnerja.
- ```-v /path/do/nekega_output_folderja:/app/output``` Mapa, kamor želite shraniti končne deformacijske mreže (npr. disp_...nii.gz).
- ```--name voxelmorph_container``` Poimenujete kontejner za lažjo referenco.
- ```/bin/bash``` Odpre interaktivno konzolo v kontejnerju.

## 3. Treniranje Modela ```train.py```

V containerju z naslednjim ukazom zaženemo treniranje:
```
python3 train.py \
    --arguments value
```
Tukaj je primer ukaza:
```
python3 train.py \
    --data_dir /app/data \
    --json_path ThoraxCBCT_dataset.json \
    --train_key training_paired_images \
    --val_key registration_val \
    --gpu 0 \
    --epochs 100 \
    --steps_per_epoch 20 \
    --batch_size 2 \
    --learning_rate 0.0001 \
    --downsample_factor 1.0 \
    --normalize
```
Kjer so:
- ```--data_dir /app/data``` kaže na pot znotraj containerja (montirano izven).
- ```--json_path ThoraxCBCT_dataset.json``` je ime datoteke, ki opisuje pare za treniranje/validacijo.
- ```--train_key```, ```--val_key``` določata ključa v ```.json```, kjer so definirani trening in validacija.
- ```--gpu 0``` izbere GPU s številko 0 (če jih imate več).

Za pomoč:
```
python3 train.py --help
```

Skripta shranjuje uteži modelov v ```/app/data/models/<datum>/<run_xx>/voxelmorph_model_epoch_<epoch_number>.h5 ```

## 4. Generiranje deformacijskih polj ```test.py```

Po uspešnem treniranju lahko generiramo deformacijska polja za želene kombinacije slik v ```.json```. Zaženemo npr:
```
python3 test.py \
    --weights /pot/do/model_weights \
    --subset registration_val \
    --json /app/data/ThoraxCBCT_dataset.json \
    --output /app/output \
    --gpu 0
```

- Naložimo izbrani model ```--weights```. 
- Vzamemo pare (npr. ```registration_val```) iz ```.json```.
- Za vsak par izračunamo in shranimo ```disp_<fixed>_<moving>.nii.gz``` v ```/app/output```.

Za pomoč:
```
python3 test.py --help
```

V lokalni mapi (zunaj containerja), ki je montirana kot ```/app/output```, dobimo ```.nii.gz``` datoteke z deformacijskimi polji.

## 5. Validacija rezultatov

Za test slik je uporabljen testni docker iz lst gitlaba

Navodila za prenos in zagon:

Kloniraj testni repozitorij

```bash
git clone https://gitlab.lst.fe.uni-lj.si/domenP/deformable-registration.git
```

Buildaj docker image
```bash
docker build -t my-deformable-image .
```

Zaženi testni docker
```bash
docker run \
    --rm \
    -u $UID:$UID \
    -v ./path/to/registered/images:/input \
    -v ./path/to/desired/output:/output/ \
    -v ./path/to/data/json:/workspace/data \
    gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration \
    python evaluation.py -v
```

## 5. Rezultati

```
 aggregated_results:
        LogJacDetStd        : 0.55240 +- 0.36495 | 30%: 0.88921
        TRE_kp              : 10.33477 +- 2.64358 | 30%: 11.77554
        TRE_lm              : 10.67345 +- 3.49542 | 30%: 10.79263
        DSC                 : 0.31413 +- 0.06497 | 30%: 0.28895
        HD95                : 49.49817 +- 13.71415 | 30%: 37.51035
```

## 6. Primeri zagona (ukazov)

1. Kloniranje in priprava:
```
git clone https://github.com/Crynetix/AMS_Izziv_VoxelMorph.git
cd AMS_Izziv_VoxelMorph/Code/Docker
```
2. Build
```
docker build -t voxelmorph-izziv .
```
3. Zagon dockerja
```
docker run -it \
    --runtime=nvidia \
    -v /home/uporabnik/data/Izziv_data/Release_06_12_23:/app/data \
    -v /home/uporabnik/OutputZaDeformacije:/app/output \
    --name matejh_container \
    voxelmorph-izziv \
    /bin/bash
```
4. Treniranje (znotraj containerja)
```
python3 train.py \
    --data_dir /app/data \
    --json_path ThoraxCBCT_dataset.json \
    --train_key training_paired_images \
    --val_key registration_val \
    --epochs 50 \
    --steps_per_epoch 25 \
    --batch_size 2 \
    --learning_rate 0.0001 \
    --gpu 0 \
    --downsample_factor 0.5 \
    --normalize
```
5. Test (znotraj containerja)
```
python3 test.py \
    --weights /app/data/models/2024-12-20/run_01/voxelmorph_model_epoch_050.h5 \
    --subset registration_val \
    --json /app/data/ThoraxCBCT_dataset.json \
    --output /app/output \
    --gpu 0
```