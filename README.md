# AMS Izziv VoxelMorph

Pozdravljeni v repozitoriju za **VoxelMorph** registracijo CT in CBCT slik!  
V nadaljevanju so navodila za gradnjo Docker slike, zagon okolja in pridobitev deformacijskih polj.

---

## Pregled

Ta projekt uporablja [VoxelMorph](https://github.com/voxelmorph/voxelmorph) za deformabilno registracijo medicinskih slik (CT ↔ CBCT). Vse zasluge gredo originalnim avtorjem.

---

## Struktura Repozitorija

```plaintext
AMS_Izziv_VoxelMorph/
├── Code/
│   └── Docker/
│       ├── Dockerfile
│       ├── train.py
│       ├── test.py
│       ├── requirements.txt
│   Models/2024-12-19/run_18/
        ├── voxelmorph_model_epoch_250.h5
├── .gitignore
└── README.md  (ta datoteka)
```

V mapi `Code/Docker/` se nahaja vsa glavna koda:

- `Dockerfile` (gradnja slike)
- `train.py` (skripta za treniranje modela VoxelMorph)
- `test.py` (skripta za generiranje deformacijskih polj)
- `requirements.txt` (zahtevane Python knjižnice)

---

## 1. Gradnja Docker slike

Poženemo ukaz:
```bash
git clone https://github.com/Crynetix/AMS_Izziv_VoxelMorph.git
```

Najprej se premaknemo v mapo, kjer je `Dockerfile`. Npr.:
```bash
cd AMS_Izziv_VoxelMorph/Code/Docker
```

Tu zaženemo:
```bash
docker build -t voxelmorph-izziv .
```

---

## 2. Zagon Docker containerja

Ko je slika zgrajena, lahko ustvarimo interaktivni kontejner:
```bash
docker run --runtime=nvidia -it \
    -v /pot/do/podatkov:/app/data \
    -v /pot/do/nekega_output_folderja:/app/output \
    --name voxelmorph_container \
    voxelmorph-izziv /bin/bash
```

- `--runtime=nvidia`: Omogoča dostop do GPU (če je na voljo).
- `-v /path/do/podatkov:/app/data`: Montira lokalno mapo s podatki (npr. slike in .json) v `/app/data` znotraj kontejnerja.
- `-v /path/do/nekega_output_folderja:/app/output`: Mapa, kamor želite shraniti končne deformacijske mreže (npr. `disp_...nii.gz`).
- `--name voxelmorph_container`: Poimenujete kontejner za lažjo referenco.
- `/bin/bash`: Odpre interaktivno konzolo v kontejnerju.

---

## 3. Treniranje modela (`train.py`)

V containerju z naslednjim ukazom zaženemo treniranje:
```bash
python3 train.py \
    --arguments value
```

Primer ukaza:
```bash
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
```

Parametri:
- `--data_dir /app/data`: Kaže na pot znotraj containerja (montirano izven).
- `--json_path ThoraxCBCT_dataset.json`: Datoteka, ki opisuje pare za treniranje/validacijo.
- `--train_key`, `--val_key`: Ključi v `.json`, kjer so definirani trening in validacija.
- `--gpu 0`: Izbere GPU s številko 0 (če jih imate več).

Za pomoč:
```bash
python3 train.py --help
```

Uteži modela se shranjujejo v:
```
/app/data/models/<datum>/<run_xx>/voxelmorph_model_epoch_<epoch_number>.h5
```

---

## 4. Generiranje deformacijskih polj (`test.py`)

Po uspešnem treniranju lahko generiramo deformacijska polja za želene kombinacije slik v `.json`. Primer ukaza:
```bash
python3 test.py \
    --weights /pot/do/model_weights \
    --subset registration_val \
    --json /app/data/ThoraxCBCT_dataset.json \
    --output /app/output \
    --gpu 0
```

Parametri:
- `--weights`: Pot do uteži modela.
- `--subset`: Delitev podatkov (npr. `registration_val`).
- `--json`: JSON datoteka z opisi slik.
- `--output`: Izhodna mapa za `.nii.gz` datoteke.

Za pomoč:
```bash
python3 test.py --help
```

---

## 5. Validacija rezultatov

Validacija se izvaja z uporabo docker slike za evalvacijo. Predpogoj je, da imate deformacijska polja v pravilni strukturi, kot je npr.:
```plaintext
disp_<case>_<cbct>_<ct>.nii.gz
```
Primer za CT ↔ CBCT:
```
disp_0011_0001_0011_0000.nii.gz
```
Tu `0011_0000` predstavlja CT, ki se registrira na CBCT `0011_0001`.

### Zagon evalvacije z dockerjem

Zaženemo:
```bash
docker run \
    --rm \
    -u $UID:$UID \
    -v ./input:/input \
    -v ./output:/output \
    gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration \
    python evaluation.py -v
```

- `./input`: Vhodna mapa z deformacijskimi polji (`disp_*.nii.gz`).
- `./output`: Izhodna mapa, kjer bodo shranjeni rezultati (`metrics.json`).

### Primer evalvacije

Primer evalvacijskega scenarija za par CT-CBCT iz validacijske delitve (OncoReg):
```bash
0011_0001 ↔ 0011_0000
0011_0002 ↔ 0011_0000
```
Rezultati bodo shranjeni v datoteko `metrics.json` z naslednjimi metrikami:
- **LogJacDetStd**: Standardna deviacija log-determinante Jacobiana.
- **TRE_kp**: Povprečna registracijska napaka za ključne točke.
- **DSC**: Dice Similarity Coefficient za segmente.
- **HD95**: Hausdorffova razdalja (95. percentil).

---

## 6. Primeri zagona

1. Kloniranje in priprava:
```bash
git clone https://github.com/Crynetix/AMS_Izziv_VoxelMorph.git
cd AMS_Izziv_VoxelMorph/Code/Docker
```

2. Gradnja Docker slike:
```bash
docker build -t voxelmorph-izziv .
```

3. Zagon Docker containerja:
```bash
docker run -it \
    --runtime=nvidia \
    -v /home/uporabnik/data/Izziv_data/Release_06_12_23:/app/data \
    -v /home/uporabnik/OutputZaDeformacije:/app/output \
    --name matejh_container \
    voxelmorph-izziv \
    /bin/bash
```

4. Treniranje modela (znotraj containerja):
```bash
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

5. Generiranje deformacijskih polj (znotraj containerja):
```bash
python3 test.py \
    --weights /app/data/models/2024-12-19/run_18/voxelmorph_model_epoch_250.h5 \
    --subset registration_val \
    --json /app/data/ThoraxCBCT_dataset.json \
    --output /app/output \
    --gpu 0
```

6. Validacija rezultatov:
```bash
docker run \
    --rm \
    -u $UID:$UID \
    -v ./input:/input \
    -v ./output:/output \
    gitlab.lst.fe.uni-lj.si:5050/domenp/deformable-registration \
    python evaluation.py -v
```

---

## 7. Rezultati validacije

Primer izhodnih metrik za validacijske pare:
```plaintext
aggregated_results:
    LogJacDetStd        : 0.55240 +- 0.36495 | 30%: 0.88921
    TRE_kp              : 10.33477 +- 2.64358 | 30%: 11.77554
    DSC                 : 0.31413 +- 0.06497 | 30%: 0.28895
    HD95                : 49.49817 +- 13.71415 | 30%: 37.51035
```
---

