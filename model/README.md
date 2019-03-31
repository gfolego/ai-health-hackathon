# Experiment for model training and generation

Reference: https://nihcc.app.box.com/v/ChestXray-NIHCC/

Download dataset images
```bash
$ sudo apt-get update && \
    apt-get install --yes aria2

$ aria2c --input-file=list.txt --continue=true \
    --check-integrity=true  --max-tries=0 \
    --summary-interval=600  --max-connection-per-server=8 \
    --min-split-size=1M

$ for f in "image*.gz" do; tar xf $f; done
```

Prepare dataset csv
```bash
$ head -n1 Data_Entry_2017.csv > dataset_test.csv

$ join -t, <(sort Data_Entry_2017.csv) <(sort test_list.txt) >> dataset_test.csv
```

Run with nvidia-docker to use GPU
```bash
$ docker build --tag hack-gpu --file Dockerfile_gpu --pull .

$ nvidia-docker run -ti --name hack-gpu --volume "$PWD":/work hack-gpu bash

$ python3 /work/train.py --help
```


