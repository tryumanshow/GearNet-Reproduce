< Paper Reproduction Just for Fun >  

# Protein Representation Learning by Geometric Structure Pretraining 
- [Paper Link](https://arxiv.org/abs/2203.06125)
- Author: Zuobai Zhang, Minghao Xu, Arian Jamasb, Vijil Chenthamarakshan, Aurelie Lozano, Payel Das, Jian Tang  
- Reproduced by: Seungwoo Ryu

------  
- __Suppose all the snippets below start from your own root directory.__  
  + Downloaded folder name is assumed to be a 'GearNet'.

### Pretraining Dataset  
- Instead of using `AlphaFoldDB`(805K) for pretraining, I used `Swiss-Prot`(540K) protein dataset.  
  Disparity of the pretraining dataset can make subtle (or considerable) difference b/w the result of original paper and that of mine.  
  Can download the data at [Here](https://alphafold.ebi.ac.uk/download), or by
  ```
  wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v3.tar -P ./
  ```  
- The expressions/schema of dataset might follow how [doc1](https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_Dec_1996.pdf) or [doc2](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/dealing-with-coordinates) expresses each protein.  


### Downstream Dataset  
- __Special Preprocessing on EC & GO__
- For `EC Number Prediction` and `GO Term Prediction`:
  - First introducted by [Paper](https://www.biorxiv.org/content/10.1101/786236v1)  
  - Caution!
    - It is not possible to use their original data at all. 
      -  As this paper used `contact map` as a feature for the model, they didn't use explicit coordinate information of atoms. Therefore, their preprocessed files do not offer any info. about intact 3D coordinates which is essential on GearNet(-Variants). Even the `.tfrecords` files offered on `Data` section of the [github page](https://github.com/flatironinstitute/DeepFRI) only contain information of contact map.
      - The code of the paper offers preprocessing code in `preprocessing/data_collection.sh`. However, the code in the 20th line
        ```
        wget https://cdn.rcsb.org/resources/sequence/clusters/bc-95.out -O $DATA_DIR/bc-95.out
        ```
        shows an error with the message `Not Found. The requested URL was not found on this server.`. Therefore, retrieving necessary information from original PDB file is impossible, and the command afterward is useless.
  - My strategy is: 
    1. Extract the pdb names from the data split given on the [paper](https://www.biorxiv.org/content/10.1101/786236v1) and gather all.
    2. Based on the collection of the name, download pdb file one by one from the web.
    3. Extract 3D coordinates information from the downloaded files.  

    - After following these steps,  
      EC: {'train': 4, 'valid': 3, 'test': 0} sets are inevitably omitted from the original dataset.  
      GO: {'train': 18, 'valid': 1, 'test': 1} sets are inevitably omitted from the original dataset.  
      whose coordinates are expressed awkward.

  - Download the split info. of original paper by:
    ```
    git clone https://github.com/flatironinstitute/DeepFRI
    mkdir -p downstream/dataset/EC_GO
    cp -r DeepFRI/preprocessing/data/* downstream/dataset/EC_GO/
    ```
  
- For `Fold Classification`  
  - First introduced by [Paper](https://arxiv.org/pdf/2007.06252.pdf)
  - Can download the data at [Here](https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar) or by
    ```
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar" -O HomologyTAPE.zip && rm -rf /tmp/cookies.txt
    ```
- For `Reaction Classification`  
  - Was introduced in a same paper introduced in `Fold Classification`  
  - Can download the data at [Here](https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar) or by  
    ```
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1udP6_90WYkwkvL1LwqIAzf9ibegBJ8rI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1udP6_90WYkwkvL1LwqIAzf9ibegBJ8rI" -O ProtFunct.zip && rm -rf /tmp/cookies.txt
    ```

- After running all the codes above, preparation for data is all done!
  

---  
  
## Preparation for the - 
### Environment   
```

conda create -n GearNet python=3.8.5 
conda activate GearNet
pip install -r requirements.txt

```

### Dataset  
#### └ For Pretraining  
```  
mkdir -p uniprot/dataset
tar -xf swissprot_pdb_v3.tar -C ./uniprot/dataset

mkdir -p uniprot/interim
python GearNet/preprocess/preprocess_pt.py --data_dir ./uniprot/dataset --save_dir ./uniprot/interim
```  
- As mentioned before, the dataset the model is pretrained on is different from the original one.   
  - Swiss-Prot data does not have information about resolution (Appendix G).  
  - The only standard used for filtering: Incorrect records such as `53.353-100.177` at the position of coordinate information.  
    - 4121 proteins among 542380 are excluded.  
  - Additionally, I excluded 3000 datasets for validation.  
  - So, the final number of data in train set is `535259`.  
  
 
#### └ For Downstream Task  
- Although datasets are already prepared in advance following published papers,  
 we need to pre-process more than those as we need 'coordinate' information for GearNet(-variants).
- To extract coordinates info. from raw pdb files and make inputs for model, implement:
  ```
  bash GearNet/preprocess/run_downstream.sh
  ```  

#### └ Or You can download the preprocessed data from 
- Locate all the downloaded folders on the root directory.  
  
  ``` 
  https://drive.google.com/drive/folders/1aE3TPok3YfF-P5mchIbUmMe3195PlY9S?usp=sharing
  ```

---

## Experiment
- Following the original paper, all the experiments are set in a DistributedDataParallel(DDP) setting.
  
### Pretraining
```
bash main.sh pretrain
```
- Can manully change options on `main.sh` script for other options.  
  - For example, if you want to...
    > `Pretrain` the `GearNet-Edge` model with `MultiviewContrastiveLearning` objective on `GPU #0,1`

    set options as
    ```
    gpu="0 1"
    enc_model="GearNet-Edge"
    task_idx=0
    ```


### Downstream
```
bash main.sh downstream
```
- Can manually change options on `main.sh` script, likewise.
- If you want to load pre-trained weights for inference, set `load` option to `True`
  - Because I couldn't train a large model, I don't have any pretrained model to load which is trained on `Pretraining objectives`.  