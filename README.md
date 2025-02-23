# Leveraging Language to Generalize Natural Images to Few-shot Medical Image Segmentation

This is the implementation of the paper "Leveraging Language to Generalize Natural Images to Few-shot Medical Image Segmentation". For more information, check out the [\[paper\]]().

## Introduction
Cross-domain Few-shot Medical Image Segmentation (CD-FSMIS) typically involves pre-training on a large-scale source domain dataset (e.g., natural image dataset) before transferring to a target domain with limited data for pixel-wise segmentation. However, due to the significant domain gap between natural images and medical images, existing Few-shot Segmentation (FSS) methods suffer from severe performance degradation in cross-domain scenarios. We observe that using only annotated masks as cross-domain cues is insufficient, while rich textual information can effectively establish knowledge relationships between visual instances and language descriptions, mitigating domain shift. To address this, we propose a plug-in Cross-domain Text-guided (CD-TG) module that leverages text-domain alignment to construct a new alignment space for domain generalization. This plug-in module consists of two components, including: (1) Text Generation Unit that utilizes the GPT-4 question-answering system to generate standardized category-level textual descriptions, and (2) Semantic-guided Unit that aligns visual and linguistic support feature embeddings while incorporating existing mask information. We integrate this plug-in module into five mainstream FSS methods and evaluate it on four widely used medical image datasets. Experimental results demonstrate its effectiveness. 

<p align="middle">
    <img src="data/assets/frame.png">
</p>
We study the CD-FSS problem, where the source and target domains have completely disjoint label space and cannot access target domain data during the training stage. 


## Datasets
The following datasets are used for evaluation in CD-FSS:

### Source domain: 

* **PASCAL VOC2012**:

    Download PASCAL VOC2012 devkit (train/val data):
    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    ```
    Download PASCAL VOC2012 SDS extended mask annotations from [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].
  
* **MSCOCO2012**:

    Download MS COCO 2012 dataset (train/val data):
    ```bash
    wget http://images.cocodataset.org/zips/train2012.zip
    wget http://images.cocodataset.org/zips/val2012.zip
    ```
    Download MS COCO 2012 annotations and segmentation masks:
    ```bash
    wget http://images.cocodataset.org/annotations/annotations_trainval2012.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2012.zip
    ```


### Target domains: 

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018
    
    Class Information: data/isic/class_id.csv

* **Chest X-ray**:

    Home: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

    Direct: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
  
* **WBC**:

    Home: https://www.sciencedirect.com/science/article/pii/S0968432817303037

    Direct: https://github.com/zxaoyou/segmentation_WBC

* **CHAOS-MRI**:

    Home: http://www.sciencedirect.com/science/article/pii/S1361841520303145

    Direct: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14
- open-clip

Conda environment settings:
```bash
conda create -n patnet python=3.7
conda activate patnet

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```

## Ablation Studies

## Visualization

## Citation
If you use this code for your research, please consider citing:
```bash
@inproceedings{lei2022cross,
   title={Cross-Domain Few-Shot Semantic Segmentation},
   author={Lei, Shuo and Zhang, Xuchao and He, Jianfeng and Chen, Fanglan and Du, Bowen and Lu, Chang-Tien},
   booktitle={European Conference on Computer Vision},
   pages={73--90},
   year={2022},
   organization={Springer}
 }
 ```

## Acknowledgement
The implementation is based on [HSNet](https://github.com/juhongm999/hsnet). <br>

## References

[1] Demir, I., Koperski, K., Lindenbaum, D., Pang, G., Huang, J., Basu, S., Hughes,
F., Tuia, D., Raskar, R.: Deepglobe 2018: A challenge to parse the earth through
satellite images. In: The IEEE Conference on Computer Vision and Pattern Recog-
nition (CVPR) Workshops (June 2018)Li, X., Wei, T., Chen, Y.P., Tai, Y.W., Tang, C.K.: Fss-1000: A 1000-class dataset
for few-shot segmentation. In: Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition. pp. 2869–2878 (2020)

[2] Codella, N., Rotemberg, V., Tschandl, P., Celebi, M.E., Dusza, S., Gutman, D.,
Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., et al.: Skin lesion analysis toward
melanoma detection 2018: A challenge hosted by the international skin imaging
collaboration (isic). arXiv preprint arXiv:1902.03368 (2019)

[3] Tschandl, P., Rosendahl, C., Kittler, H.: The ham10000 dataset, a large collection
of multi-source dermatoscopic images of common pigmented skin lesions. Scientific
data 5, 180161 (2018)

[4] Candemir, S., Jaeger, S., Palaniappan, K., Musco, J.P., Singh, R.K., Xue, Z.,
Karargyris, A., Antani, S., Thoma, G., McDonald, C.J.: Lung segmentation in
chest radiographs using anatomical atlases with nonrigid registration. IEEE trans-
actions on medical imaging 33(2), 577–590 (2013)

[5] Jaeger, S., Karargyris, A., Candemir, S., Folio, L., Siegelman, J., Callaghan, F.,
Xue, Z., Palaniappan, K., Singh, R.K., Antani, S., et al.: Automatic tuberculosis
screening using chest radiographs. IEEE transactions on medical imaging 33(2),
233–245 (2013)

[6] Li, X., Wei, T., Chen, Y.P., Tai, Y.W., Tang, C.K.: Fss-1000: A 1000-class dataset
for few-shot segmentation. In: Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition. pp. 2869–2878 (2020)

