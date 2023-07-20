<h1 align="center">
  <br>
Do Perceptually Aligned Gradients Imply Robustness?
  <br>
</h1>
<p align="center">
  <a href="https://royg27.github.io/">Roy Ganz</a> •
  <a href="https://bahjat-kawar.github.io">Bahjat Kawar</a> •
  <a href="https://elad.cs.technion.ac.il/">Michael Elad</a>
</p>
<p align="center">
[ICML 2023 - Oral] Official Code Repository of <a href="https://arxiv.org/abs/2207.11378">Do Perceptually Aligned Gradients Imply Robustness?</a>
</p>

<p align="center">
  <img src="https://github.com/royg27/PAG-ROB/blob/main/PAG-ROB.png" />
</p>

### Installation

First, clone this repository:

```bash
git clone https://github.com/royg27/PAG-ROB.git
cd PAG-ROB
```

Next, to install the requirements in a new conda environment, run:

```bash
conda env create -f environment.yml
```

### Preparing Perceptually Aligned Gradients Data

The Perceptually Aligned Gradients' realization for the Score-Based Gradients for the CIFAR-10 dataset is provided in the following table:

PAG realization | Data | Labels
--- | :---: | :---: 
Score-Based Gradients |  <a href="https://drive.google.com/file/d/1kpUNM3j7V_YxQ7xQuDe_M_hHZSOvkCOx/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/12pIOWxCHCLjlUPwxvFS87u93GBqWO9XK/view?usp=drive_link">Download</a>

The data should be placed in the data folder, forming the following structure:

    PAG-ROB
    ├── configs
    │   ├── ......
    ├── data
    │   ├── c10_sbg_data.pt
    │   ├── c10_sbg_label.pt
    ├── models
    │   ├── ......
    ├── TRAIN_CIFAR10.py
    
### Training

```bash
python TRAIN_CIFAR10.py --config_path <config>
```

where `<config>` specifies the desired training configuration (e.g., `configs/cifar10_sbg_rn18.yaml`)

### Trained Checkpoints

We provide pretrained checkpoints on the CIFAR-10 dataset in the table below:

RN18 OI | RN18 CM | RN18 NN | RN18 SBG | ViT SBG
--- | :---: | :---: | :---: | :---: 
<a href="https://drive.google.com/file/d/1R-Cp2-wIi1JpG6-YLBvqV-quoBetmV1e/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1lPkiKzJPU25hMxNZWv-8tYk9-BwoQL7f/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1AuBppVh9ghRXxNnJvuBjRg9qv4rCIuIE/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1vdtGi_DjhWOlPwuk7d-RVU1amnVfz8z_/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/18cYjbwdAUdH8jKx11py9NW_ovXVbtpL0/view?usp=drive_link">Download</a> |



### Citation
If you find this code or data to be useful for your research, please consider citing it.


    @misc{ganz2023perceptually,
          title={Do Perceptually Aligned Gradients Imply Adversarial Robustness?}, 
          author={Roy Ganz and Bahjat Kawar and Michael Elad},
          year={2023},
          eprint={2207.11378},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

