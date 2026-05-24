# PA3AD: Physics-inspired Pseudo Anomaly Generation and Prototype Feature Guidance for 3D Anomaly Detection

<p align="center">
  <a href="#-news"><img alt="News" src="https://img.shields.io/badge/News-Code%20Released-brightgreen"></a>
  <a href="#-installation"><img alt="Python" src="https://img.shields.io/badge/python-3.10-blue"></a>
  <a href="#-installation"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3.1-red"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

Official repository for the paper **"Physics-inspired Pseudo Anomaly Generation and Prototype Feature Guidance for 3D Anomaly Detection"**. 

3D point cloud anomaly detection plays a vital role in industrial manufacturing but faces significant challenges due to the scarcity and high acquisition cost of real anomalous samples. To address the difficulty of learning discriminative features from inherently anomaly-free training data, we propose **PA3AD**.

## 📢 News
- **[2026-05]** Source code is released!
- **[2026-03]** The repository is created.
- **[2026-03]** Our paper has been submitted to *Pattern Recognition*.

## 💡 Key Innovations

PA3AD tackles the scarcity of real anomalies and distribution shifts through two core designs:
1. **Physics-Inspired Pseudo-Anomaly Generation**: A novel module that creates diverse, physically plausible pseudo-anomalous point clouds from normal data via multi-physics modeling.
2. **Prototype Feature Guidance**: We incorporate momentum-updated prototypes and a difference-aware fusion block via a weight-sharing mechanism. This effectively captures stable normal representations and their discrepancies with pseudo-anomalies.

<p align="center">
  <img src="assets/architecture.png" alt="PA3AD Architecture" width="800"/>
</p>

## 📂 Datasets

Our method is evaluated on the following standard industrial benchmarks:
- **Anomaly-ShapeNet**
- **Real3D-AD**

Please organize the datasets as follows:

```
PA3AD/
├── datasets/
│   ├── AnomalyShapeNet/
│   │   └── dataset/
│   │       ├── obj/
│   │       │   ├── ashtray0/
│   │       │   │   ├── template_0.obj
│   │       │   │   └── ...
│   │       │   ├── bottle0/
│   │       │   └── ...
│   │       └── pcd/
│   │           ├── ashtray0/
│   │           │   ├── test/
│   │           │   │   ├── positive_xxx.pcd
│   │           │   │   └── negative_xxx.pcd
│   │           │   └── GT/
│   │           │       └── negative_xxx.txt
│   │           └── ...
│   └── Real3D/
│       └── Real3D-AD-PLY/
│           ├── airplane/
│           │   ├── template_0.ply
│           │   └── ...
│           ├── candybar/
│           └── ...
```

## 🛠️ Installation

```bash
git clone https://github.com/NingxiaoJian/PA3AD.git
cd PA3AD
```

**Option 1: Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate PA3AD
```

**Option 2: Pip**
```bash
# First install PyTorch with CUDA 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# Then install other dependencies
pip install -r requirements.txt
```

## 🚀 Training

**Anomaly-ShapeNet:**
```bash
python PA3AD_training.py --dataset AnomalyShapeNet --category ashtray0 --batch_size 16 --epochs 2000 --lr 0.001 --optimizer AdamW --backbone_arch MinkUNet34TransformerLocalGlobal --logpath ./log/AnomalyShapeNet/ashtray0/ --use_shared_backbone --data_repeat 100
```

**Real3D-AD:**
```bash
python PA3AD_training.py --dataset Real3D --category airplane --batch_size 16 --epochs 2000 --lr 0.001 --optimizer AdamW --backbone_arch MinkUNet34TransformerLocalGlobal --logpath ./log/Real3D/airplane/ --use_shared_backbone --data_repeat 100
```

## 📊 Evaluation

**Anomaly-ShapeNet:**
```bash
python PA3AD_eval.py --dataset AnomalyShapeNet --category ashtray0 --backbone_arch MinkUNet34TransformerLocalGlobal --logpath ./weights/AnomalyShapeNet/ --checkpoint_name ashtray0_model.pth --use_shared_backbone
```

**Real3D-AD:**
```bash
python PA3AD_eval.py --dataset Real3D --category airplane --backbone_arch MinkUNet34TransformerLocalGlobal --logpath ./weights/Real3D/ --checkpoint_name airplane_model.pth --use_shared_backbone
```

## 📄 Citation

If you find this work useful, please cite:
```bibtex
@article{pa3ad2026,
  title={Physics-inspired Pseudo Anomaly Generation and Prototype Feature Guidance for 3D Anomaly Detection},
  author={},
  journal={Pattern Recognition},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

