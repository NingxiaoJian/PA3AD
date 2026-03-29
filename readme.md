# PA3AD: Physics-inspired Pseudo Anomaly Generation and Prototype Feature Guidance for 3D Anomaly Detection

<p align="center">
  <a href="#-news"><img alt="News" src="https://img.shields.io/badge/News-Code%20Coming%20Soon-orange"></a>
  <a href="#-installation"><img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-blue"></a>
  <a href="#-installation"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.10%2B-red"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

Official repository for the paper **"Physics-inspired Pseudo Anomaly Generation and Prototype Feature Guidance for 3D Anomaly Detection"**. 

3D point cloud anomaly detection plays a vital role in industrial manufacturing but faces significant challenges due to the scarcity and high acquisition cost of real anomalous samples. To address the difficulty of learning discriminative features from inherently anomaly-free training data, we propose **PA3AD**.

## 📢 News
- **[2026-03]** The repository is created. **The source code will be released shortly. Stay tuned!**
- **[2026-03]** Our paper has been submitted to *Pattern Recognition*.

## 💡 Key Innovations

PA3AD tackles the scarcity of real anomalies and distribution shifts through two core designs:
1. **Physics-Inspired Pseudo-Anomaly Generation**: A novel module that creates diverse, physically plausible pseudo-anomalous point clouds from normal data via multi-physics modeling.
2. **Prototype Feature Guidance**: We incorporate momentum-updated prototypes and a difference-aware fusion block via a weight-sharing mechanism. This effectively captures stable normal representations and their discrepancies with pseudo-anomalies.

*(Please upload your architecture diagram to the repository and update the path below)*
<p align="center">
  <img src="assets/architecture.png" alt="PA3AD Architecture" width="800"/>
</p>

## 📂 Datasets

Our method achieves superior detection performance and consistently outperforms existing state-of-the-art approaches on the following standard industrial benchmarks:
- **Anomaly-ShapeNet**
- **Real3D-AD**

*(Detailed data preparation instructions will be updated with the code release)*

## 🛠️ Installation (Coming Soon)

```bash
git clone [https://github.com/NingxiaoJian/PA3AD.git](https://github.com/NingxiaoJian/PA3AD.git)
cd PA3AD
# Installation details (e.g., pip install -r requirements.txt) will be provided soon.
