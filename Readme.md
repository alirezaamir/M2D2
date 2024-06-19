# M2D2: Maximum-Mean-Discrepancy Decoder for Temporal Localization of Epileptic Brain Activities

## Overview
This repository houses the source code and documentation for the M2D2 approach, which utilizes deep learning for automatic offline detection and labeling of seizures in electroencephalographic (EEG) signals. This model aims to assist medical experts by improving generalization of EEG-based monitoring across different clinical settings.

**Paper Link**: [Read the Manuscript](https://ieeexplore.ieee.org/document/9899698)

**Research Institution**: [Embedded Systems Laboratory (ESL) at EPFL](https://www.epfl.ch/labs/esl/)

## Simplified Abstract
M2D2 leverages deep learning to enhance the monitoring and labeling of epilepsy patients via EEG signals. Traditional deep learning methods often suffer from poor generalization across different clinical environments and require extensive manual labeling. Our proposed Maximum-Mean-Discrepancy Decoder (M2D2) addresses these challenges, achieving robust offline detection and labeling of seizures, which significantly assists in clinical diagnostics and patient monitoring.

## Main Dataset
The datasets used in this work are [Epilepsiae](https://epilepsy-database.eu/) and the Children’s Hospital Boston, [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/).

## Contributions
- **Robust Seizure Detection**: M2D2 significantly outperforms other deep learning models in temporal localization, offline detection and labeling of seizures across different clinical settings.
- **Improved Generalization**: Demonstrates strong generalization capabilities with F1-scores of 76.0% and 70.4% in different clinical environments.
- **Assistance to Medical Experts**: Facilitates the diagnostic process and reduces the dependency on extensive manual labeling of EEG data.

## Usage 
The source code for applying m2d2 on the seizure detection task is in [this file](src/mmd_train.py). The corresponding model architecture is defined in [here](src/utils/vae_model.py)


## Reference
If you have used M2D2, we would appreciate it if you cite the following papers in your academic articles:

```
Amirshahi, A., Thomas, A., Aminifar, A., Rosing, T., & Atienza, D. (2022).
M2D2: Maximum-Mean-Discrepancy Decoder for Temporal Localization of Epileptic Brain Activities.
IEEE Journal of Biomedical and Health Informatics, 27(1), 202-214.
```
