# GUAVA Data Preprocessing Pipeline

This repository contains scripts to preprocess data for GUAVA. The pipeline includes tasks like video trimming, camera calibration, background matting, and body segmentation.

## Installation

To get started, create a Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate guava
```

Ensure all necessary dependencies are installed by checking the `environment.yml` file for any additional package requirements.

## Tasks Overview

### 1. Trim Videos

The `trim` option processes and trims synchronized video streams based on the start and end sequences for each video.
---

### 2. Camera Calibration
The `calibrate` option performs intrinsic and extrinsic camera calibration using checkerboard patterns.

- **Additional Documentation**: See the [Camera Calibration](EasyMocap/README.md).

---

### 3. Background Matting
The `background_matting` option applies background matting to the video streams. In this repository we are using "BiReFNet" model

- **Additional Documentation**: See the [Background Matting](BiRefNet/README.md).

---

### 4. Body Segmentation (Sapiens)
The `sapiens` option runs body segmentation using the Sapiens model.

- **Additional Documentation**: See the [Body Segmentation](sapiens/lite/docs/SEG_README.md).


### Example Usage
```command
python main.py --root_dir /home/vippin/thesis/extra --output /home/vippin/thesis/extra/demo_guava --sapiens --calibrate --background_matting
```
## Logging

Logs for each run are saved in `output.log`. The logging includes detailed information about each step and errors, if any.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📚 Citation
Please consider citing these works if you find this repo is useful for your projects.

- Sapiens
```bibtex
@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```
- Background Matting (BiRefNet)
```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume = {3},
  pages = {9150038},
  year={2024}
}
```
- Camera Calibration
```bibtex
@Misc{easymocap,  
    title = {EasyMoCap - Make human motion capture easier.},
    howpublished = {Github},  
    year = {2021},
    url = {https://github.com/zju3dv/EasyMocap}
}
