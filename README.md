# NeRF for 3D Reconstruction from X-ray Angiography

Code for the *NeRF for 3D Reconstruction from X-ray Angiography*, which is described in our paper "[NeRF for 3D Reconstruction from X-ray Angiography](https://diglib.eg.org/handle/10.2312/vcbm20231210)".

Online demo: https://nerfforangiography.netlify.app/ (due to utilizing free services, there is a delay when hovering over the heatmap).

Warning: This code may contain bugs and is not fully optimized.

## Overview
The visualization tool code can be found in /cag-vis folder. This can be ran with npm. The code to obtain the .csv/images to load in the tool can be found in the /analysis folder.

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{maas2023nerf,
  title={NeRF for 3D Reconstruction from X-ray Angiography: Possibilities and Limitations},
  author={Maas, Kirsten WH and Pezzotti, Nicola and Vermeer, Amy JE and Ruijters, Danny and Vilanova, Anna},
  booktitle={VCBM 2023: Eurographics Workshop on Visual Computing for Biology and Medicine},
  pages={29--40},
  year={2023},
  organization={Eurographics Association}
}
```