# Preparing Normals

We use [Omnidata](https://github.com/EPFL-VILAB/omnidata) as the foundation model to extract monocular normals. Feel free to tryother models, such as [Metric3D](https://github.com/YvanYin/Metric3D).

## Installation


```sh
git clone https://github.com/EPFL-VILAB/omnidata.git
cd omnidata/omnidata_tools/torch
mkdir -p pretrained_models && cd pretrained_models
wget 'https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt'
wget 'https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt'
```

## Processing

### Waymo dataset

If you download the dataset from [PVG](https://drive.google.com/file/d/1eTNJz7WeYrB3IctVlUmJIY0z8qhjR_qF/view?usp=sharing) to `dataset`, run the following command to generate the normal maps

```sh
python scripts/extract_mono_cues_waymo.py --data_root ./dataset/waymo_scenes --task normal
```

### KITTI dataset

```sh
python scripts/extract_mono_cues_kitti.py --data_root ./dataset/kitti_mot/training --task normal
```

