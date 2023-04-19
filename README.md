## Milestones
- **Semantic Segmentation** or Object Detection
- Research on state-of-the-art point cloud models
- Find fitting data sets
- Fine-tuning of a model trained on point clouds
- hyper-parameter optimization using [wandb](https://wandb.ai/site)
- (optional) rebuild model from scratch
- Adapt model to target hardware / local environment -> data acquisition
- Generate segmentation / bounding boxes for each point cloud frame
- Integrate the model into an autonomous system (optional)
- Video stream (optional)

## Networks / Codebases
- [Pointcept](https://github.com/Pointcept/Pointcept)

## Benchmarks / Datasets
- [rgb-d](https://paperswithcode.com/datasets?mod=rgb-d)
- [ScanNet](https://paperswithcode.com/dataset/scannet)
- [NYUv2](https://paperswithcode.com/dataset/nyuv2)
- [ShapeNet](https://shapenet.org)
- [SUN RGB-D](https://rgbd.cs.princeton.edu)

## Papers
- [SparseUNet+MSC](https://arxiv.org/pdf/2303.14191v1.pdf)
- [SpConvNW](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf)
- ([4D Spatio-Temporal ConvNets](https://arxiv.org/abs/1904.08755))

## Hardware
- **Camera type**: [intel realsense depth camera d455](https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf)


## Notes

- __IoU__ := Intersection over Union, evaluates the overlap of the Ground Truth and Prediction region.
