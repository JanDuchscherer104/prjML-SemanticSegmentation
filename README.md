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

### alterantively
- Fine-tune a  model and implement in Rust for usage on an LASIM device / vechicle

## Papers
- [3D Semantic Segmentation of Point Clouds](https://www.vision.rwth-aachen.de/media/papers/know-what-your-neighbors-do-3d-semantic-segmentation-of-point-clouds/W63P26.pdf)
- [analytics-vidhya/3d-point-cloud-semantic-segmentation](https://medium.com/analytics-vidhya/3d-point-cloud-semantic-segmentation-using-deep-learning-techniques-6c4504a97ce6)

## Networs
### PointNet++
#### Advantages of PointNet++:
-  Is able to learn local features with increasing contextual scales, enabling it to recognize fine-grained patterns and generalize to complex scenes.
- The hierarchical structure allows it to extract local features capturing fine geometric structures from small neighborhoods and progressively abstract larger local regions along the hierarchy.
-  Uses a mini-PointNet to encode local region patterns into feature vectors, which is a powerful feature extractor and can capture complex patterns.
- Can handle unordered point clouds (without costly transformation) as inputs, making it applicable to a wide range of tasks, including 3D object recognition, scene understanding, and point cloud segmentation.
#### Disadvantages of PointNet++:
- Requires a large amount of training data to learn robust and accurate features, which can be a challenge in some applications.
- It's hierarchical structure  can be computationally expensive, especially when dealing with large point clouds.
- PointNet++ does not explicitly model the spatial relationships between points, which can limit its ability to recognize patterns in highly cluttered scenes.
- PointNet++ may struggle to handle point clouds with varying densities, as the sampling and grouping layers may not be able to capture the appropriate local structures in the data.

### RandLA-Net
#### Advantages of RandLA-Net:
- RandLA-Net is upto 200 times faster than other point cloud processing architectures, due to the lightweight architecture and the absence of expensive pre/post-processing steps.
- RandLA-Net can process large-scale 3D point clouds in a single pass, without requiring any pre/post-processing steps such as voxelization, block partitioning, or graph construction.
- The local feature aggregator of RandLA-Net obtains successively larger receptive fields by considering local spatial relations and point features, which enables it to capture the context information of each point.
- RandLA-Net proposes a local aggregation module that avoids dropping significant features and encodes local geometric structures effectively.
#### Disadvantages of RandLA-Net:
- The random sampling approach used in RandLA-Net can result in the loss of significant features and may not work well for point clouds with complex structures.
- Since RandLA-Net continuously downsamples the input point cloud, it may not be able to capture fine-grained details in the data.
- The absence of graph construction and kernelization may limit the ability of RandLA-Net to model complex relationships between points in the data.
- RandLA-Net may require a large amount of training data to learn robust and accurate features, which can be a challenge in some applications.

### PointCNN
#### Advantages of PointCNN:
- PointCNN is able to capture spatially-local correlation in point clouds, which is useful for tasks such as segmentation and classification.
- The architecture of PointCNN includes a hierarchical convolution design which helps to reduce the number of points while retaining important features.
- PointCNN uses the χ-Conv operator which is able to convolve with neighborhood points and transform them into local coordinate systems, allowing for more efficient feature learning.
- Denser connections are used in the χ-Conv layers to prevent a rapid drop in training samples and maintain depth.
- Dilated convolutions are employed to maintain receptive field growth rate while keeping network depth.
#### Disadvantages of PointCNN:
- PointCNN may not be as effective at capturing global features as other architectures.
- The denser connection used in the χ-Conv layers may result in higher computational costs and memory usage.
- The training of PointCNN may be less efficient due to the rapid drop in training samples in the top layers, requiring additional optimization techniques to be employed.

#### RSNet
#### Advantages:
- RSNet is capable of modeling required dependencies between point clouds, which most other semantic segmentation networks do not do.
- The local dependency module is highly efficient, with a time complexity of O(n) w.r.t the number of input points and O(1) w.r.t the local context resolutions.
- RSNet takes raw point clouds as inputs and outputs semantic labels for each of them.
- RSNet is shown to surpass previous state-of-the-art methods on three widely used benchmarks while requiring less inference time and memory.

### Dynamic Graph CNN
#### Advantages:
- DGCNN captures local geometric structures and maintains permutation invariance, making it suitable for high-level tasks on point clouds like classification and segmentation.
- EdgeConv generates edge features that describe relationships between points and their neighbors, making the model capable of grouping points both in Euclidean space and semantic space.
- DGCNN dynamically updates the set of k-nearest neighbors of a point after each layer of the network, making the model more flexible and adaptable.
- The DGCNN architecture can be easily incorporated into existing pipelines for point cloud-based graphics, learning, and vision.
#### Disadvantages:
- The point cloud transform block in DGCNN requires an estimated 3 × 3 matrices, which may not always accurately align input point sets to a canonical space.
- The EdgeConv block uses a multi-layer perceptron (MLP) with the number of layer neurons to generate edge features, which may not be as efficient as other approaches for generating edge features.
- DGCNN's graphs are not fixed, which may make it difficult to compare results across different runs of the model.


## Benchmarks / Datasets
- [S3DIS](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=window-normalization-enhancing-point-cloud)
- [semantic3d](https://paperswithcode.com/dataset/semantic3d)


## Hardware
- **Camera type**: [intel realsense depth camera d455](https://www.intelrealsense.com/wp-content/uploads/2020/06/Intel-RealSense-D400-Series-Datasheet-June-2020.pdf)


## Notes from 'lecture'
