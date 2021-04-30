# MTP

The weight_sharing.ipynb file contains the code to reproduce the results from the paper "[Weight Sharing is Crucial to Succesful Optimization - Shai Shalev-Shwartz](https://arxiv.org/pdf/1706.00687.pdf)". Run the PLOTS_WS.py after running the ipynb file to get the final plots.

## Multimodel Initializations

We try out different initializations for a multimodel dataset and record their performances.

### Multi-spheres dataset

We first uniformly choose points on a unit n dimensional cube and project them onto a sphere/shell embedded in the cube. Now we randomly pick 500 points. We repeat it for each shell we will need.

The dataset consists of multiple sets of concentric shells with inner shell as class zero and outer shell as class 1. These shells are saperated by a large distance compared to their radii.

![2-D shells dataset](https://raw.githubusercontent.com/sarath1221/MTP/main/Plots/shells_dataset.png)

Similar datasets are created for 3,4,10 dimensions and neural network models are trained with different initialization strategies.

### Initialization Strategies 

1- Xavier init - As proposed by Xavier Glorot and Yoshua Bengio in "Understanding the difficulty of training deep feedforward neural networks".

2- Perpendicular init - Here 2 points are randomly choosen for each node and the node is initialized with the hyperplane perpendicular to these points.

3- Parallel init - Here n points are choosen randomly for each node and the node is initialized with the hyperplane passing through these points.


The following are some results -

This is the result for 2 dimensional multimodel dataset
![](https://raw.githubusercontent.com/sarath1221/MTP/main/Plots/2_DIMEN.png)

This is the result for 2 dimensional unimodel dataset ( where only one set of concentric shells is used to train the model )
![](https://raw.githubusercontent.com/sarath1221/MTP/main/Plots/uni_2_dim.png)
