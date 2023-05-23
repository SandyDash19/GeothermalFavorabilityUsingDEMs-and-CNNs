# MP2
Problem Description

The goal of this project is to develop a predictive model using Convolutional Neural Network (CNN) which
studies Favorable Structural Settings (FSS) in the images of Detrended Elevation Maps (DEM) and predicts
heatflow residuals. Accurate prediction of heatflow residuals also known as geothermal favorability, is of
interest to United States Geological Survey (USGS) so that earth’s heat can be used to produce electricity.
We are given a dataset of 222 DEM patches of size 40 km x 40 km which equals to 200 x 200 pixels, with
222 labels which are heatflow residuals to train the CNN. We are also provided with 56 DEM patches as
test dataset without labels. The training dataset is further split into training and validation sets using the
80:20% ratio.

Patches containing zero wells are discarded, and the training and test datasets are created in an 80/20 split.
For a residual ri , we will again utilize the categories
1. low: ri ≤ 25
2. transition: ri ∈ (25, 50]
3. high: ri ∈ (50, 200]
4. very high: ri > 200
The resulting labels 1-4 are ordinal labels. As with MP1, your final predictor will be evaluated using the
following loss
ℓ(y, ŷ) = |y − ŷ| ,
(1)
where y, ŷ ∈ {1, · · · , 4}.
Your task is to train an accurate CNN-based predictor of the heatflow residual given an elevation image
patch.

File dictionary
AlexNetBN.py            - AlexNet model
CnnClassification.py    - Classification approach for the above problem.
CnnOrdinalRegression.py - Regression approach for the above problem.
ReadImages.py           - Open the dataset and plot the traning and test images. 
Technical_Report.pdf    - Technical Report for this project. 
