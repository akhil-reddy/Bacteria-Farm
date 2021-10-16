# Bacteria-Farm

## Architecture

![Flowchart](https://user-images.githubusercontent.com/17645442/137587791-56189d8d-3302-40ce-be38-720647983116.png)

## Description

An intuitive density-based clustering algorithm, which is split into two phases:

### Measure of Center Extraction

The model uses a standard clustering algorithm to extract both the measure of center ( typically centroids ) and the proportion of test data in each of the resultant clusters. Moreover, the clustering at this phase can be halted prematurely once these parameters start to converge towards their final values. We have this flexibility as the standard algorithm is essential just for the presentation of those prerequisited parameters to the core model. 

### Core Model

The core model identifies a set of front runners - data points which sufficiently represent each cluster - and uses them to expand their clusters until the terminating conditions are met. This method is detailed further under the `Documents` directory.

## Target Applications

This model is particularly suited for applications with intent to improve the accuracy ( of clustering ) while accepting a trade-off on the runtime of the model ( as it inherently involves prior clustering ). Additionally, the model provides flexibility with the initial algorithm used ( for  dataset-level customizations ) and also features noise specification, which are valuable add-ons for such applications.

## Reads

Please scan through the `Documents` directory for an academic reading.

## Code

1. The final version of the model is 'BFFR Sequential and Optimized.py' under the `Code` directory.
2. It evolved as follows ( all scripts are under the same `Code` directory )
  * BF.py
  * BF Improved.py
  * BFFR.py
  * BFFR Sequential.py
  * BFFR Sequential and Optimized.py

## Usage

```
python3 "BFFR Sequential and Optimized.py"
```
