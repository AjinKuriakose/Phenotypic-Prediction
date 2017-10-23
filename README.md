# Phenotypic-Prediction
Phenotypic Prediction from Transcriptomic Features 

You are provided with output from Salmon, an RNA-seq mapping and quantification tool, on a number of datasets. The various samples come from different phenotypes; types of cancers in this case. Hence, each dataset is given a label based on the originating cancer type. Your task in this project is to build a model that takes the Salmon output and predicts the original label. You can train your model using all or some subset of the input data but make sure to not overfit the model since we will be testing it on a different dataset. You can begin by selecting basic features from the Salmon output, such as TPM, transcript lengths and counts. These are provided in the “quant.sf” file. However, we expect you will come up with a more complex feature selection method that will likely provide better results. We will provide a baseline that you must outperform. The goal is to achieve the maximum accuracy in classifying the samples using multi-class classification models. Ideally, you should test multiple feature selection methods and models. Your report should include all the results. 

The special properties of this dataset make it difficult to immediately obtain good prediction accuracies by building a well-known classification model like SVM, Random Forest or NaiveBayes on it. Each phenotypic category has data points assigned to it (around 20 or fewer samples) while each data point has a huge number of raw features (At least equal to total number of known transcripts for human being in this case which is ~100k, but we have additional features as well). So the main challenge in this problem is selecting and generating few number of descriptive and discriminative features. For that you can first use some visualization- and statistical-based feature analysis (e.g. use a Decision Tree to find the most discriminative features that come at the top levels of the tree, compute the correlation matrix to find and remove highly correlated features) use different feature selection methodologies (e.g. those based on entropy), use kernel models or feature segmentation methods to create advanced features, and dimensionality reduction models to change the feature domain and extract principal components . After that you can apply a classification model to the reduced datasets and run a grid search to find the best fit parameters for your model. For evaluation of different models you can use ROC plot and F-score and Accuracy metrics.
Input: output of Salmon 
Transcripts tpm, count, length
Equivalence Classes for each data set
Bias Models
Output: A phenotypic Prediction Model for Cancer Detection
Midway Result:
    1) feature selection + 1 dimensionality reduction algorithm + 1 Multi-Classification Model 
   that beats the baseline.
Keywords: Transcript Quantification, TPM, Prediction, Salmon
