Project: Word level Sign Language Detection using transformer models and comparision with traditional models

I already have video dataset for 2000 Words/classes you need to train for 10-100 classes.


Also please include validation loss and accuracy graphs also.

how many classes you will train initially and make sure model will be able to train after on many classes.
 
Please also include testing of model also.

Give top 10 predictions sorted, then calculate top 3%, 5%, 7%, and 10% accuracy.


1. Data Analysis & Preprocessing
1.1 Dataset Summary
•Load and analyse the dataset from Kaggle.
•Extract and report:
oNumber of unique word classes.
oTotal number of videos.
oMean number of videos per class.
oNumber of different persons signing the same word.
1.2 Data Preprocessing
•Convert videos into frames.
•Normalize the frames (resizing, colour normalization).
•Perform data augmentation (if necessary).
•Extract key points using MediaPipe or OpenPose (optional).
•Split the dataset into train and test, ensuring all classes are present in both.
2. Model Selection & Training
2.1 Selecting Models
•Transformer-based model: Experiment with Timesformer, ViViT, or Video Swin
Transformer.
•Traditional models for comparison:
oCNN + LSTM (for sequential image data).
o3D CNN (for spatial-temporal analysis).
2.2 K-Fold Cross Validation (7 Folds)
•Implement 7-fold cross-validation.
•Train models on different folds and compute validation loss & accuracy.
2.3 Training Procedure
•Use Adam optimizer with an appropriate learning rate.
•Choose Categorical Cross-Entropy Loss (since it's a multi-class classification
problem).
•Track loss and accuracy per epoch.
•Save the best model based on validation accuracy.3. Model Evaluation & Results
3.1-Fold-Wise Validation Graphs
•
Plot Validation Loss & Accuracy vs. Epochs for each fold.
3.2 Class-Wise Prediction & Accuracy Metrics
•Compute Top 10 predicted classes for each test sample.
•Calculate Top 3%, 5%, 7%, and 10% accuracy.
3.3 Bounding Box Visualization
•Use YOLO or OpenCV to detect hands and draw bounding boxes around them.
•Show correct and incorrect predictions with bounding boxes.
3.4 Class Distribution Analysis
•Ensure that all classes are present in both training and testing sets.
•Plot class distributions to check for imbalances.
5. Final Deliverables
•Validation loss & accuracy graphs for each fold.
•Bounding box visualizations of sign detections.
•Sorted Top 10 predictions per sample.
•Accuracy metrics at different thresholds (Top 3%, 5%, 7%, and 10%).
•Train & test class distributions.