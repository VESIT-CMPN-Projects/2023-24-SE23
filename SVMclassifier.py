import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops

#print(os.listdir("D:/ProjectDataset/OwnDataset/Mango"))

# Resizing images to 128*128
Size = 128

# Capture Images and labels into array
# Creating a empty array
TrainImages = []
TrainLabels = []

for directory_path in glob.glob("D:/ProjectDataset/OwnDataset/Mango/Train/*"):
    label = directory_path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.JPG")):
     #   print(img_path)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (Size, Size))
        TrainImages.append(img)
        TrainLabels.append(label)
        
TrainImages = np.array(TrainImages)
TrainLabels = np.array(TrainLabels)

TestImages = []
TestLabels = []

for directory_path in glob.glob("D:/ProjectDataset/OwnDataset/Mango/Test/*"):
    Sec_label = directory_path.split("\\")[-1]
    # print(Sec_label)
    for img_path in glob.glob(os.path.join(directory_path, "*.JPG")):
        # print(img_path)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (Size, Size))
        TestImages.append(img)
        TestLabels.append(Sec_label)
        
TestImages = np.array(TestImages)
TestLabels = np.array(TestLabels)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(TestLabels)
TestLabelsEncoded = le.transform(TestLabels)
le.fit(TrainLabels)
TrainLabelsEncoded = le.transform(TrainLabels)

# x train is for training images 
# Original assignment
x_train, y_train, x_test, y_test = TrainImages, TrainLabelsEncoded, TestImages, TestLabelsEncoded

# Feature Extraction Function
def FeatureExtractor(dataset):
    ImageDataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
    # print(image)
    
        df = pd.DataFrame()
    
        img = dataset[image, :, :]
    
    
        GLCM = graycomatrix(img, [1], [0])
        GLCM_Energy = graycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_Corr = graycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_Corr
        GLCM_Diss = graycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_Diss
        GLCM_Hom = graycoprops(GLCM, 'homogeneity')[0]
        df['Hommogen'] = GLCM_Hom
        GLCM_Contr = graycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_Contr
    
        GLCM2 = graycomatrix(img, [3], [0])
        GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_Corr2 = graycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_Corr2
        GLCM_Diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_Diss2
        GLCM_Hom2 = graycoprops(GLCM2, 'homogeneity')[0]
        df['Hommogen2'] = GLCM_Hom2
        GLCM_Contr2 = graycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_Contr2
    
        GLCM3 = graycomatrix(img, [5], [0])
        GLCM_Energy3 = graycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_Corr3 = graycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_Corr3
        GLCM_Diss3 = graycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_Diss3
        GLCM_Hom3 = graycoprops(GLCM3, 'homogeneity')[0]
        df['Hommogen3'] = GLCM_Hom3
        GLCM_Contr3 = graycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_Contr3
    
        GLCM4 = graycomatrix(img, [0], [np.pi/4])
        GLCM_Energy4 = graycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_Corr4 = graycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_Corr4
        GLCM_Diss4 = graycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_Diss4
        GLCM_Hom4 = graycoprops(GLCM4, 'homogeneity')[0]
        df['Hommogen4'] = GLCM_Hom4
        GLCM_Contr4 = graycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_Contr4
        
        GLCM5 = graycomatrix(img, [1], [np.pi/2])
        GLCM_Energy5 = graycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_Corr5 = graycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_Corr5
        GLCM_Diss5 = graycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_Diss5
        GLCM_Hom5 = graycoprops(GLCM5, 'homogeneity')[0]
        df['Hommogen5'] = GLCM_Hom5
        GLCM_Contr5 = graycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_Contr5
        
        ImageDataset = ImageDataset.append(df)
    
    return ImageDataset
# Extracting features from Training Images
# Import the necessary library for SVM classifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(TrainImages, TrainLabelsEncoded, test_size=0.2, random_state=42)

# Flatten the images from 128x128 to 16384
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Create a SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier using the training data
svm_classifier.fit(x_train_flat, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(x_test_flat)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix


# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
