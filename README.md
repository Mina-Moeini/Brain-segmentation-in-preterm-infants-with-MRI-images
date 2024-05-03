
# Improving Surgical Decision Support for Premature Newborns with U-Net in collaboration with Necker Hospital

Post-hemorrhagic hydrocephalus (PHH) is a medical complication prevalent in premature infants, characterized by abnormal cerebrospinal fluid (CSF) accumulation following intraventricular hemorrhage (IVH). This project addresses the critical need for improved diagnostic tools in managing PHH. Our aim is to create deep learning algorithms for precise segmentation of CSF in MRI images of preterm infants, enhancing diagnostic clarity for healthcare professionals.

The challenges in this work arise from the anatomical variability of ventricles in preterm infants with PHH, compounded by difficulties in MRI imaging of premature brains. Limited input data further complicates deep learning model training, a challenge mitigated through data augmentation techniques. Despite these hurdles, our segmentation model, trained on 10 T2-weighted MRIs with a 7:2:1 split for training, testing, and validation, demonstrated exceptional performance. At Epoch 25, the model achieved an accuracy of 99.14% on the test dataset, emphasizing its potential in contributing to enhanced diagnostic accuracy and treatment planning for PHH in premature infants.



## Dataset
This project's dataset, provided by Necker Hospital, comprises a challenging set of 10 T2 MRI images of premature infants, accompanied by manual masks. I do not have permission to share the dataset. Given the limited dataset size, we address the significant challenge of training a deep neural network effectively. For the first step we should see the MRI images in the [3D Slicer](https://www.slicer.org/), it looks like :

![Ex_MRI](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/1.jpg)



 To optimize the dataset for model training, a meticulous data preprocessing pipeline has been implemented, detailed as follows:



## Preprocessing

 - **Reading Nifti Data**: We can read Nifti Data with [```NiBabel```](https://nipy.org/nibabel/#nibabel)

 - **Image Cropping**: Due to the inherent challenges in photographing small organs of premature infants, we initiated the preprocessing by cropping the MRI images and their masks. This process focused on retaining only the sections containing the infants' heads, ensuring cleaner and more relevant data. The cropping operation was performed using the 3D Slicer program, and the resulting images were saved in the “.nii.gz” format to reduce file size for efficient computational processing.

 - **Resize images**: In the subsequent preprocessing step, we aimed to standardize the sizes of the cropped images and masks. Recognizing that the average dimensions of the data were approximately (256, 256, 256), we opted to resize the images and masks accordingly using the OpenCV library. It is noteworthy that for segmentation purposes, the interpolation method used during resizing was set to cv2.INTER_NEAREST to maintain binary segmentation values. This ensures that the segmented image retains its binary nature after resizing. you can see the below an example of mri images and related masks after resizing : 

![Ex_resize](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/2.png)

 - **Conversion to 2D:** In an effort to augment the dataset, we decided to transform the 3D images and corresponding masks into 2D representations. This conversion involved cutting the images and masks along the axial slice, as it was deemed to provide a clearer depiction of the cerebrospinal fluid (CSF) area. Following the conversion to 2D, we proceeded to normalize the images, ensuring consistency in pixel values for further analysis.

 - **Data augmentation:** In order to address the challenges associated with limited data and to enhance the adaptability of our model to variations commonly encountered in medical images, we employed data augmentation techniques. The  [```Monai```](https://monai.io/) library, specifically designed for medical images, played a crucial role in this process. The augmentation strategies implemented in this project aim to ensure the model's ability to handle variations such as noise and head rotation, while still providing reliable results.

 ![15](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/3.png)

 ![12](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/4.png)

## U-Net
The [```U-Net```](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture is a fundamental component of our project, chosen for its effectiveness in semantic segmentation tasks, particularly in medical image analysis. Named after its U-shaped structure, this convolutional neural network (CNN) architecture was introduced by Ronneberger et al. in 2015.
U-Net excels in semantic segmentation tasks, precisely delineating regions of interest within medical images. In our project, it focuses on segmenting the cerebrospinal fluid in MRI scans of premature infants with post-hemorrhagic hydrocephalus.

## U-Net Architecture Implementation
The method development for this project, focusing on the U-Net architecture, involved a systematic process utilizing TensorFlow and Keras. The key stages of the method development are elucidated below:

 - **Model Architecture Creation:** The U-Net architecture was constructed using the TensorFlow framework. This deep learning architecture is renowned for its effectiveness in image segmentation tasks. The model's structure was defined to match the input dimensions of the MRI images (256 x 256 x 1).
 - **Data Division:** The available dataset was divided into distinct sets, adhering to a meticulous strategy. The division comprised 7 photos for training, 1 photo for validation, and 2 photos for testing. The careful consideration of patient-specific data distribution aimed to prevent any overlap between patients in different sets, ensuring robust training and evaluation.
 - **Model Compilation:** The model was compiled using the Adam optimizer with a learning rate of 0.0001 and binary cross-entropy as the loss function. The choice of these parameters is crucial for the effective training of the model.
 - **Model Training with Checkpointing:** The training process commenced, involving 25 epochs, on the training set (X_train, y_train). The validation set (X_val, y_val) was employed to monitor the model's performance during training. The ModelCheckpoint callback was implemented to save the best-performing model based on validation loss, mitigating the risk of overfitting.
 - **Model Evaluation:** Throughout the training process, metrics such as training and validation loss were monitored to assess convergence and prevent overfitting. Visualizations of loss curves over epochs provide a succinct overview of the model's learning trajectory and convergence behavior. In the training phase, the U-Net model exhibits commendable progress as it iteratively refines its segmentation capabilities. The training process unfolds with a clear trajectory towards optimization, as reflected by the consistent decrease in error and concurrent increase in accuracy.
 ![evaluation](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/5.png)

 ## Result
A selection of MRI images from the test set, along with their corresponding manual masks and the model's segmentation results, is showcased. This visual representation allows for an intuitive understanding of the model's ability to capture the intricacies of CSF regions.
![1](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/6.png)
![2](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/7.png)
In the context of image segmentation and medical image analysis, metrics such as Dice coefficient, Jaccard index, sensitivity, specificity, precision, and recall are commonly used to evaluate the performance of segmentation algorithms.

![11](https://github.com/Mina-Moeini/Brain-segmentation-in-preterm-infants-with-MRI-images/blob/main/images/9.png)



## 🔗 Contact

|||
|-|-|
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:m.moeini67@gmail.com) |[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mina-moeini)

## Authors

 [@Mina-Moeini](https://github.com/Mina-Moeini)
