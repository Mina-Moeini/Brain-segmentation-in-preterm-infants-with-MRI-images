
# Improving Surgical Decision Support for Premature Newborns with U-Net in collaboration with Necker Hospital

Post-hemorrhagic hydrocephalus (PHH) is a medical complication prevalent in premature infants, characterized by abnormal cerebrospinal fluid (CSF) accumulation following intraventricular hemorrhage (IVH). This project addresses the critical need for improved diagnostic tools in managing PHH. Our aim is to create deep learning algorithms for precise segmentation of CSF in MRI images of preterm infants, enhancing diagnostic clarity for healthcare professionals.

The challenges in this work arise from the anatomical variability of ventricles in preterm infants with PHH, compounded by difficulties in MRI imaging of premature brains. Limited input data further complicates deep learning model training, a challenge mitigated through data augmentation techniques. Despite these hurdles, our segmentation model, trained on 10 T2-weighted MRIs with a 7:2:1 split for training, testing, and validation, demonstrated exceptional performance. At Epoch 25, the model achieved an accuracy of 99.14% on the test dataset, emphasizing its potential in contributing to enhanced diagnostic accuracy and treatment planning for PHH in premature infants.



## Dataset
This project's dataset, provided by Necker Hospital, comprises a challenging set of 10 T2 MRI images of premature infants, accompanied by manual masks. I do not have permission to share the dataset. Given the limited dataset size, we address the significant challenge of training a deep neural network effectively. For the first step we should see the MRI images in the [3D Slicer](https://www.slicer.org/), it looks like :

![Ex_MRI]([https://private-user-images.githubusercontent.com/80553485/318394843-87490291-10e1-495d-9dc5-d5ba6e07551f.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzI3MzIsIm5iZiI6MTcxMTk3MjQzMiwicGF0aCI6Ii84MDU1MzQ4NS8zMTgzOTQ4NDMtODc0OTAyOTEtMTBlMS00OTVkLTlkYzUtZDViYTZlMDc1NTFmLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNTM1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcwNzQ2M2Y0OGE2NGE5MDYwMzI4NjFmNjhjYzBhY2FmMzNhM2VhZmQ5NmQ1NjI0YzVmYzE5YmI1MmQ0ZWZhOWImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Gx4aj47yGr8RsdyaOgK764MEcQ8SJ2vm0_DRc1hMIkw](https://private-user-images.githubusercontent.com/80553485/318394843-87490291-10e1-495d-9dc5-d5ba6e07551f.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQ3Mjk1NjYsIm5iZiI6MTcxNDcyOTI2NiwicGF0aCI6Ii84MDU1MzQ4NS8zMTgzOTQ4NDMtODc0OTAyOTEtMTBlMS00OTVkLTlkYzUtZDViYTZlMDc1NTFmLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MDMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTAzVDA5NDEwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM5YWQ1YTc1OTQxOTFhNTIyOTRiZmFhYTE5N2QxNjVmMmQxYjI5OWY1ZGIxZmQ1NWM1NjA4MDg5ZDRmZDkyOTYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.9QmojJI90e96y-G_AMehRyQxJIYqE6gqfqCBU2NFzfg))



 To optimize the dataset for model training, a meticulous data preprocessing pipeline has been implemented, detailed as follows:



## Preprocessing

 - **Reading Nifti Data**: We can read Nifti Data with [```NiBabel```](https://nipy.org/nibabel/#nibabel)

 - **Image Cropping**: Due to the inherent challenges in photographing small organs of premature infants, we initiated the preprocessing by cropping the MRI images and their masks. This process focused on retaining only the sections containing the infants' heads, ensuring cleaner and more relevant data. The cropping operation was performed using the 3D Slicer program, and the resulting images were saved in the “.nii.gz” format to reduce file size for efficient computational processing.

 - **Resize images**: In the subsequent preprocessing step, we aimed to standardize the sizes of the cropped images and masks. Recognizing that the average dimensions of the data were approximately (256, 256, 256), we opted to resize the images and masks accordingly using the OpenCV library. It is noteworthy that for segmentation purposes, the interpolation method used during resizing was set to cv2.INTER_NEAREST to maintain binary segmentation values. This ensures that the segmented image retains its binary nature after resizing. you can see the below an example of mri images and related masks after resizing : 

![Ex_resize](https://private-user-images.githubusercontent.com/80553485/318398144-380ad7b3-b0a7-472b-a431-76b78961771e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzI3MzIsIm5iZiI6MTcxMTk3MjQzMiwicGF0aCI6Ii84MDU1MzQ4NS8zMTgzOTgxNDQtMzgwYWQ3YjMtYjBhNy00NzJiLWE0MzEtNzZiNzg5NjE3NzFlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNTM1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMxM2MxNDE2NjNiOGM2N2M1YmZiYTE3ZmZhMGJiNWJiOGFjZjYyOTZmMDIyYzhlOTk2MzMxNDAwNTA4ZTcwNzAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.K8P0dGDWrGTXQeFXa8WFemmSPu2frvbdCHpNTI0D7-g)

 - **Conversion to 2D:** In an effort to augment the dataset, we decided to transform the 3D images and corresponding masks into 2D representations. This conversion involved cutting the images and masks along the axial slice, as it was deemed to provide a clearer depiction of the cerebrospinal fluid (CSF) area. Following the conversion to 2D, we proceeded to normalize the images, ensuring consistency in pixel values for further analysis.

 - **Data augmentation:** In order to address the challenges associated with limited data and to enhance the adaptability of our model to variations commonly encountered in medical images, we employed data augmentation techniques. The  [```Monai```](https://monai.io/) library, specifically designed for medical images, played a crucial role in this process. The augmentation strategies implemented in this project aim to ensure the model's ability to handle variations such as noise and head rotation, while still providing reliable results.

 ![15](https://private-user-images.githubusercontent.com/80553485/318399998-44eadbc9-c477-466f-a962-ec74eebc69b4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzI5MDUsIm5iZiI6MTcxMTk3MjYwNSwicGF0aCI6Ii84MDU1MzQ4NS8zMTgzOTk5OTgtNDRlYWRiYzktYzQ3Ny00NjZmLWE5NjItZWM3NGVlYmM2OWI0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNTY0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRlN2U1Njk4MWVmMDM5NWJkYThlMzFiZTkyNTUwNzgyNmRmOWExZmEzZjY4MzE5OWNmMjc0MWVlMzg2MTUzOGQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.huAmJrFXgSM4_Mdn-He54xbjR-6JR9KCjuEm0uVFyic)

 ![12](https://private-user-images.githubusercontent.com/80553485/318400311-7455a1eb-ec5f-466a-802c-3e3f4c77ef1d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzI5MDUsIm5iZiI6MTcxMTk3MjYwNSwicGF0aCI6Ii84MDU1MzQ4NS8zMTg0MDAzMTEtNzQ1NWExZWItZWM1Zi00NjZhLTgwMmMtM2UzZjRjNzdlZjFkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNTY0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJlYWM2ODQwZTFjYzQwMjkwNTkyOTA1ZWE2ZDg4NzAzMjg1MGY5NWI5YzRjMTg0MmU4NWZkM2ZmZDhmMTg2ZWYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.l01F8bwDftFig-iIOyTQr6ND2XVCF3AHkQen2J2V8rA)

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
 ![evaluation](https://private-user-images.githubusercontent.com/80553485/318421761-0e160b3f-a19a-4c74-93d3-0cc8e7697afb.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzIwMjYsIm5iZiI6MTcxMTk3MTcyNiwicGF0aCI6Ii84MDU1MzQ4NS8zMTg0MjE3NjEtMGUxNjBiM2YtYTE5YS00Yzc0LTkzZDMtMGNjOGU3Njk3YWZiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNDIwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTk0MWE2MDFhYjdkOGI0NWMwNzc4YjNmMTdkMDIwYmUzODkyNjRjMjRiZGZlOTg3NDg0NmZiZmFjYjVlOTQxMmImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.l0aYoEoHXPMucMnT-shFgPZCT2i8r9GcpgP0oSliG-w)

 ## Result
A selection of MRI images from the test set, along with their corresponding manual masks and the model's segmentation results, is showcased. This visual representation allows for an intuitive understanding of the model's ability to capture the intricacies of CSF regions.
![1](https://private-user-images.githubusercontent.com/80553485/318422252-7d8bbe05-31e8-4502-b6ab-ceb1dbdac665.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzIyMDQsIm5iZiI6MTcxMTk3MTkwNCwicGF0aCI6Ii84MDU1MzQ4NS8zMTg0MjIyNTItN2Q4YmJlMDUtMzFlOC00NTAyLWI2YWItY2ViMWRiZGFjNjY1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNDUwNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdmNWNiMTg1NTdhNjkwNmEwYmFiOWI0ZTcyMjUzNTBkZDViNjBhMjIwNzZmMmM5YTc0OGFlNjdmODcwZjk5MzAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.xmJC_mXvNiF0BK2yxxx0S_cgRlxfRSgmq-5AAdHZIEM)
![2](https://private-user-images.githubusercontent.com/80553485/318422298-6a127dbc-024e-4759-935c-fd7ae2815059.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzIyMTgsIm5iZiI6MTcxMTk3MTkxOCwicGF0aCI6Ii84MDU1MzQ4NS8zMTg0MjIyOTgtNmExMjdkYmMtMDI0ZS00NzU5LTkzNWMtZmQ3YWUyODE1MDU5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNDUxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYwOWUxN2M4NWI0NTAxZGEyNmM0NjlmNDgwMGY4YjRmOTBiZWQxMDYxMDI5OTJlMjU2NTk2ZmQ2YmM5MDQwMzImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.nX0A67oyNcMNNzlIg8dxv1nyXwIia7lf7q_rY1ByPQo)
![3](https://private-user-images.githubusercontent.com/80553485/318422323-b811dd5e-7fa2-4fe3-bc79-b16efab319a8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzIyMjcsIm5iZiI6MTcxMTk3MTkyNywicGF0aCI6Ii84MDU1MzQ4NS8zMTg0MjIzMjMtYjgxMWRkNWUtN2ZhMi00ZmUzLWJjNzktYjE2ZWZhYjMxOWE4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNDUyN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc5NGI2NTU0MDVlNGYxYzE0YjM5NmI0NzI0M2QzM2NhZjNhNDY5YzBkNTk3OTM0YzMzZmQ3ZGEwYTZkZDVhMjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.V70nkqOiYaFbc4Bj17yg4LQSj6V9bBJbjtSKRsfO_YQ)
In the context of image segmentation and medical image analysis, metrics such as Dice coefficient, Jaccard index, sensitivity, specificity, precision, and recall are commonly used to evaluate the performance of segmentation algorithms.

![11](https://private-user-images.githubusercontent.com/80553485/318423253-6fb2294f-c064-43ea-bae4-0244e61184f2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTE5NzI1NTcsIm5iZiI6MTcxMTk3MjI1NywicGF0aCI6Ii84MDU1MzQ4NS8zMTg0MjMyNTMtNmZiMjI5NGYtYzA2NC00M2VhLWJhZTQtMDI0NGU2MTE4NGYyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA0MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNDAxVDExNTA1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ3ZWFmNWVlNGFjMTZlMjMyZmE4MWViMjVmY2I4NWU4NTZmNjViNWQ5ZjdlNmNlMDdhMGZhNTU2ZTMzOWE1MzEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.fnfP17A2CfdPncJnPBw3rIm9XAASa6J0vjBDVr3A43M)



## 🔗 Contact

|||
|-|-|
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:m.moeini67@gmail.com) |[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mina-moeini)

## Authors

 [@Mina-Moeini](https://github.com/Mina-Moeini)
