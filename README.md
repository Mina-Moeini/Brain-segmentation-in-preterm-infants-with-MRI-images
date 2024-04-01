
# Improving Surgical Decision Support for Premature Newborns with U-Net in collaboration with Necker Hospital

Post-hemorrhagic hydrocephalus (PHH) is a medical complication prevalent in premature infants, characterized by abnormal cerebrospinal fluid (CSF) accumulation following intraventricular hemorrhage (IVH). This project addresses the critical need for improved diagnostic tools in managing PHH. Our aim is to create deep learning algorithms for precise segmentation of CSF in MRI images of preterm infants, enhancing diagnostic clarity for healthcare professionals.

The challenges in this work arise from the anatomical variability of ventricles in preterm infants with PHH, compounded by difficulties in MRI imaging of premature brains. Limited input data further complicates deep learning model training, a challenge mitigated through data augmentation techniques. Despite these hurdles, our segmentation model, trained on 10 T2-weighted MRIs with a 7:2:1 split for training, testing, and validation, demonstrated exceptional performance. At Epoch 25, the model achieved an accuracy of 99.14% on the test dataset, emphasizing its potential in contributing to enhanced diagnostic accuracy and treatment planning for PHH in premature infants.



## Dataset
This project's dataset, provided by Necker Hospital, comprises a challenging set of 10 T2 MRI images of premature infants, accompanied by manual masks. I do not have permission to share the dataset. Given the limited dataset size, we address the significant challenge of training a deep neural network effectively. For the first step we should see the MRI images in the [3D Slicer](https://www.slicer.org/), it looks like :

![Example MRI ](file:///C:/Users/Asus/Desktop/IMA/PRAT/Final_Report/20.jpg)
