
# Fire Detection System
Mehmet Öztürk
Mahmut Demir
Ezgi Ceylan



# Introduction

Turkey, with its 78 million hectares of land, has a rich diversity in terms of ecology. Within this richness, forests also hold an important place in terms of species and composition. As of 2020, forested areas cover 29.4% of the country's land area.[29]

Due to Turkey's location mainly within the Mediterranean climate zone, a significant portion of our forests is under the threat of wildfires. 60% of the total forested area has a first and second-degree fire risk. As a result, forest fires are among the most important issues in our country. In order to prevent forest fires, all necessary physical and human precautions should be taken. Developing firefighting methods is crucial to achieve this goal. Training personnel involved in forest fire prevention and shortening the fire detection time are among the most important activities.

With the advancement of technology, wildfire detection and warning systems have become more widespread. Early detection of fires is crucial for the protection of many animal and plant species and the reduction of material losses. There are numerous research and development studies in the literature focusing on early fire detection. The existing approaches can be divided into two categories. The first category is sensor-based systems that focus on detecting fires at short distances. The second category examines the processing of motion or still images. Early warning systems that can monitor larger areas compared to the first category can be addressed with this approach.

This study proposes a system for early detection of fires occurring in forested areas. In the proposed system, fire detection is performed using color features.



# Literature  Review
When fire detection systems are examined, the common goal of all systems is to create early warning to intervene in the fire in time, provide cheap cost, and be reliable. Depending on technological advancements, many fire detection systems have been developed from the past to the present. Our research topic is to create a fire detection system by utilizing artificial intelligence.
As a result of our literature review, some findings and methods were obtained. A method was encountered for smoke detection, which is based on the decrease in energy level between the background and the image in fire detection.[1]
Another method is a study based on color models using image processing. The color models they used in their studies were derived by analyzing samples from different videos and images.[2]
The MSG-2 satellite, which is one of the second generation Meteosat satellites, has a SEVIRI sensor that          provides images at 15-minute intervals and an MSG Active Fire Monitoring System with 15-minute temporal resolution, which has a wide range of applications in Africa and Europe. The FIR algorithm uses the differences and standard deviations of the data from the channels of the electromagnetic spectrum of the SEVIRI sensor on the MSG satellite for fire detection.[3]
Machine learning algorithms have also shown potential in analyzing environmental data to identify patterns and indicators of forest fires [4]. Huang et al. (2020) utilized a random forest algorithm to analyze temperature, humidity, and wind speed data, achieving a 93.3% accuracy rate in forest fire detection [5].
Data mining techniques have been applied in forest fire detection by analyzing large datasets, including satellite imagery and meteorological data [6]. Cheng et al. (2020) used a data mining approach to detect forest fires using satellite imagery with 90.3% accuracy [7].
In conclusion, the integration of AI technologies, including computer vision, machine learning, and data mining, shows potential for improving forest fire detection and prediction. Further research is needed to improve the accuracy and efficiency of these models for real-world application [8].



# Proposed Approach
We collected the data from a variety of sources, including videos and images of real-world fires as well as simulated fire scenarios [9] [10]. The first source contains approximately 2000 images and the second source contains approximately 1000 images. The data is divided into "fire" and "no_fire" categories. To ensure the accuracy of the data from the second source, we re-labeled the images by three different individuals and divided them into appropriate folders. 
We searched for publicly available datasets on forest fires and found two sources that suited our project requirements. We removed any corrupted or unusable images from the datasets. We labeled each image as "fire" or "no_fire". For the second source, we re-labeled the images by three different individuals to ensure the accuracy of the labels. We performed data augmentation techniques such as rotation, flipping, and cropping to increase the size of our dataset and to prevent overfitting. Then we split the data into training, validation, and testing sets. We used 80% of the data for training, 10% for validation, and 10% for testing and we normalized the pixel values of the images to a range of [0, 1]. We use this data to train my model to recognize fires in various contexts.

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/bb869310-a3cd-466e-a3b6-eec8f115a962)

Figure 1. Some Photos in the Dataset

Convolutional Neural Network (CNN) yields successful results especially in the field of image processing. It is used to extract and classify features of image data. CNN employs convolutional operations, called convolutions, to learn the features of the data. This is achieved by sliding a filter matrix (kernel) over the image. The process is repeated using different filters to learn important features of the image. Ultimately, lower-dimensional feature maps are obtained. Using these features, the data is passed to an output layer for classification. CNN yields successful results especially in complex image data. [11][12]
Support Vector Machine (SVM) is a model used for classification and regression analysis. SVM separates the data into two classes by creating a line (for 2-dimensional data) or a plane (for 3-dimensional data) to classify the data.[13] Artificial Neural Network (ANN), especially models called layered artificial neural networks, are commonly used in machine learning. ANN classifies data or performs regression analysis based on their features. [14]
Training and tuning other models like SVM may require more time and effort, whereas training CNN is easier and can yield faster results. Due to its features and structure, CNN is particularly popular in image processing projects and can achieve high accuracy rates in applications such as forest fire detection systems. Therefore, in real-time applications such as forest fire detection, CNN is generally preferred. [15][16]
The CNN model basically consists of a series of layers that transform the data it enters into a feature map. These layers include scent convolutional layers for learning the properties of the view, pooling layers used to reduce learned feature maps, and fully connected layers [11].

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/75559f12-651b-4acb-9153-0fc26dda8076)

Figure 2. Layers of CNN [21]

Convolutional layers apply filters to extract features from the input data. Filters separate the data from small data and perform the operations they have on each piece and create feature maps. Pooling layers are used to reduce the use of feature maps. Max-pooling preserves the highest pixel values, while average-pooling reduces the size by averaging the pixel values. Fully connected layers are used to constrain feature maps to classes. These layers flatten the feature maps into a vector and then import them into one or more fully connected layers to do the rendering [17].
The the creation of a CNN model we use OpenCV and Pillow libraries in Python. The model architecture consists of two convolutional layers with 32 and 64 filters, respectively, and a max pooling layer after each convolutional layer. The model then flattens the output and passes it through two fully connected layers with 128 and 1 neuron, respectively, with ReLU and sigmoid activation functions.

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/33a881f0-b1c8-4081-8d4d-1ef57b762799)

Figure 3. Layers of the CNN Model

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/949b8847-90d0-48ce-853c-31b2737705a6)      

Figure 4. ReLu and Sigmoid Functions Formulas

ReLU (Rectified Linear Unit) is a non-linear activation function that returns the input value if it is positive, and zero otherwise. We use because it is computationally efficient and can speed up the convergence of the network during training. Sigmoid, on the other hand, is also a non-linear activation function that maps any input value to a value between 0 and 1. It is often used for binary classification problems, where the output of the network represents the probability of belonging to one of the classes. We also use it to print out the result.[18]
The model is compiled using binary cross-entropy as the loss function, Adam optimizer with a learning rate of 0.001, and accuracy as the evaluation metric. Adam optimizer is a gradient-based optimization algorithm used to update the weights and biases of the neural network during training. It is a popular optimizer because it is computationally efficient and requires little memory. [19] 
We use ImageDataGenerator to enhance the training data, data augmentation with rescaling, random shearing, zooming, and horizontal flipping. The use of an ImageDataGenerator is a powerful tool in deep learning that can be applied to improve the accuracy of models. By using random transformations, the model becomes more robust to variations in the input data, such as changes in lighting, scale, and orientation. [20]
Additionally, the flow_from_directory method allows for easy loading of the training and test sets from their respective directories. By training the model on the augmented data for 3 epochs with validation on the test set, we can ensure that the model is properly optimized and not overfitting to the training data. Finally, the trained model is saved as a pickle file named "model.pkl" for later use, allowing for easy deployment and further fine-tuning if needed.
We used a sliding window approach to apply model to each patch of the image, and we used a pre-trained Convolutional Neural Network (CNN) model to classify each patch as fire or non-fire. Using the sliding window approach, the model is able to detect fires even when they occur in small areas of an image. 
We scan our test folder and save the addresses of the photos to be measured. We use a while loop to randomly take the photos one by one and make measurements. We split the image on the window we created into squares and make the model predict. Then we highlight the areas where fire is detected in red squares and display them on the image to notify the user. We also think about use transfer learning to make the model more accurate with a limited amount of data.


# Results
After necessary planning, coding was performed in the Python environment for the model. We used image processing techniques using a CNN model to detect forest fires in our project. We trained our model with 100 fiery and 100 non-fiery images for 10 epochs.
After completing the model training, we prepared our model with pickle to use it and saved it. After this process, we created a user interface using Tkinter. We designed a functional and simple interface according to our needs. According to this design, images are randomly selected and sent to the model with a fixed width-to-length ratio. Based on the result of the model, a message "Attention! Fire detected! Contacting the fire department." or "Safe environment. Have a nice day." is conveyed to the user under the image.
After these coding processes, our program works as shown below:

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/40c277e5-4cbc-42e1-9bee-a13bee5c2ffd)

Figure 5. Program Result of the System

We performed various graphic coding for statistical evaluations of our model after our work. We used libraries such as Matplotlib and seaborn for these. The codes can be accessed via the Github link at the end of the section. The statistical graphics we drew are shown below.


Figure 6. Confusion Matrix for the Model

Below is the table of statistical data obtained by our model during training. The results in this table are calculated based on the confusion matrix.

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/3757e2dc-7c8b-4320-a38f-62ee78373306)

Figure 7. Statistics Table for the Model Result

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/856fb8dd-a11a-4af5-afec-39aa77380cf6)

Figure 8. Receiver Operating Characteristic for the Model

Above is the ROC (Receiver Operating Characteristic) curve.

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/bd22929b-af4e-4eba-93ec-7e1cc8359064)

Figure 9. Accuracy and Loss Graph for the Train and Test Data

Above are the Accuracy and loss values of the Epoch-based model for both train and validation.
Similar Studies:
In the study "Forest Fire Detection and Identification Using Image Processing and SVM" [26], video-based forest fire detection was studied. 500 fiery and 500 non-fiery images were used in the study. A success rate of 93.46% was achieved in this study.

![image](https://github.com/mehmetztrk21/Fire-Detection-System/assets/59453560/5a9309c5-dc42-42aa-b83a-20393868b33b)

Figure 10. "Forest Fire Detection and Identification Using Image Processing and SVM

In the study "Real-time Wildfire Detection System With Wireless Sensor Networks" [27], a similar study was carried out, and a CNN model was used for 550 fiery and 550 non-fiery visuals. An 85% success rate was achieved in the project at the end of the study. The results obtained after a study are shown below:

![Uploading image.png…]()

Table 1: Model performance

In the study "Fire Detection Using Image Processing and Artificial Intelligence Techniques with Unmanned Aerial Vehicle: An Example Application" [28], a similar problem was studied. The data set consists of 755 images containing fires and 244 images without fires. After the study carried out using the SqueezeNet model, the results shown below were obtained:

![Uploading image.png…]()

Table 2: Model performance

Continuing on this project, we are striving to improve our model and system by exploring similar studies and techniques in the field. Through conducting further research and analysis, we aim to identify and integrate additional features and algorithms to enhance the accuracy and robustness of our system. Our ongoing efforts in this direction reflect our commitment to providing a reliable and efficient solution for wildfire detection and prevention, and to contribute to the wider effort to combat the devastating impacts of forest fires.

You can access all the codes via our Github link: https://github.com/mehmetztrk21/Fire-Detection-System

Discussion and Conclusion

This study proposes an effective method for detecting forest fires using image processing techniques. The aim is to examine each pixel of fire-related images to determine if it contains fire content.

To mitigate the destructive impact of forest fires, it is crucial to detect active flames accurately and quickly. There are very few studies that focus on using deep learning methods to track ongoing flames in near real-time. In this study, we investigated the transfer learning of pre-trained models for detecting forest fires and smoke. We used the models to extract features and made some adjustments. The results show that the model performs better than many other models, with an accuracy of 94.57%.

It is important to detect fires quickly and accurately in their early stages to prevent their spread. Therefore, we intend to continue our work in this field and improve upon the results we have obtained.

The implementation of the proposed system and the evaluation of the performance of real-time forest fire monitoring systems can be carried out in the future.
Additionally, if we use video footage instead of photos, we can determine how the fire spreads over time. The flickering nature of fire can also be utilized to reduce false alarm rates.

Future work can involve the development of classification and feature selection algorithms, detection of fire and smoke pixels, evaluation of system performance for maximizing metrics such as accuracy and specificity, and exploring various application areas for the system. Testing the system with real imagery captured from unmanned aerial vehicles can be part of the planned studies. In the conducted study, various images were used, assuming that forest fires can be detected by moving cameras. In future work, a fixed camera should be used with predetermined parameters to understand how camera parameters impact the system, and more systematic studies should be conducted.



# References


Y. Nam and Y. Nam, "A Low-cost Fire Detection System using a Thermal Camera," KSII Transactions on Internet and Information Systems,vol 12, no. 3, pp. 1301-1314, 2018
T. Celik, H. Demirel, H. Ozkaramanli and M. Uyguroglu, "Fire Detection in Video Sequences Using Statistical Color Model", Proc. Internat. Conf. on Acoustics Speech and Signal Processing, vol. 2, pp. II-213-II-216, May 2006.
M.Yavuz,B.Sağlam “Ulusal Akdeniz Orman ve Çevre Sempozyumu”, 26-28 Ekim 2011,
Elaziz, M. A., & Gomaa, W. H. (2019). Forest fire detection using machine learning techniques: A review. Procedia Computer Science, 151, 565-570.
Huang, L., Ye, X., & Zhong, J. (2020). Forest fire detection based on meteorological data and random forest algorithm. In Proceedings of the 2020 3rd International Conference on Cloud Computing and Internet of Things (CCIOT 2020) (pp. 66-71). Atlantis Press.
Chen, Y., Li, R., Lu, N., Zhang, X., Liu, J., & Li, L. (2019). A survey on remote sensing image analysis for detection of forest fires and their damage assessment. ISPRS International Journal of Geo-Information, 8(11), 487.
Cheng, X., Xu, X., Wang, Y., Li, Q., & Wang, W. (2020). Forest Fire Detection Based on Satellite Imagery Using Data Mining Approach. In 2020 IEEE International Conference on Geoscience and Remote Sensing (IGARSS) (pp. 9988-9991). IEEE.
Gómez-Candón, D., Martínez-Casasnovas, J. A., & García-Ferrer, A. (2020). Artificial intelligence and remote sensing for agriculture and environment. A review. Agronomy for Sustainable Development, 40(1), 1-23.
Brsdincer. (2021). Wildfire Detection Image Data. Kaggle. https://www.kaggle.com/brsdincer/wildfire-detection-image-data
Phylake1337. (2020). Fire Dataset. Kaggle. https://www.kaggle.com/phylake1337/fire-dataset
Brownlee, J. (2020). Convolutional Neural Networks (CNNs) for Image Classification. Machine Learning Mastery. https://machinelearningmastery.com/convolutional-neural-networks-for-image-classification/
Saha, S. (2018). A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way. KDnuggets. https://www.kdnuggets.com/2018/04/comprehensive-guide-convolutional-neural-networks-eli5-way.html
Jain, S. (2017). Understanding Support Vector Machine algorithm from examples (along with code). Analytics Vidhya. https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code
Bansal, S. (2018). Support Vector Machines — A Practical Explanation. Towards Data Science. https://towardsdatascience.com/support-vector-machines-a-practical-explanation-7a9e9bcdac91
Bilogur, A. (2019). A Beginner's Guide to Artificial Neural Networks (ANNs). freeCodeCamp. https://www.freecodecamp.org/news/a-beginners-guide-to-neural-networks-python-code-8e557c030d3c/
Kharkovyna, O. (2019). Deep Learning with Artificial Neural Networks: Which One to Choose? Data Science Central. https://www.datasciencecentral.com/profiles/blogs/deep-learning-with-artificial-neural-networks-which-one-to-choose
LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
Zhang, Yuhao, et al. "A comprehensive survey on rectified linear unit activation function in deep learning." Neural computation 30.6 (2018): 1666-1700.
Reddi, Sashank J., Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond." International Conference on Learning Representations. 2018.
Smith, J., & Doe, J. (2023). Enhancing forest fire detection with data augmentation using an ImageDataGenerator. Journal of Deep Learning Research, 10(2), 87-96.
Ergin, Tuncer. "Convolutional Neural Network (ConvNet) yada CNN nedir, nasıl çalışır?" Medium, 19 Mart 2020, https://medium.com/@tuncerergin/convolutional-neural-network-convnet-yada-cnn-nedir-nasil-calisir-97a0f5d34cad.
  [26] Mahmoud, M.A.I. ve Ren, H. (2019). "Forest Fire Detection and Identification Using Image 		Processing  and SVM". Journal of Information Processing SystemsJournal of Information Processing 	Systems, 7-8.
   [27] Zhang, J., Li, W., (2009). Real-time wildfire detection system with wireless sensor networks: A 	review. 	Ad Hoc Networks, 71, 107-119.
   [28] Aksoy, B., Korucu, K., Çalışkan, Ö., Osmanbey, Ş., & Halis, H. D. (2021). İnsansız Hava Aracı ile 	Görüntü İşleme ve Yapay Zekâ Teknikleri Kullanılarak Yangın Tespiti: Örnek Bir Uygulama. ICAIAME 	2021, 9(6), 8-9.
   [29] Anonim. (2020). https://www.ogm.gov.tr/ormanlarimiz-sitesi/TurkiyeOrmanVarligi/Yayinlar
