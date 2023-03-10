**<center><font size=5>Tumor Detector - AvaSure Project</font></center>**
***
**author**: Alejandro Alemany

**date**: January 21st, 2022

**Table of Contents**
- <a href='#intro'>1. Project Overview</a> 
    - <a href='#dataset'>1.1. Data Set Description</a>
    - <a href='#tumor'>1.2. What is Brain Tumor?</a>
- <a href='#env'>2. Methodogy for this project</a>
- <a href='#import'>3. Documentation</a>
- <a href='#test'>4. Testing the model</a>

# <a id='intro'>1. Project Overview</a>

This project is a capstone assignment I completed during my master's studies. The accuracy accomplished at the time was close to 60%. 
The goal of this notebook and project is not to improve that score but to deploy the model and follow MLOps principles to accomplish it. Therefore, when testing the model, it is crucial to notice that the model could be better, but it was not the goal of this deployment to improve it. 
Can you retrain the model and get better results? Better algorithms and data are now available, and we can significantly improve the model. Therefore, I plan to work on better accuracy in the future.

## <a id='dataset'>1.1. Data Set Description</a>

The image data that was used for this problem is Brain MRI Images for Brain Tumor Detection. It conists of MRI scans of two classes:

* `NO` - no tumor, encoded as `0`
* `YES` - tumor, encoded as `1`

Unfortunately, the data set description doesn't hold any information where this MRI scans come from and so on.

## <a id='tumor'>1.2. Identifying Brain Tumors</a>

> A brain tumor is a mass or lump of abnormal cells found in the brain. There are several types of brain tumors. Some brain tumors are not cancerous (benign), and others are cancerous (malignant). Brain tumors can start in the brain (primary brain tumors), or cancer can begin in other parts of the body and then spread to the brain (secondary or metastatic brain tumors). The rate at which a brain tumor grows can vary greatly. The growth rate and location of the brain tumor determine how it will affect the functioning of the nervous system.
>
> ![](https://upload.wikimedia.org/wikipedia/commons/5/5f/Hirnmetastase_MRT-T1_KM.jpg)
>
> *Brain metastasis in the right cerebral hemisphere from lung cancer, shown on magnetic resonance imaging.*

Source: [Wikipedia](https://en.wikipedia.org/wiki/Brain_tumor)

# <a id='env'>2. Methodology for this project</a>

![image](https://user-images.githubusercontent.com/55760198/213964743-5f399f57-867f-4f67-8e85-99d64da8c19f.png)

1. First, we will integrate our model into MLFlow to conduct experiments. 
2. After selecting the best experiment, we will store it in a Docker container.
3. We will push the container to AMAZONECR.
4. We are going to deploy our model to Amazon SageMaker from our container stored in AMAZONECR
5. SageMaker will create the endpoints so we can access the model. 
6. Deployed model is saved in S3 for quick access, and the user can access the model through the endpoints. 

# <a id='import'>3. Documentation</a>

**What is MLFLOW?**

It is an open-source machine-learning platform created by Databricks. It manages the entire ML lifecycle (from inception to production) and is designed to work with any Machine Learning library.
We ran two experiments:

![Screenshot 2023-01-21 181516](https://user-images.githubusercontent.com/55760198/213964855-0d358af7-19f3-4f7d-b99a-162688d6117c.png)

We selected the experiment with the best accuracy and prepared to store it and push it to AmazonECR. We can find the information needed inside the model:

![Screenshot 2023-01-21 181711](https://user-images.githubusercontent.com/55760198/213964884-b713358c-6085-4563-869a-fa1f9602ad43.png)

![Screenshot 2023-01-22 093424](https://user-images.githubusercontent.com/55760198/213964900-deaa7911-605d-4ed5-9ec3-518e4ea28dee.png)

The container in AmazonECR is ready to deploy on Amazon SageMaker. From there, we can create the endpoints we will use to access the model.

![Screenshot 2023-01-22 124350](https://user-images.githubusercontent.com/55760198/213964935-74237277-3849-48dd-b84e-d1baa4d7131e.png)

![Screenshot 2023-01-22 165740](https://user-images.githubusercontent.com/55760198/213965055-2df38ffc-14f5-4df9-b8e7-733085bfaa81.png)


# <a id='test'>4. Testing the model</a>

I have built a web application in which we can submit images to the API and classify it. It is a simple web app with two endpoints (GET, POST) to handle the transfer of information in a friendly-user approach. 

![image](https://user-images.githubusercontent.com/55760198/213965129-6cf70724-ff59-44de-ae08-6974d8020acf.png)

We uploaded a brain radiography and hit enter:

![image](https://user-images.githubusercontent.com/55760198/213965188-b7b85a4f-f6d1-4748-b44c-1c46f2b8e226.png)

![image](https://user-images.githubusercontent.com/55760198/213965224-657d41a5-b5d9-41d7-898f-1ace699dda0d.png)

Running another a second test:

![image](https://user-images.githubusercontent.com/55760198/213965255-78b2158e-3e67-4777-aff3-79539b5a21ce.png)

Running a third test:

![image](https://user-images.githubusercontent.com/55760198/213965284-d07b1b83-8106-4f97-8a4c-75c97d342e4c.png)


```python

```
