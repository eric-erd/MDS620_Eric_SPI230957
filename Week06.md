# Week 6 – Tutorial

## Section A: Lecture 6 Review
### 1.	Compare Supervised Learning and Unsupervised Learning. Is Active Learning considered a form of Supervised or Unsupervised Learning? 

#### Supervised and Unsupervised learnings are 2 major categories of machine learning.
#### In supervised learning, we use a dataset which contains labels which is simply correct answers of given datas. It's like students learn new things with a teacher who really knows all the correct answers. In every time students make mistakes, teacher will correct students and measure them by grade. It's like a loss function. Most of the time we use gradient decent to reduce loss values which helps to change the model's numbers (matrix)
#### In Unsupervised learning, there is no labeled data, no correct answers only what we have is a data without any label. At least we have some data even though it has no label which means it's better than having no data. We can do analysis using some famous algos like k-means and get some insights about the data that we have. If I use same metaphor, it's like everyone is living their life first time. We don't know everything about life, there is no life teacher, we just live and trained by ourself.
#### Active learning strategy helps to reduce human resources and costs to labeling datas that don't have any label. Basically it's considered to be semi-supervised learning as the data that we have is not completly labeled. We are trying to label all of that datas using some strategy to reduce costs of human annotations.

### 2.	Explain the concept of Active Learning and its importance in machine learning.
#### When we have a few amount of data that is labelled and we need to label all other datas by minimizing human annotation cost, the active learning strategy will be used. Which helps to label unlabeled data by running algorithm.

### 3.	Compare and contrast the roles of Data Augmentation and Active Learning in enhancing the performance of machine learning models.
#### Data augmentation is used to create new data using current data, Active learning is used to label unlabeled data that already we have. Both method are useful to increase amount of labeled data to train a model.

### 4.	Describe the training process of Active Learning in a pseudocode.
#### Specific percent of all datas should be labeled by human first, we will train a model using only labeled data that predicts all possible labels then we predict all other unlabeled data using the model we trained. We also need to calculate uncertainty level of each predict results using some specific method which fits the project for example entropy. Uncertainty level indicates our model is not sure about those predicts. The most uncertain predicts need to be labeled by human. Then we will use this labeled data to enhance our current model's result and to continue this process until we'll consider our model is good enough to label all remaining unlabeled data.

### 5.	Describe the role of the "oracle" in the Active Learning process. What are the challenges associated with relying on oracles in real-world applications? How can these challenges be mitigated?
#### Oracle is just simply a human annotator like labelling unlabeled data by someone. Some challenges are human still labelling data by wrong, very costy, and time is also costy. Cross checking techniques helps to reduce human error (bias) and algos like Active learning helps to reduce both money cost and time cost.

### 6.	Describe some applications of Active Learning in real life.
#### I was faced exact same situation that Active learning helps at 2018. My former team was trying to create a classification model using some text data that's collected from web sites. What we need to do was to know what sort of service they (web sites) offer. I don't remember all of the classes but some are IT, economy, advertise, marketing and so on. We need to create a model to predict what services the web site offer based on content that's on the web page.

## Section B: Handwritten Digits Recognition with Active Learning
### 1.	Provide Detailed Comments:
### Add detailed comments in each segment of the provided code to explain what each part does. Be thorough in your explanations to ensure clarity.
### Take screenshots of each segment that you added comments to the Week6.md file in your GitHub repository.
### 2.	Run the Code:
### •	Execute the provided code within your Jupyter Notebook.
### 3.	Capture Results:
### •	Take screenshots of the results you achieved after running the code.
### •	Include these screenshots in your Jupyter Notebook and also paste them into the Week6.md file in your GitHub repository.
### 4.	Analyze Results:
### •	Analyze and provide explanation for the achived results in your Week6.md file in your GitHub repository.

<!-- Section C: Assessment 2 – Group Project
1.	Form a group of 2 students.
2.	Read the Assessment 2 Task Instructions together.
3.	Discuss with your group members and propose a plan of implementing and reporting the project in Week6.md in your Github Repository.

Provided Code for Section B
This example demonstrates an active learning technique to learn handwritten digits using label propagation.











In this lab, the training process provides the machine learning algorithm correct labels of the most uncertain examples. No human interaction is required.
We start by training a label propagation model with only 10 labeled points, then we select the top five most uncertain points to label. Next, we train with 15 labeled points (original 10 + 5 new ones). We repeat this process four times to have a model trained with 30 labeled examples. Note you can increase this to label more than 30 by changing max_iterations. Labeling more than 30 can be useful to get a sense for the speed of convergence of this active learning technique.
A plot will appear showing the top 5 most uncertain digits for each iteration of training. These may or may not contain mistakes, but we will train the next model with their true labels.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.semi_supervised import LabelSpreading

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

# Comments to be provided
# selecting first 330 rows of data for train by using indices from original data
X = digits.data[indices[:330]]
# selecting target/answers/labels for selected training data
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

# Comments to be provided
# gets total number of samples that selected for training
n_total_samples = len(y)
n_labeled_points = 10
max_iterations = 5

# Comments to be provided
# getting indices of unlabeled indices for active learning
unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()
# Comments to be provided
# iterating number of iteration which provided above. Which is a hyperparam for training process.
for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = np.copy(y)
    # Comments to be provided
    # unlabeling original data using unlabeled_indices var, because original data is completely labeled. 
    y_train[unlabeled_indices] = -1

    # Comments to be provided
    # LabelSpreading is a library which helps to labeling data by training given dataset.
    lp_model = LabelSpreading(gamma=0.25, max_iter=20)
    lp_model.fit(X, y_train)

    # Comments to be provided
    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print(
        "Label Spreading model: %d labeled & %d unlabeled (%d total)"
        % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples)
    )

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # Comments to be provided
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

    # Comments to be provided
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[
        np.isin(uncertainty_index, unlabeled_indices)
    ][:5]

    # keep track of indices that we get labels for
    delete_indices = np.array([], dtype=int)

    # for more than 5 iterations, visualize the gain only on the first 5
    if i < 5:
        f.text(
            0.05,
            (1 - (i + 1) * 0.183),
            "model %d\n\nfit with\n%d labels" % ((i + 1), i * 5 + 10),
            size=10,
        )
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        # for more than 5 iterations, visualize the gain only on the first 5
        if i < 5:
            sub = f.add_subplot(5, 5, index + 1 + (5 * i))
            sub.imshow(image, cmap=plt.cm.gray_r, interpolation="none")
            sub.set_title(
                "predict: %i\ntrue: %i"
                % (lp_model.transduction_[image_index], y[image_index]),
                size=10,
            )
            sub.axis("off")

        # labeling 5 points, remote from labeled set
        (delete_index,) = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += len(uncertainty_index)

f.suptitle(
    (
        "Active learning with Label Propagation.\nRows show 5 most "
        "uncertain labels to learn with the next model."
    ),
    y=1.15,
)
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2, hspace=0.85)
plt.show() -->
