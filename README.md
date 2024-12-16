# NFL Injury Analysis
In this project, I developed a machine learning model that predicted the likelihood an NFL player got injured during a game based on a variety of factors (player position, weather, age). I implemented various classic ML algorithms (Stochastic Gradient Descent, Gradient Boosted trees, Logistic Regression) and deep neural networks for prediction, but found that Logistic Regression performed the best. I used tools such as Python, Pandas, Scikit-learn, and PyTorch for the data wrangling and modeling. This project was part of the NFL's 1st & Future competition on Kaggle which aims to used data science to better understand player health, safety and performance.

This was a project I consider impressive as I was able to successfully build a model that addressed a relevant issue in sports and health. This project went through the entire data science project life cycle, as I spent several weeks finding relevant data sources and forming hypotheses, another few weeks cleaning and wrangling the data to extract relevant input and output features, then two weeks building the machine learning models and improving them through PCA dimensionality reduction, SMOTE oversampling, and hyperparameter tuning.

# Important Files
**data_prep.ipynb**\
This file tests hypotheses related to player injury for statistical significance and cleans the data to only include relevant information.

**modeling.ipynb**\
This file preprocesses the data to prepare it for model input, implements Stochastic Gradient Descent, Gradient Boosted trees, Logistic Regression with Scikit-learn, and displays metrics such as ROC curve, learning loss, and accuracy to evaluate each model.

**modeling_NN.ipynb**\
This file preprocesses the data to prepare it for model input, implements a deep neural network in PyTOrch, and displays metrics such as ROC curve, learning loss, and accuracy to evaluate the model.
