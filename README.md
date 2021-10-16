# Waste Segregator

![Python 3.7](https://img.shields.io/badge/Made_With-Python_3.7-green?style=for-the-badge&logo=appveyor)
![GitHub repo size](https://img.shields.io/github/repo-size/squash-bit/Waste-Segregator)
![GitHub stars](https://img.shields.io/github/stars/squash-bit/Waste-Segregator?style=social)

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Procedure](#procedure)
  * [Model](#model)
  * [Installation](#installation)
  * [Garbage Prediction](#waste_prediction)
  

## Overview
This is a Garbage Classification Web Application built using both Streamlit and PyTorch.

## Motivation
The present way of separating waste/garbage is the hand-picking method, whereby someone is
employed to separate out the different objects/materials. The person, who separate waste, is prone to diseases due to the harmful substances in the garbage. With this in mind, it motivated me to develop an automated system which is able to sort the waste in a more accurate way than the manual one. With the system in place, the beneficial separated waste can still be recycled and converted to energy and fuel for the growth of the economy. 


For people who are not experts at Django or Flask, Streamlit can be a good alternative to build custom Python web apps for data science. I chose image classification as the task here because computer vision is one of the most popular areas of AI currently, powered by the advancement of deep learning algorithms.

## Procedure
  * Install Streamlit
  * Build UI
  * Build Model
  * Test Results
  * Next Steps
  
## Model
I have chosen the pretrained ResNet50 model to perform classification. To figure out how the model was built, take a look at the jupyter notebook entitled `Waste_Segregator.ipynb`...
  
## Installation
```bash
pip install -r requirements.txt
```
then, you run the code by typing the command below in your terminal
```bash
streamlit run app.py
```

## Garbage Prediction
<!--- Add image --->
![Study for Finals](https://github.com/squash-bit/Waste-Segregator/blob/master/Assets/screenshot1.png?raw=true)

![Study for Finals](https://github.com/squash-bit/Waste-Segregator/blob/master/Assets/screenshot2.png?raw=true)

![Study for Finals](https://github.com/squash-bit/Waste-Segregator/blob/master/Assets/screenshot3.png?raw=true)