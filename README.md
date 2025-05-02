# Transoesophageal Echocardiography Image Analysis

This repository contains the code and analysis for evaluating image quality in Transoesophageal Echocardiography (TOE) images using various image similarity and regression techniques. The project involves processing ultrasound images captured using a high-fidelity simulator, comparing them to gold standard images, and performing various statistical analyses to assess the quality of images produced by expert and novice participants.

## Background

Transoesophageal echocardiography (TOE) is an imaging technique used to capture high-quality images of the heart using a flexible probe with an ultrasound transducer. The goal of this project is to analyze the quality of TOE images and determine factors that influence the performance of the interventionists, such as the level of experience.

The dataset contains 195 TOE images from 20 volunteers, where each volunteer captured 10 cross-sectional views. The images are labeled with quality scores from expert anaesthetists based on a set of criteria and general impression of image quality.

## Requirements

### Python Libraries
To run the code, you will need the following Python libraries:
- `scipy`
- `numpy`
- `scikit-image`
- `scikit-learn`
- `opencv-python`
- `matplotlib`

You can install the required libraries using the following command:

```bash
pip install scipy numpy scikit-image scikit-learn opencv-python matplotlib
