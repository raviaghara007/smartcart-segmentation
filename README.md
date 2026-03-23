# 🛒 SmartCart Customer Segmentation System

*An AI-powered customer segmentation system using Machine Learning (Clustering) to analyze customer behavior and generate business insights.*

---

## 📌 Table of Contents

* [Overview](#overview)
* [Live Application](#live-application)
* [Business Problem](#business-problem)
* [Dataset](#dataset)
* [Machine Learning Approach](#machine-learning-approach)
* [Tools & Technologies](#tools--technologies)
* [Project Structure](#project-structure)
* [Application Features](#application-features)
* [Deployment](#deployment)
* [Author](#author)

---

## Overview

This project implements an end-to-end **Customer Segmentation System** using unsupervised machine learning techniques.

The system analyzes customer demographics, purchasing behavior, and engagement patterns to group customers into meaningful segments.

A **Streamlit-based interactive dashboard** is built on top of the model to visualize clusters, explore data, and generate actionable business insights.

---

## 🚀 Live Application

🔗 **Live Demo:**  
https://smartcart-segmentation-izzmvczh59ac6rdvk8bw9p.streamlit.app/

---


## Business Problem

Businesses often apply the same marketing strategy to all customers, which leads to:

* Inefficient marketing campaigns
* Poor customer engagement
* Loss of high-value customers
* Increased churn

The goal of this project is to segment customers into distinct groups so businesses can:

* Target customers more effectively
* Improve customer retention
* Increase revenue through personalization

---

## Dataset

The dataset contains customer information related to demographics, spending habits, and purchasing behavior.

### Key Features

* Income
* Age
* Education
* Marital Status
* Total Spending across product categories
* Number of purchases (web, store, catalog)
* Recency (last purchase)
* Customer tenure

### Dataset Size

* ~2240 customers
* 20+ features

---

## Machine Learning Approach

* **Problem Type:** Unsupervised Learning (Clustering)

### Algorithms Used

* K-Means Clustering
* Agglomerative Clustering

### Preprocessing Steps

* Handling missing values
* Feature engineering (Age, Total Spending, Customer Tenure)
* One-Hot Encoding for categorical features
* Feature Scaling using StandardScaler

### Dimensionality Reduction

* PCA (Principal Component Analysis) for visualization

### Evaluation

* Elbow Method
* Silhouette Score

---

## Tools & Technologies

* **Python** – Core programming language
* **Pandas & NumPy** – Data processing
* **Scikit-learn** – Machine learning models
* **Plotly & Matplotlib** – Data visualization
* **Streamlit** – Interactive dashboard
* **Git & GitHub** – Version control

---

## Project Structure

```text
smartcart-segmentation/
│
├── smartcart_app.py          # Streamlit dashboard
├── smartcart.ipynb           # Data analysis & model building
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── smartcart_customers.csv   # Data


```

---

## Application Features

* Interactive customer segmentation dashboard
* Cluster visualization (2D & 3D PCA plots)
* Cluster-wise business insights
* Exploratory Data Analysis (EDA)
* Optimal cluster selection using Elbow & Silhouette methods
* Data filtering and exploration
* Download clustered dataset

---

## Deployment

The application can be deployed using **Streamlit Cloud**.

---

## Author

**Ravi Aghara**

📧 Email: [aaghararavi@gmail.com](mailto:aaghararavi@gmail.com)
🐙 GitHub: https://github.com/raviaghara007
🔗 LinkedIn: https://www.linkedin.com/in/raviaghara07/
