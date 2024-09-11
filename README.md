# Second Year ENSICAEN Project 

This is a research project conducted throughout the second year at the ENSICAEN National Engineering School of Caen

Students : **Cecile LU, Paul NGUYEN, Zeyd BOUMAHDI, Anis AHMEDZAID, Noura OUTLIOUA**
Supervisors: **Christophe ROSENBERGER, Tanguy GERNOT**

Our project involves performing machine learning/prediction tasks on encrypted data. To achieve this, we aim to train our models to recognize numerical images. For further details please refer to the documents in the ```/doc``` directory.

The ```/doc``` directory also contains a state of the art article on the topic of Machhine Learning over Encrypted Data

## Models

The details on the models can be found inside the doc in "Rapport_Apprentissage_Securise"

- **EncCNN**: This model is based on Convolutional Neural Networks (CNNs) and is designed to operate on encrypted data.
- **EncFCNN**: This model is a variant of Fully Convolutional Neural Networks (FCNNs) adapted for encrypted data.
- **FHE**: Fully Homomorphic Encryption (FHE) is a cryptographic technique that enables computations on encrypted data without decryption.

## Documentation

- **kick_off**: This document outlines the initial plans, objectives, and timeline for the project.
- **final_presentation**: This presentation summarizes the project's outcomes, including results, conclusions, and future directions.
- **Secure_Learning_Report**: This report provides a detailed analysis of secure learning techniques applied in the project, discussing their effectiveness and limitations.

## Demo

To execute the web demo, you need to install streamlit ```pip install streamlit```

After installing it you can launch it with this command :
```streamlit run demo/demo.py```
