# CAN-IDS
OS Project

## Work-Flow:
* Go with prerequisites first. Not everything can be learned before. You can learn things by doing it.

                           CAN DATASET
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       YOUR LAPTOP       │
                 │                         │
                 │  1. Data Cleaning       │
                 │  2. EDA                 │
                 │  3. Feature Engineering │
                 │  4. Data Sampling       │
                 │  5. Train/Test Split    │
                 └────────────┬────────────┘
                              │
                              ▼
                     PROCESSED DATASET
                              │
                              │ Upload
                              ▼
                 ┌─────────────────────────┐ 
                 │    COLAB / KAGGLE GPU  │
                 │                         │
                 │  6. Model Training      │
                 │  7. Hyperparameter      │
                 │     Optimization        │
                 │  8. Model Comparison    │
                 │  9. Deep Learning       │
                 │ 10. Model Selection     │
                 └────────────┬────────────┘
                              │
                              ▼
                       BEST MODEL
                              │
                    Export (.pkl/.onnx)
                              │
                              │ Download
                              ▼
                 ┌─────────────────────────┐
                 │       YOUR LAPTOP       │
                 │                         │
                 │ 11. Load Trained Model  │
                 │ 12. Real-Time CAN       │
                 │     Traffic             │
                 │ 13. Feature Extraction  │
                 │ 14. Real-Time Inference │
                 │ 15. Attack Detection    │
                 │ 16. Logging / Dashboard │
                 └─────────────────────────┘

****** Phase-1 : Choosing the data sets && Dividing the training and testing data ******
                 REAL CAN DATA
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   CarChallenge 2020          Car-Hacking
          │                       │
          │                  External Test
          ↓                       │
    Feature Engineering           │
          │                       │
          ↓                       ↓
       ML Models ─────────→ Generalization
          │
          ↓
   Best IDS Model
          │
          ↓
      ROAD Dataset
          │
          ↓
 Advanced/Stealthy Attack
       Evaluation
          │
          ↓
   Real-Time CAN IDS


***** Phase-1: Data Integrity check completed ***** 
** No restricted constraints ** 

======================================================

***** Phase-2: Data Cleaning ***** 










   

## Tech Stack
* Python
* Scikit-learn / TensorFlow (for model training)
* Pandas & NumPy (for data processing)


Step-0 : Collect the data
Data collection done


Step-1 : Choose the proper environment for this project 
Environment Decision done


Step-2 : What to be installed
Using VCAN modules for simulation but not decided which one to use 






