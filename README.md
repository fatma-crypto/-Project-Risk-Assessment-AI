📌 Features

Risk Prediction Engine: Utilizes a trained Machine Learning model to calculate the exact probability of a project failing or facing severe risks (High Risk vs. Low Risk).
Dynamic Feature Extraction: Automatically computes advanced metrics in the background like Budget per Month, Workload Index, and Risk Intensity.
AI Explanations & Chatbot: Provides natural language insights to explain why a project was deemed high risk, dynamically comparing inputs against a baseline of successful past projects.
Glassmorphism UI: A beautiful, modern, and engaging frontend built using custom CSS styles in Streamlit.
Visual Analytics: Interactive display of Feature Importance and Probability scores.


📂 Repository Structure
│
├── data/                                 # Cleaned / preprocessed data files
├── modeling/                             # Saved ML models, training scripts, and evaluation metrics
│   ├── train.py                          # Script used for training the model
│   ├── best_model.joblib                 # Serialized final Machine Learning model
│   └── feature_importance.png            # Visual output of the most important metrics
│
├── preprocessing/                        # Data cleaning and pipeline logic
│   └── pipeline.py
│
├── Project_Risk_Assessment_Complete.ipynb # Complete end-to-end Jupyter Notebook (EDA -> Modeling)
├── app.py                                # The main Streamlit web application
├── project_risk_raw_dataset.csv          # Core Dataset
└── .gitignore                            # Rules for ignoring environment and cache  
