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
└── .gitignore                            # Rules for ignoring environment and cache files
🛠️ Installation & Setup
Clone the repository:

git clone https://github.com/fatma-crypto/NTI-Final-Project.git
cd NTI-Final-Project
Create a Virtual Environment (Recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install Dependencies: Ensure you have Pandas, NumPy, Scikit-Learn, and Streamlit installed:

pip install streamlit pandas numpy scikit-learn joblib
🚀 Usage
To run the web application locally:

streamlit run app.py
This will launch the app in your default web browser at http://localhost:8501.

How to Predict:
Navigate to the sidebar on the left.
Enter the project parameters (e.g., Budget, Team Size, Complexity, Methodology).
Click on the Analyze Risk button.
The panel will display whether the project is at High or Low risk, alongside an explanation of the contributing factors. You can also ask the Risk Chatbot for further details!
