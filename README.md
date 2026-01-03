# 🏥 U.S. Chronic Disease Predictive Intelligence
### **A Hybrid Machine Learning & Geospatial Forecasting Dashboard**

## 📌 Background & Motivation
Chronic diseases—such as Diabetes, CVD, and Cancer—account for nearly **75% of global deaths** and the majority of U.S. healthcare spending. Current public health surveillance is often **reactive**, reporting data years after collection. 

**The Goal:** This project moves from "What happened?" to **"What will happen?"** It provides a predictive framework to assist policymakers in proactive resource allocation, focusing on the rising burden of an aging population and demographic health disparities.

---

## 🛠️ Technical Architecture & Modeling
This project utilizes a **Hybrid Forecasting Engine** to overcome the limitations of traditional tree-based models in time-series extrapolation.

* **Champion Model:** **Random Forest Regressor** ($R^2 \approx 94\%$). Random Forest was selected over Gradient Boosting for its robustness against the high variance and noise inherent in public health data, ensuring better generalization for future scenarios.
* **Geospatial Intelligence:** Raw coordinates were parsed into continuous `Latitude` and `Longitude` features to capture spatial correlations.
* **Unsupervised Learning:** Applied **K-Means Clustering** to categorize U.S. states into "Health Profiles," revealing that regional environment is a critical predictor of disease prevalence.
* **Trend Integration:** Since Random Forest cannot inherently extrapolate, I integrated **disease-specific annual growth rates** (0.8% to 2.3%) synthesized from **CDC (2025)** and **AHA (2024)** research reports.

---

## 🖥️ The Streamlit Application
The project is deployed as an interactive Web App, allowing stakeholders to:
1.  **Scenario Simulator:** Adjust a slider from the **2022 Baseline** to **2030** to see compounding health burdens.
2.  **State Rankings:** Identify "Top 5 / Low 5" most affected states for any specific chronic condition.
3.  **Demographic Deep-Dive:** Compare prevalence across Race, Ethnicity, Gender, and Age groups with interactive Plotly visuals.

---


## ⚠️ Key Challenges & Solutions
* **The Extrapolation Trap:** Standard Machine Learning models (like Random Forest) are non-parametric; they cannot predict values outside the range of their training data. Since our data ended in 2022, the model would naturally "flatline" for 2030. I solved this by building a Hybrid Inference Engine that applies research-backed annual growth multipliers to the model's 2022 baseline.

* **Metric Fragmentation:** The CDC dataset contains multiple, often conflicting metrics:

     **Crude vs. Age-Adjusted Rates:** Crude rates show the actual burden, while Age-Adjusted rates remove the "aging" effect.

     **Prevalence vs. Numerical Counts:** Dealing with percentages vs. raw patient counts.

     **Stratification Overlap:** Managing data points that vary by Job, Age, Race, and Gender simultaneously. Solution: I implemented a strict data-normalization pipeline to filter for "Crude Prevalence" to ensure "Apple-to-Apple" comparisons across all 50 states and demographic groups.

* **Data Heteroscedasticity:** Public health reporting variance is not uniform across states. I utilized a Random Forest Ensemble to average out this noise, providing more stable predictions than a traditional Linear Regression.

---

## 🚀 Future Enhancements
* **SDoH Integration:** Incorporating Social Determinants of Health (poverty rates, food access) to reduce unexplained variance.
* **Deep Learning:** Implementing **LSTM (Long Short-Term Memory)** networks for fluid temporal modeling.
