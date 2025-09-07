# Fraud Detection Explorer  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)  
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-orange?logo=plotly)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)  

---

## 📌 Project Overview  

**Fraud Detection Explorer** is an interactive **Streamlit** web application for **Exploratory Data Analysis (EDA)** of financial transaction data.  
It enables **data scientists, ML engineers, and analysts** to:  

- Visualize transaction data patterns.  
- Explore feature distributions.  
- Understand class imbalances.  
- Test fraud prediction with a pre-trained **Isolation Forest model**.  

This tool provides a hands-on interface for **detecting potential fraudulent activities** in datasets.  

---

## ✨ Features  

| Feature | Description |
|---------|-------------|
| 🗂️ **Data Preview** | View dataset head, tail, and summary statistics. |
| 📊 **Feature Exploration** | Interactive Plotly histograms & box plots for features. |
| ⚖️ **Class Distribution** | Visualize imbalance between fraud and non-fraud transactions using bar & pie charts. |
| 🔥 **Correlation Heatmap** | Explore relationships between features with an interactive heatmap. |
| 🤖 **Model Prediction** | Demo section: Input feature values via sliders & get fraud prediction from a pre-trained Isolation Forest model. |

---

## 🛠 Technology Stack  

- **Python**  
- **Streamlit**  
- **Pandas / NumPy**  
- **Plotly**  
- **Scikit-learn**  
- **Joblib**  

---

## ⚙️ Requirements  

Dependencies are listed in `requirements.txt`:  

```
streamlit==1.22.0
pandas==1.5.3
numpy==1.24.3
plotly==5.11.0
scikit-learn==1.2.2
joblib==1.2.0
```

Install them with:  

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started  

1. **Clone the repository**  

```bash
git clone https://github.com/your-username/fraud-detection-explorer.git
cd fraud-detection-explorer
```

2. **Install dependencies**  

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**  

```bash
streamlit run fraud_detection_eda.py
```

---

## 📖 Usage  

Once the app is running:  

### 🗂️ Data Upload  
- Upload your financial transaction dataset (`.csv` format).  
- Preview **head, tail, and summary statistics**.  

### 📊 Feature Exploration  
- Select any feature.  
- View **interactive histograms** & **box plots**.  

### ⚖️ Class Distribution  
- Check **fraud vs. normal transactions**.  
- View **bar & pie charts** for imbalance.  

### 🔥 Correlation Heatmap  
- Analyze feature relationships.  
- Use **hover effects** to inspect correlations.  

### 🤖 Model Prediction  
- Adjust sliders for feature inputs.  
- Get a **fraud prediction (Fraud / Not Fraud)** from the **Isolation Forest model**.  

---

## ☁️ Deployment  

You can easily deploy this app to **Streamlit Community Cloud**:  

1. Push your project to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).  
3. Connect your GitHub repo.  
4. Deploy with entry point:  

```
fraud_detection_eda.py
```

---

## 🐛 Troubleshooting  

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure dependencies are installed with `pip install -r requirements.txt`. |
| App not loading large dataset | Use a **sample dataset** or optimize preprocessing. |
| Plotly charts not displaying | Refresh browser or check Streamlit logs. |

---

## 📜 License  

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  

---

## 🙏 Acknowledgments  

- [Streamlit](https://streamlit.io/) for the interactive app framework.  
- [Plotly](https://plotly.com/python/) for interactive visualizations.  
- [Scikit-learn](https://scikit-learn.org/stable/) for the Isolation Forest model.  
- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) for sample data.  
