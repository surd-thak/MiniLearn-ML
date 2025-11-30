# **MiniLearn**

A minimal, from-scratch machine learning library built for **educational purposes**. MiniLearn provides clean, Scikit-learn–inspired APIs and implementations of classical machine learning algorithms using only **Python, NumPy, and Matplotlib**.
This project is designed to help learners understand *how* ML algorithms work under the hood.

---

## **Highlights**

* 🔧 **From-scratch ML algorithms** with a unified `.fit()` / `.predict()` interface
* 🧮 Implemented models:

  * Linear Regression
  * Polynomial Regression
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * K-Means Clustering
  * Multi-layer Neural Network with **manual backpropagation**
* 📊 **Diagnostic visualization tools** (loss curves, accuracy trends, decision boundaries)
* 🧪 **Evaluated on MNIST** — Neural Network achieved **~95% accuracy**
* 🎯 Ideal for learning, experimentation, and understanding core ML concepts

---

## **Installation**

```bash
# Clone the repository
git clone https://github.com/surd-thak/MiniLearn-ML
cd MiniLearn-ML

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# On Windows (CMD):
.\venv\Scripts\activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## **Features**

### **Linear Models**

* Linear Regression
* Polynomial Regression
* Logistic Regression

### **Instance-Based Methods**

* K-Nearest Neighbors (KNN)

### **Clustering**

* K-Means

### **Neural Networks**

* Fully-connected feedforward neural network
* Manual backpropagation implementation
* Supports multiple activation functions

### **Utility Tools**

* Train/test splitting
* Feature scaling
* Visualization helpers
* Metrics (accuracy, loss tracking, etc.)

---

## **Usage**

Examples for each algorithm are available in the `examples/` directory:

```bash
python examples/example_logistic_regression.py
python examples/example_kmeans.py
python examples/example_neural_network.py
```

---

## **Project Structure**

```
MiniLearn-ML/
│
├── linear_model/
├── neural_network/
├── clustering/
├── neighbors/
├── utils/
├── examples/
└── README.md
```

---

## **Future Improvements**

* Add `pyproject.toml` for packaging
* Unit tests (pytest)
* More datasets and visualizations


---




