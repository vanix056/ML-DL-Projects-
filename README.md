# ML & DL Projects

A comprehensive collection of **Machine Learning** and **Deep Learning** projects implemented in Python using Jupyter Notebooks. The repository covers foundational algorithms, advanced neural network architectures, NLP with transformers, and two end-to-end production-grade practice projects with REST API backends.

---

## Overview

This repository serves as a structured learning and reference resource spanning the full ML/DL spectrum — from implementing gradient descent from scratch to deploying BERT models via TensorFlow Serving. Projects are organized into two top-level categories:

- **`ML/`** — Classical machine learning algorithms, feature engineering, and practice projects.
- **`DL/`** — Deep learning concepts, neural network architectures, NLP, computer vision, and model optimization.

Each topic is self-contained within its own folder, typically containing one or more Jupyter Notebooks with step-by-step implementation and explanations.

---

## Key Features

- **30+ ML/DL topics** implemented from first principles through production-ready pipelines.
- **End-to-end practice projects** including a Flask-based real estate price predictor and a FastAPI-powered intelligent log classification system.
- **NLP coverage** spanning Word2Vec, word embeddings, RNNs, and fine-tuned BERT models for text classification.
- **Computer vision** with CNNs on MNIST, CIFAR-10, flower datasets, and transfer learning using pre-trained models.
- **Model deployment** examples using TensorFlow Serving with Docker, Postman-tested REST endpoints, and model versioning.
- **Production ML techniques** including hyperparameter tuning, imbalanced data handling, data augmentation, dropout regularization, quantization, and distributed training.
- **Hybrid log classification pipeline** combining regex rules, a fine-tuned sentence-transformer classifier, and an LLM (Groq/DeepSeek) for unstructured legacy logs.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3 |
| **Deep Learning** | TensorFlow 2.x, Keras, TensorFlow Datasets, TensorFlow Hub, TensorFlow Text, TensorFlow Model Optimization |
| **Machine Learning** | scikit-learn, imbalanced-learn |
| **NLP** | Gensim (Word2Vec), Sentence Transformers (`all-MiniLM-L6-v2`), BERT via TF Hub |
| **LLM Integration** | Groq API (DeepSeek R1 Distill Llama 70B) |
| **Data & Visualization** | Pandas, NumPy, Matplotlib, Seaborn, SciPy |
| **Image Processing** | OpenCV (cv2), Pillow |
| **Web Frameworks** | Flask, FastAPI |
| **Model Serialization** | joblib, pickle |
| **Deployment** | TensorFlow Serving, Docker |
| **Notebooks** | Jupyter Notebook / JupyterLab |
| **Monitoring** | TensorBoard |

---

## Installation

### Prerequisites

- Python 3.9+
- `pip` or `conda`
- Jupyter Notebook or JupyterLab

### Clone the Repository

```bash
git clone https://github.com/vanix056/ML-DL-Projects-.git
cd ML-DL-Projects-
```

### Install Core Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn \
    imbalanced-learn gensim sentence-transformers opencv-python \
    Pillow scipy joblib flask fastapi uvicorn python-dotenv groq word2number
```

### Launch Jupyter

```bash
jupyter notebook
```

Navigate to any notebook within the `ML/` or `DL/` folders to get started.

---

## Usage

### Exploring Notebooks

Open any notebook to run ML/DL experiments interactively:

```bash
jupyter notebook DL/BERT_text_classification/BERT_email_classification.ipynb
jupyter notebook ML/Linear\ Regression/Linear_reg.ipynb
```

### Running Practice Project APIs

#### Real Estate Price Prediction (Flask)

```bash
cd "ML/Practice Projects/Real-estate--price-prediciton/server"
python server.py
```

The server exposes:
- `GET /get_location_names` — Returns available Bangalore locations.
- `POST /predict_home_price` — Accepts `total_sqft`, `location`, `bhk`, `bath`; returns estimated price.

Open `client/index.html` in a browser to use the frontend UI.

#### Log Classification System (FastAPI)

```bash
cd "ML/Practice Projects/Log-classification-system"
# Create a .env file with your GROQ_API_KEY
uvicorn server:app --reload
```

Upload a CSV file to `POST /classify/` with `source` and `log_message` columns. Returns a classified CSV with a `target_label` column.

### TensorFlow Serving (Docker)

```bash
docker run -it -v /path/to/DL/tf_serving:/tf_serving \
  -p 8601:8601 --entrypoint /bin/bash tensorflow/serving
tensorflow_model_server --rest_api_port=8601 \
  --model_name=email_model \
  --model_base_path=/tf_serving/saved_models/
```

---

## Project Structure

```
ML-DL-Projects-/
├── ML/
│   ├── Linear Regression/         # Simple, multivariate, gradient descent, one-hot encoding
│   ├── Logistic Regression/       # Binary classification examples
│   ├── Desicion Trees/            # Titanic survival prediction
│   ├── KNN/                       # K-Nearest Neighbors classifier
│   ├── Naive Bayes/               # Probabilistic text & tabular classification
│   ├── Random Forest/             # Ensemble classifier on flower dataset
│   ├── Support Vector Machines/   # SVM on digit recognition
│   ├── K-means_clustering/        # Unsupervised clustering
│   ├── Bagging/                   # Bootstrap aggregation ensemble
│   ├── PCA/                       # Principal Component Analysis
│   ├── Feature Engineering/       # Outlier detection: IQR, Z-score
│   ├── Hyperparameter Tuning/     # Grid search and cross-validation
│   └── Practice Projects/
│       ├── Real-estate--price-prediciton/
│       │   ├── model/             # Training notebook + Bangalore housing dataset
│       │   ├── server/            # Flask API + serialized model artifacts
│       │   └── client/            # HTML/CSS/JS frontend
│       └── Log-classification-system/
│           ├── classify.py        # Routing logic (regex → BERT → LLM)
│           ├── bert.py            # Sentence-transformer classifier
│           ├── llm.py             # Groq LLM integration
│           ├── log_regex.py       # Rule-based pattern matching
│           ├── server.py          # FastAPI endpoint
│           └── training/          # Training notebook + synthetic log dataset
│
└── DL/
    ├── nn_from_scratch/           # Logistic regression / NN implemented with NumPy
    ├── keras_fashion_mnist_neural_net/  # Keras sequential model on Fashion MNIST
    ├── digits_recognition/        # ANN on MNIST handwritten digits
    ├── CNN/                       # CNN for digit recognition
    ├── cnn_cifar10_small_image_classification/  # CNN on CIFAR-10
    ├── data_augmentation/         # CNN with augmentation on flower dataset
    ├── transfer_learning/         # Pre-trained CNN for image classification
    ├── activation_functions/      # ReLU, sigmoid, tanh visualization
    ├── loss/                      # Loss/cost function deep-dive
    ├── gradient_descent/          # Gradient descent theory and implementation
    ├── sgd_vs_gd_vs_miniBatch/    # Optimizer comparison
    ├── derivatives/               # Calculus foundations for backprop
    ├── matrix_math/               # Matrix operations for neural networks
    ├── dropout_layer/             # Dropout regularization in ANNs
    ├── precision_recall/          # Evaluation metrics walkthrough
    ├── imbalanced/                # SMOTE and class weighting techniques
    ├── chrun_prediction/          # ANN-based churn prediction
    ├── Bank Customer Chrun Prediction/   # ANN churn with bias mitigation
    ├── Telecom Churn Prediction/  # Telecom churn end-to-end pipeline
    ├── word_embedding/            # Supervised word embeddings
    ├── word2vec_gensim/           # Word2Vec with Gensim
    ├── NLP Stuff/                 # Word2Vec exploratory notebooks
    ├── BERT_intro/                # BERT architecture walkthrough
    ├── BERT_text_classification/  # Fine-tuned BERT for spam email detection
    ├── text_classification_rnn/   # RNN for text classification
    ├── tensorboard/               # TensorBoard integration and callbacks
    ├── tf_data_pipeline/          # tf.data API with prefetching and caching
    ├── prefatch/                  # Dataset prefetch and cache optimization
    ├── tf_serving/                # TensorFlow Serving with Docker
    ├── distributed_training/      # MirroredStrategy multi-GPU training
    ├── gpu_benchmarking/          # GPU vs CPU benchmarking on DGX
    └── quantization/              # Post-training model quantization
```

---

## Model Architecture

### Neural Network from Scratch
Binary logistic regression implemented using only NumPy — forward pass, loss computation, gradient calculation, and weight updates without any ML framework.

### Convolutional Neural Networks (CNNs)
Sequential CNN architectures with Conv2D, MaxPooling, BatchNormalization, and Dense layers applied to MNIST, CIFAR-10, and custom image datasets. Transfer learning experiments use pre-trained feature extractors from TensorFlow Hub.

### Recurrent Neural Networks (RNNs)
LSTM/GRU-based models for sequence classification on text data.

### BERT (Bidirectional Encoder Representations from Transformers)
Fine-tuned BERT models via TensorFlow Hub for email spam classification. The log classification practice project uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with a scikit-learn classifier head (`log_classifier.joblib`).

### ANN for Churn Prediction
Multi-layer fully connected networks with dropout regularization, trained on banking and telecom customer datasets with imbalanced class handling via SMOTE and class weights.

---

## Dataset

| Project | Dataset |
|---|---|
| Real Estate Price Prediction | Bangalore House Prices (`bengaluru_house_prices.csv`) |
| Log Classification | Synthetic log messages (`synthetic_logs.csv`) |
| Email Spam Classification | Spam SMS/email dataset (`spam.csv`) |
| Bank Churn Prediction | Bank customer tabular dataset |
| Telecom Churn | Telecom customer tabular dataset |
| CIFAR-10 / Fashion MNIST | TensorFlow Datasets (downloaded automatically) |
| Digits Recognition | MNIST (via `tensorflow_datasets`) |
| Flower Classification | TensorFlow flower dataset (data augmentation) |
| Titanic | Titanic survival dataset |

---

## Training

Most models are trained directly within Jupyter Notebooks. To re-train:

```bash
# Real Estate model
jupyter notebook "ML/Practice Projects/Real-estate--price-prediciton/model/banglore_home_prices_final.ipynb"

# Log Classifier (sentence-transformer + sklearn)
jupyter notebook "ML/Practice Projects/Log-classification-system/training/training.ipynb"

# BERT email classifier
jupyter notebook "DL/BERT_text_classification/BERT_email_classification.ipynb"
```

---

## Evaluation Metrics

| Model Type | Metrics Used |
|---|---|
| Regression | RMSE, MAE, R² Score |
| Binary Classification | Accuracy, Precision, Recall, F1-Score, AUC-ROC |
| Multi-class Classification | Accuracy, Confusion Matrix, Classification Report |
| Imbalanced Classification | Macro F1, ROC-AUC, Precision-Recall curves |
| NLP / Text Classification | Accuracy, F1-Score, Precision, Recall |

---

## Results

| Project | Outcome |
|---|---|
| Bangalore House Price Predictor | Linear regression model deployed behind Flask REST API with a browser UI |
| BERT Email Spam Classifier | Fine-tuned BERT achieving high classification accuracy on spam/ham emails; deployed via TF Serving |
| Log Classification System | Hybrid pipeline (regex + BERT + LLM) with configurable source-based routing; outputs labeled CSV |
| Bank/Telecom Churn ANN | Deep neural networks with SMOTE-corrected training demonstrating improved minority-class recall |
| CIFAR-10 CNN | Baseline and augmented CNNs with accuracy comparisons between GPU and CPU runtimes |

---

## API Routes

### Real Estate Price Prediction (Flask — `localhost:5000`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/get_location_names` | Returns list of all available Bangalore locations |
| `POST` | `/predict_home_price` | Predicts property price given area, location, BHK, and bathrooms |

**Request (POST `/predict_home_price`):**
```
Content-Type: application/x-www-form-urlencoded
Body: total_sqft=1500&location=Whitefield&bhk=3&bath=2
```

**Response:**
```json
{ "estimated_price": 85.25 }
```

---

### Log Classification System (FastAPI — `localhost:8000`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify/` | Accepts a CSV file; returns classified CSV with `target_label` column |

**Input CSV format:**
```
source,log_message
ModernCRM,IP 192.168.1.1 blocked due to potential attack
LegacyCRM,Invoice generation aborted for order ID 8910
```

**Classification routing logic:**
- `LegacyCRM` sources → Groq LLM (DeepSeek R1) → `Workflow Error` / `Deprecation Warning`
- All other sources → Regex rules first → fallback to BERT sentence-transformer classifier

---

## Configuration

### Log Classification System

Create a `.env` file in `ML/Practice Projects/Log-classification-system/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The Groq API key is required for LLM-based classification of `LegacyCRM` log sources.

### TF Serving

Model configuration files (`models.config.a/b/c`) in `DL/tf_serving/` control model versioning and label assignments. Refer to `DL/tf_serving/readme.md` for Docker run commands and Postman request examples.

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Keep each topic self-contained within its own folder.
3. Ensure notebooks are clean (outputs cleared before committing is recommended).
4. Use descriptive commit messages.
5. Submit a pull request with a clear description of the addition or fix.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Author

**Vanix056**

- GitHub: [@vanix056](https://github.com/vanix056)

---

*Built for learning, reference, and production experimentation across the ML/DL spectrum.*
