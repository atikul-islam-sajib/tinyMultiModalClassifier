# **TinyMultiModalClassifier**  

A lightweight **multi-modal classifier** that integrates **Vision Transformer (ViT)** for image processing and a **Transformer Encoder Block** for text processing. The extracted features are concatenated and passed through a **classifier** for final classification.

---

## **Model Architecture**  

```
+--------------------+       +-------------------------+       +---------------------+
|   Input Image     | ----> |     Vision Transformer  | ----> |                     |
|   (RGB Image)     |       |     (ViT Backbone)      |       |                     |
+--------------------+       +-------------------------+       |                     |
                                                               |   Concatenation     |
+--------------------+       +-------------------------+       |      Layer         |
|   Input Text      | ----> |  Transformer Encoder    | ----> |                     |
|   (Tokenized)     |       |  (Text Feature Extractor)|       |                     |
+--------------------+       +-------------------------+       +---------------------+
                                                               |
                                                               v
                                                   +----------------------+
                                                   |   Classifier Layer    |
                                                   |  (Fully Connected)    |
                                                   +----------------------+
                                                               |
                                                               v
                                                   +----------------------+
                                                   |      Prediction       |
                                                   |    (Class Labels)     |
                                                   +----------------------+
```

---

## **Features**  

✔️ **Multi-Modal Learning** - Combines both image and text modalities.  
✔️ **Vision Transformer (ViT)** - Extracts high-level image features.  
✔️ **Transformer Encoder Block** - Captures textual dependencies.  
✔️ **Feature Fusion** - Merges extracted features before classification.  
✔️ **Efficient Training** - Supports **multi-GPU** training with optimized data pipelines.  

---


## **Project Structure**  

```
tinyMultiModalClassifier/
│
├── src/
│   ├── cli.py                 # Command-line interface for training & testing
│   ├── model.py               # Model implementation
│   ├── dataloader.py          # Data loading pipeline
│   ├── trainer.py             # Training loop
│   ├── utils.py               # Utility functions
    .....
    .....
    .....
│
├── config.yml                 # Configuration file
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
```

## **Installation**  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/atikul-islam-sajib/tinyMultiModalClassifier.git
   cd tinyMultiModalClassifier
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```


3.  **Configuration (`config.yml`)**  

Modify `config.yml` to adjust model hyperparameters, dataset paths, and training options.  

### **Full Configuration File**  

```yaml
artifacts:
    raw_data_path: "./data/raw/"
    processed_data_path: "./data/processed/"
    checkpoints: "./artifacts/checkpoints/"
    train_models: "./artifacts/checkpoints/train_models/"
    best_model: "./artifacts/checkpoints/best_model/"
    files: "./artifacts/files/"
    metrics: "./artifacts/metrics/"
    train_images: "./artifacts/outputs/train_images/"
    val_images: "./artifacts/outputs/val_images/"
    test_image: "./artifacts/outputs/test_image/"

patchEmbeddings:
    channels: 3                    # Number of image channels (RGB = 3)
    patch_size: 16                  # Patch size for Vision Transformer
    image_size: 224                 # Input image size
    dimension: 256                   # Feature embedding size

transfomerEncoderBlock:
    nheads: 8                       # Number of attention heads
    activation: "leaky"              # Activation function type
    dropout: 0.1                     # Dropout rate for regularization
    num_encoder_layers: 6            # Number of transformer encoder layers
    dimension_feedforward: 4096      # Feed-forward network dimension
    layer_norm_eps: 1e-5             # Epsilon value for layer normalization

dataloader:
    dataset: "./data/raw/image_dataset.zip"  # Path to dataset
    batch_size: 32                           # Number of samples per batch
    split_size: 0.30                         # Train-validation split ratio
    
trainer:
    model: None                 # Placeholder for model type
    epochs: 200                  # Number of training epochs
    lr: 2e-4                     # Learning rate
    weight_decay: 5e-4           # Weight decay for regularization
    beta1: 0.9                   # Adam optimizer beta1
    beta2: 0.999                 # Adam optimizer beta2
    momentum: 0.95               # SGD optimizer momentum
    lr_scheduler: False          # Whether to use learning rate scheduler
    step_size: 20                # Step size for learning rate scheduler
    gamma: 0.1                   # Learning rate decay factor
    l1_regularization: False     # Whether to apply L1 regularization
    l2_regularization: False     # Whether to apply L2 regularization
    l1_lambda: 0.0               # L1 regularization weight
    l2_lambda: 0.0               # L2 regularization weight
    adam: True                   # Use Adam optimizer
    SGD: False                   # Use SGD optimizer
    mlflow: False                # Enable MLflow logging
    verbose: False               # Enable verbose output
    device: "cuda"               # Training device (cuda/mps/cpu)

tester:
    model: "best"                # Load best trained model for testing
    device: "cuda"               # Device to run testing
    plot_images: True            # Enable visualization of predictions
``` 

---

## **Training the Model**  

To start training, run:  
```bash
python src/cli.py --train
```

This will:  
✔️ Load the dataset  
✔️ Train the model  
✔️ Save checkpoints and logs  

### **Command Breakdown**
- `--train`: Enables training mode.  
- Reads hyperparameters from `config.yml`.  
- Saves model checkpoints to `artifacts/checkpoints/`.  
- Logs accuracy, loss, and confusion matrix.  

---

## **Testing the Model**  

To evaluate the model on the test set, run:  
```bash
python src/cli.py --test
```

This will:  
✔️ Compute **Accuracy, Precision, Recall, F1-Score**  
✔️ Generate a **Confusion Matrix**  
✔️ Save the results  

### **Command Breakdown**
- `--test`: Enables testing mode.  
- Loads the **best trained model** from `artifacts/checkpoints/best_model/`.  
- Runs inference on the test dataset.  
- Saves performance metrics in `artifacts/metrics/`.  

---

## **Evaluation Metrics**  

- **Accuracy** - Measures overall correctness of the model.  
- **Precision** - Fraction of relevant instances among the retrieved instances.  
- **Recall** - Measures completeness by identifying true positive cases.  
- **F1-Score** - Harmonic mean of precision and recall.  
- **Confusion Matrix** - Provides a breakdown of correct vs incorrect classifications.  

### **Example Evaluation Output**  
```json
{
    "Accuracy": 92.5,
    "Precision": 89.3,
    "Recall": 90.1,
    "F1-Score": 89.7,
    "Confusion Matrix": [
        [345, 12, 8],
        [10, 290, 15],
        [5, 18, 325]
    ]
}
```

---

## **Contributing**  

Contributions are welcome! Feel free to submit **pull requests** or open **issues** to improve the project.  

---

## **License**  

This project is released under the **MIT License**. You can use, modify, and distribute it freely for research and educational purposes.  