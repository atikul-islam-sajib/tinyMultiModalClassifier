## **Multi-Modal Classifier: Vision Transformer & Transformer Encoder Block**  

This repository provides a **lightweight multi-modal classification model** that seamlessly integrates **Vision Transformer (ViT)** for image feature extraction and a **Transformer Encoder Block** for text processing. By leveraging **self-attention mechanisms**, this model efficiently captures both spatial relationships in images and sequential dependencies in text.  

The extracted features from both modalities are **concatenated** and passed through a **fully connected classifier** to generate the final prediction. This architecture is designed to handle **multi-modal data** efficiently and can be used for tasks such as **image-text classification, multi-modal sentiment analysis, and document classification**.

---

## **How It Works**  

The multi-modal classification model follows a structured pipeline:

1. **Image Feature Extraction (ViT)**  
   - The **Vision Transformer (ViT)** is used to process images by dividing them into patches.
   - Each patch is **linearly embedded**, followed by **positional encoding**.
   - The sequence of patches is then passed through **self-attention layers** to extract meaningful features.

2. **Text Feature Extraction (Transformer Encoder Block)**  
   - The input text is tokenized and converted into embeddings.
   - A **Transformer Encoder Block** applies self-attention and layer normalization.
   - The processed text embeddings capture the semantic relationships between words.

3. **Feature Fusion & Classification**  
   - The **image and text features** are concatenated.
   - The concatenated representation is passed through a **classifier**.
   - The model outputs the **final class prediction**.

---

## **Model Architecture**

```plaintext
    +----------------+       +-----------------------------+       +--------------------+
    |    Image       | ----> | Vision Transformer (ViT)    | ----> | Extracted Features |
    +----------------+       +-----------------------------+       +--------------------+
                                      ||
                                      ||
    +----------------+       +-----------------------------+       +--------------------+
    |    Text        | ----> | Transformer Encoder Block   | ----> | Extracted Features |
    +----------------+       +-----------------------------+       +--------------------+
                                      ||
                                      ||
                            +-----------------------------+
                            | Feature Concatenation       |
                            +-----------------------------+
                                      ||
                                      ||
                            +-----------------------------+
                            | Fully Connected Classifier  |
                            +-----------------------------+
                                      ||
                                      ||
                            +-----------------------------+
                            |      Final Prediction       |
                            +-----------------------------+
```

---

## **Visual Representation**

### **Vision Transformer (ViT)**
![Vision Transformer](https://raw.githubusercontent.com/google-research/vision_transformer/main/docs/vit.png)  
*ViT splits the input image into fixed-size patches and processes them like a sequence.*

---

### **Transformer Encoder Block**
![Transformer Encoder](https://upload.wikimedia.org/wikipedia/commons/3/3f/Transformer_Encoder.png)  
*A Transformer Encoder applies **self-attention** and **layer normalization** to process text embeddings.*

---

## **Key Features**
✔ **Multi-Modal Learning** – Combines image and text features for robust classification.  
✔ **Self-Attention Mechanism** – Captures spatial dependencies in images and semantic relationships in text.  
✔ **Feature Fusion** – Effectively merges different modalities to enhance accuracy.  
✔ **Scalability** – Can be extended to handle more modalities (e.g., audio, video).  

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

## **Training and Testing Commands**  

| **Mode**      | **Command** | **Description** | **Customizable Parameters** | **Example Usage** |
|--------------|------------|----------------|-----------------------------|-------------------|
| **Train**    | `python src/cli.py --train` | Loads dataset, trains model, and saves checkpoints/logs. | `--epochs`: Number of training epochs.<br> `--lr`: Learning rate.<br> `--batch_size`: Batch size for training.<br> `--device`: Specify training device (`cuda`, `mps`, `cpu`). | `python src/cli.py --train --epochs 150 --lr 3e-4 --batch_size 16 --device cuda` |
| **Test**     | `python src/cli.py --test` | Loads the best model, evaluates on the test set, computes accuracy, precision, recall, F1-score, and generates a confusion matrix. | `--batch_size`: Batch size for testing.<br> `--device`: Specify testing device (`cuda`, `mps`, `cpu`). | `python src/cli.py --test --batch_size 32 --device cuda` |

```
---
```

## **Evaluation Metrics**  

| **Metric**            | **Description**                                                      | **Example Value** |
|----------------------|----------------------------------------------------------------------|------------------|
| **Accuracy**        | Measures overall correctness of the model.                           | `92.5%`         |
| **Precision**       | Fraction of relevant instances among the retrieved instances.        | `89.3%`         |
| **Recall**          | Measures completeness by identifying true positive cases.            | `90.1%`         |
| **F1-Score**        | Harmonic mean of precision and recall.                               | `89.7%`         |
| **Confusion Matrix** | Breakdown of correct vs incorrect classifications per class.        | `[[345, 12, 8], [10, 290, 15], [5, 18, 325]]` |
```
---

## **Contributing**  

Contributions are welcome! Feel free to submit **pull requests** or open **issues** to improve the project.  

---

## **License**  

This project is released under the **MIT License**. You can use, modify, and distribute it freely for research and educational purposes.  