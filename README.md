# Product Description Generator

## Overview

The **Product Description Generator** is a deep learning-based solution that automatically generates descriptions for product names using a transformer-based model. It leverages **PyTorch** and **Hugging Face's BERT tokenizer** to process input and generate high-quality product descriptions.

## Features

- Generates a descriptive text based on a given product name.
- Supports **batch processing** for multiple product names at once.
- Uses **top-k sampling** and **temperature scaling** for diverse and creative text generation.
- Utilizes a transformer-based sequence-to-sequence model.
- Runs efficiently on **CPU and GPU**.

## Prerequisites

Before using this project, make sure you have the following installed:

- Python 3.7+
- PyTorch
- Transformers (Hugging Face Library)
- NumPy

### Install dependencies

```bash
pip install torch transformers numpy
```

## Model Architecture

The model is based on a **Transformer Encoder-Decoder** architecture. It consists of:

- **Encoder:** Processes the product name and extracts contextual embeddings.
- **Decoder:** Generates the product description one token at a time.

## Usage

### 1. Initialize Tokenizer and Model

```python
from transformers import BertTokenizer
from product_description_generator import ProductDescriptionGenerator

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize generator
generator = ProductDescriptionGenerator(
    model_path='best_model.pt',
    tokenizer=tokenizer
)
```

### 2. Generate Description for a Single Product

```python
product_name = "Wireless Bluetooth Headphones"
description = generator.generate_description(
    product_name,
    max_length=128,
    temperature=0.7,
    top_k=50
)
print(f"Product: {product_name}")
print(f"Generated Description: {description}")
```

### 3. Generate Descriptions in Batch

```python
product_names = [
    "Smart Watch with Heart Rate Monitor",
    "Ultra HD 4K TV 55-inch",
    "Professional Coffee Maker"
]

descriptions = generator.generate_batch_descriptions(
    product_names,
    batch_size=2,
    max_length=128,
    temperature=0.7,
    top_k=50
)

for product, desc in zip(product_names, descriptions):
    print(f"\nProduct: {product}")
    print(f"Generated Description: {desc}")
```

## Model Training (Optional)

If you need to train the model from scratch or fine-tune it, follow these steps:

1. Prepare a dataset of product names and descriptions.
2. Train a transformer-based encoder-decoder model.
3. Save the trained model as `best_model.pt`.

## Customization

- You can **adjust hyperparameters** like `max_length`, `temperature`, and `top_k` to control the creativity and diversity of generated descriptions.
- The `ProductTransformer` model can be replaced with a more advanced architecture if needed.

## Limitations

- The generated descriptions might require manual refinement.
- Performance depends on the quality of the training data.
- Some descriptions may be repetitive or generic.

## License

This project is open-source and available under the **MIT License**.

## Author

Developed by **[Your Name]**. Feel free to contribute or reach out for improvements!
