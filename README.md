# German-English Translation using Transformers

This repository contains the implementation of a machine translation model based on the Transformer architecture to translate text between German and English. The project is part of a mini-project for the 6th-semester Deep Learning Laboratory course at Mepco Schlenk Engineering College, Sivakasi.

## Project Overview

The project aims to leverage the power of sequence-to-sequence (Seq2Seq) neural network models with attention mechanisms for accurate and efficient German-English translation. The implementation focuses on:

- Preprocessing linguistic data.
- Training a Transformer-based model.
- Evaluating the model using BLEU scores and visualizing attention mechanisms.
- Utilizing PyTorch and TorchText for seamless integration and scalability.

## Features

- **Transformer Architecture**: Implements encoder-decoder models with multi-head attention mechanisms and feed-forward layers.
- **Dataset**: Utilizes the Multi30K dataset, containing parallel German-English sentences.
- **Attention Mechanism**: Visualizes attention weights for linguistic insights.
- **Evaluation**: Measures translation quality using BLEU scores.

## Directory Structure

```plaintext
.
├── README.md         # Project overview and instructions
├── src/              # Source code for the translation models
│   ├── model.py      # Transformer model implementation
│   ├── train.py      # Training and evaluation scripts
│   └── utils.py      # Utility functions for preprocessing and visualization
├── Figures/            # Multi30K dataset files
│   ├── results/
│   ├── Proposed system/
│   ├── Sample Dataset/            # Visualizations and results (e.g., attention maps, BLEU scores)
└── requirements.txt  # List of dependencies
```

## Getting Started

### Prerequisites

Ensure the following are installed:

- Python 3.9+
- PyTorch 1.9.0
- TorchText 0.10.0
- NumPy 1.20.3
- Matplotlib 3.8.0
- spaCy (with German and English models: `de_core_news_sm` and `en_core_web_sm`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/German-English-Translation-Transformers.git
   cd German-English-Translation-Transformers
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:

   The Multi30K dataset is automatically downloaded and split into train, validation, and test sets using TorchText utilities.

4. Download spaCy language models:

   ```bash
   python -m spacy download de_core_news_sm
   python -m spacy download en_core_web_sm
   ```

## Usage

### Training the Model

Run the training script:

```bash
python src/train.py
```

### Translation

Translate a sample German sentence:

```python
from src.model import translate_sentence
sentence = "Das ist ein Beispiel."
translation, attention = translate_sentence(sentence, src_field, trg_field, model, device)
print("Translation:", " ".join(translation))
```

### Visualize Attention

Visualize the attention weights for a sample translation:

```python
from src.utils import display_attention
display_attention(sentence, translation, attention)
```
## Model Architecture
! [Model Architecture](Figures/Proposed System/System Architecture.png)

## Results

- **BLEU Score**: Achieved a BLEU score of X.X% (to be updated with actual results).
- **Attention Visualizations**: Attention maps highlight the alignment between source and target sentences.
- **Model Performance**: Loss and accuracy graphs available in the `results/` directory.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for enhancements or bug fixes.

## Authors

- [Manoj S](https://github.com/Jonam-2004)
- [Prasanna Venkatesh S](https://github.com/anna123venkat)
- [Timothy Florian LV](https://github.com/TimothyFlorian)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

1. [Comprehensive Guide to Attention Mechanisms](https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/)
2. [Multi30K Dataset](https://www.statmt.org/wmt16/multimodal-task.html)
3. [AWS Machine Translation](https://aws.amazon.com/what-is/machine-translation/)
4. [Google Research on Machine Translation](https://research.google/research-areas/machine-translation/)
5. [Papers with Code: Multi30K](https://paperswithcode.com/dataset/multi30k)
