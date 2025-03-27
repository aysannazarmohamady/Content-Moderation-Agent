# Content Moderation Agent

## Project Overview

The Content Moderation Agent is an intelligent system designed to detect toxic content in online discussions using machine learning and natural language processing. Built upon the [Toxicity Detection with Context](https://www.kaggle.com/datasets/adilshamim8/toxicity-detection-context) dataset from Kaggle, it can identify toxic comments both with and without considering their conversational context.

This agent serves as a practical implementation of AI agents for content moderation, demonstrating how context-aware systems can better understand and classify potentially harmful content in online platforms.

## Key Features

- **Context-Aware Toxicity Detection**: Evaluates content toxicity with or without surrounding context
- **Multiple Classification Models**: Supports TF-IDF and BERT-based models with comparative analysis
- **Suggestion Generation**: Provides recommendations for improving toxic content
- **API and CLI Interfaces**: Flexible deployment options for different use cases
- **Comprehensive Evaluation**: Detailed metrics to measure performance across different models
- **Feedback Mechanism**: System for continuous improvement through user feedback


## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/content-moderation-agent.git
cd content-moderation-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

```bash
# Create data directories
mkdir -p data/raw data/processed

# Download data from Kaggle
pip install kaggle
python -c "import kagglehub; kagglehub.dataset_download('adilshamim8/toxicity-detection-context', path='data/raw')"

# Alternatively, download manually from https://www.kaggle.com/datasets/adilshamim8/toxicity-detection-context
# and place the files in the data/raw directory
```

## Usage

### Training a Model

```bash
# Train a TF-IDF model without context
python -m src.models.train --data-path data/raw --model-type tfidf

# Train a BERT model with context
python -m src.models.train --data-path data/raw --model-type bert --with-context
```

### Command Line Interface

```bash
# Run the CLI with a pre-trained model
python app.py --model-path models/tfidf_no_context_model.pkl

# Run with context-aware BERT model
python app.py --model-path models/bert_with_context_model.pkl --with-context
```

### API Server

```bash
# Start the API server
python app.py --server --port 5000 --model-path models/tfidf_no_context_model.pkl
```

### API Endpoints

- `GET /health`: Health check endpoint
- `POST /moderate`: Moderate a single text
  ```json
  {
    "text": "Text to moderate",
    "context": "Optional context for the text"
  }
  ```
- `POST /batch_moderate`: Moderate multiple texts
  ```json
  {
    "texts": ["Text 1", "Text 2"],
    "contexts": ["Context 1", "Context 2"]
  }
  ```
- `POST /feedback`: Provide feedback for misclassification
  ```json
  {
    "text": "Misclassified text",
    "actual_toxic": true
  }
  ```

### Python API

```python
from src.agent.moderator import ContentModerationAgent

# Initialize agent
agent = ContentModerationAgent(
    model_path="models/bert_with_context_model.pkl",
    context_aware=True
)

# Moderate a single text
text = "This is a sample text to moderate."
context = "Previous discussion context."
result = agent.moderate(text, context=context)

print(f"Is Toxic: {result['is_toxic']}")
print(f"Confidence: {result['confidence']:.2f}")
if result['is_toxic'] and result['suggestions']:
    print("Suggestions:")
    for suggestion in result['suggestions']:
        print(f"- {suggestion}")
```

## Model Performance

The project includes comprehensive evaluation of different model configurations. Here are sample metrics from our tests:

| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| TF-IDF without context | 0.91 | 0.86 | 0.82 | 0.84 | 0.93 |
| TF-IDF with context | 0.92 | 0.88 | 0.83 | 0.85 | 0.94 |
| BERT without context | 0.94 | 0.91 | 0.87 | 0.89 | 0.96 |
| BERT with context | 0.95 | 0.93 | 0.89 | 0.91 | 0.97 |

Context-aware models consistently outperform their context-free counterparts, demonstrating the value of contextual information in toxicity detection.

## Research Insights

Our implementation explores the key findings from the paper "Toxicity Detection: Does Context Really Matter?" and demonstrates that:

1. In approximately 5% of cases, context significantly changes the perceived toxicity of a comment
2. Context can both amplify toxicity (making otherwise innocent comments seem toxic) and mitigate it (providing justification for seemingly toxic language)
3. Context-aware models show improved performance across all metrics, particularly in edge cases

## Applications

The Content Moderation Agent can be integrated into various platforms:

- **Social Media Platforms**: Automatic detection of toxic comments
- **Online Forums**: Pre-screening of user-generated content
- **Customer Service**: Monitoring of interactions for toxic language
- **Educational Platforms**: Ensuring appropriate communications
- **Content Management Systems**: Flagging potentially harmful content for human review

## Limitations

- The current implementation is primarily focused on English text
- Sarcasm and subtle forms of toxicity remain challenging to detect
- Cultural and contextual nuances may affect toxicity perception
- The model reflects biases present in the training data

## Future Work

- Multilingual support for toxicity detection
- Integration with more sophisticated LLMs for better context understanding
- Explainable AI features to help users understand why content was flagged
- Active learning implementation for continuous model improvement
- User interface for non-technical moderators

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- This project uses the [Toxicity Detection with Context](https://www.kaggle.com/datasets/adilshamim8/toxicity-detection-context) dataset from Kaggle
- The research is based on the paper "Toxicity Detection: Does Context Really Matter?" (ACL 2020)
- Built with [LangChain](https://python.langchain.com/docs/get_started/introduction) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## Contact

- Aysan Nazarmohammadi - [aysan.nazarmohamady@yahoo.com](mailto:aysan.nazarmohamady@yahoo.com)
- Project Link: [https://github.com/aysannazarmohamady/Content-Moderation-Agent](https://github.com/aysannazarmohamady/Content-Moderation-Agent)
