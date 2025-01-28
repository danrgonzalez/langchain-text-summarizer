# LangChain Text Summarizer

A simple implementation of a text summarization chain using LangChain. This project demonstrates how to build a basic pipeline that processes text input and returns a concise summary using LangChain components.

## Features

- Text splitting for handling long documents
- Configurable summarization using OpenAI's language models
- Simple API for text processing
- Example implementation with sample text

## Setup

1. Clone the repository:
```bash
git clone https://github.com/danrgonzalez/langchain-text-summarizer.git
cd langchain-text-summarizer
```

2. Install dependencies:
```bash
pip install langchain openai
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

```python
from summarizer import summarize_text

text = """Your long text here..."""
summary = summarize_text(text)
print(summary)
```

## Configuration

You can modify the following parameters in the code:
- `chunk_size`: Size of text chunks for processing (default: 2000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `temperature`: LLM temperature setting (default: 0)

## License

MIT License