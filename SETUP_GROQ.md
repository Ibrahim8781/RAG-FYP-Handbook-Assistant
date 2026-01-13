# Setup Instructions - Groq Integration

## Quick Setup Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Groq API Key
1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign up / Log in
3. Create a new API key
4. Copy the key

### 3. Configure Environment
Create a `.env` file in the project root:

```bash
# Copy the example file
copy .env.example .env

# Or on Linux/Mac
cp .env.example .env
```

Then edit `.env` and add your API key:
```
GROQ_API_KEY=your_actual_api_key_here
```

### 4. Run Ingestion (First Time Only)
```bash
python ingest.py
```

This creates the FAISS index from your handbook PDF.

### 5. Launch the App
```bash
# Web Interface (Recommended)
streamlit run app.py

# Or CLI
python ask.py
```

## What Changed?

### ✅ LLM Integration
- **Groq API** for natural, conversational answers
- **Improved prompts** for better page citations
- **Token usage tracking** for monitoring

### ✅ Better Answers
- Natural conversational tone (instead of bullet points)
- Proper sentence structure and flow
- Clear explanations with context
- Accurate page citations throughout

### ✅ Configuration
- Environment variables for API keys
- Configurable models
- Better error handling

## Testing Your Setup

Run a test query:
```bash
python ask.py
```

Try asking:
> "What are the required chapters of a Development FYP report?"

You should get a natural, conversational answer with page citations like:
> "Based on the handbook, a Development FYP report requires several key chapters (p. 15-17). The report should include..."

## Troubleshooting

### Error: "Groq API key not found"
- Make sure `.env` file exists in project root
- Check that `GROQ_API_KEY` is set correctly
- No quotes needed around the key in `.env`

### Error: "Import could not be resolved: llm_utils"
- Make sure `llm_utils.py` is in the same directory
- Try running from the project root directory

### Error: "Required files not found"
- Run `python ingest.py` first to create the FAISS index
- Make sure `FYP-Handbook-2023.pdf` is in the project directory

### Slow responses
- First request initializes models (can take 5-10 seconds)
- Subsequent requests are faster (~1-2 seconds)
- Consider using `llama-3.1-8b-instant` for faster responses

## Model Options

Edit `.env` to change the model:

```bash
# Best quality (default)
GROQ_MODEL=llama-3.1-70b-versatile

# Faster responses
GROQ_MODEL=llama-3.1-8b-instant

# Alternative
GROQ_MODEL=mixtral-8x7b-32768
```

## Rate Limits

Groq free tier includes:
- 30 requests per minute
- 14,400 tokens per minute

This is plenty for typical usage!

## Cost

Groq is **FREE** for the models we're using! 🎉

## Need Help?

Check:
1. `.env` file has correct API key
2. Internet connection is working
3. `ingest.py` has been run successfully
4. All dependencies are installed

Still issues? Check the Groq Console for API status.
