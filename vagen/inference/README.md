# Use Openai API
```
export OPENAI_API_KEY=<sk-your_openai_api_key>
```

# Use Claude API
```
pip install anthropic
export ANTHROPIC_API_KEY=<your_claude_api_key>
```
# Use Gemini API
```
pip install google.generativeai
export GOOGLE_API_KEY=<your_gemini_api_key>
```
## Test your inference pipeline
```
python vagen/server/server.py
./scripts/debug/debug_api_frozenlake/run.sh
```