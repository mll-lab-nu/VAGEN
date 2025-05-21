## Create data from raw_data
```
python create_dataset.py --csv raw_data/data.csv --old_templates raw_data/templates/old_prompts.yaml --new_templates raw_data/templates/new_prompts.yaml --output_dir data
```

## generate api response
### Gemini
```
python gemini_analysis.py --samples data/old_samples.jsonl --output analysis/gemini/gemini_old_results.jsonl
python gemini_analysis.py --samples data/new_samples.jsonl --output analysis/gemini/gemini_new_results.jsonl
```
### GPT 4.1 nano
```
python gpt_analysis.py --samples data/old_samples.jsonl --output analysis/gpt/gpt_old_nano_results.jsonl --model gpt-4.1-nano --model_name gpt-4.1-nano
python gpt_analysis.py --samples data/new_samples.jsonl --output analysis/gpt/gpt_new_nano_results.jsonl --model gpt-4.1-nano --model_name gpt-4.1-nano
```


## analysis result
```
python state_analysis.py --input analysis/gemini/gemini_old_results.jsonl --output analysis/gemini/old_accuracy.jsonl
python state_analysis.py --input analysis/gemini/gemini_new_results.jsonl --output analysis/gemini/new_accuracy.jsonl
```
```
python state_analysis.py --input analysis/gpt/gpt_old_nano_results.jsonl --output analysis/gpt/old_nano_accuracy.jsonl
python state_analysis.py --input analysis/gpt/gpt_new_nano_results.jsonl --output analysis/gpt/new_nano_accuracy.jsonl
```