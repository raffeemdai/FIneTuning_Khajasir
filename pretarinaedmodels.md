```markdown
# LLM Models: Pretrained (Base) vs Instruct — Ollama & Hugging Face

## 1. Pretrained (Base) Models

1. Llama 3.1 8B (base) — Meta
   - Hugging Face: https://huggingface.co/meta-llama/Llama-3.1-8B
   - Ollama: https://ollama.com/library/llama3.1 (tag: llama3.1:8b-text)

2. Llama 3.2 3B (base) — Meta
   - Hugging Face: https://huggingface.co/meta-llama/Llama-3.2-3B
   - Ollama: https://ollama.com/library/llama3.2 (tag: llama3.2:3b-text)

3. Llama 2 7B (base) — Meta
   - Hugging Face: https://huggingface.co/meta-llama/Llama-2-7b-hf
   - Ollama: https://ollama.com/library/llama2 (tag: llama2:7b-text)

4. Mistral 7B v0.3 (base) — Mistral AI
   - Hugging Face: https://huggingface.co/mistralai/Mistral-7B-v0.3
   - Ollama: https://ollama.com/library/mistral (tag: mistral:7b-text)

5. Gemma 2 9B (base) — Google
   - Hugging Face: https://huggingface.co/google/gemma-2-9b
   - Ollama: https://ollama.com/library/gemma2

6. Gemma 3 4B (base / "pt") — Google
   - Hugging Face: https://huggingface.co/google/gemma-3-4b-pt
   - Ollama: https://ollama.com/library/gemma3

7. Qwen2.5 7B (base) — Alibaba
   - Hugging Face: https://huggingface.co/Qwen/Qwen2.5-7B
   - Ollama: https://ollama.com/library/qwen2.5

8. Falcon 7B (base) — TII
   - Hugging Face: https://huggingface.co/tiiuae/falcon-7b
   - Ollama: https://ollama.com/library/falcon

9. StarCoder2 15B (base, code) — BigCode
   - Hugging Face: https://huggingface.co/bigcode/starcoder2-15b
   - Ollama: https://ollama.com/library/starcoder2

10. TinyLlama 1.1B (base) — TinyLlama Project
    - Hugging Face: https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    - Ollama: https://ollama.com/library/tinyllama


## 2. Instruct (Instruction-Tuned / Chat) Models

1. Llama 3.1 8B Instruct — Meta
   - Hugging Face: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - Ollama: https://ollama.com/library/llama3.1

2. Llama 3.3 70B Instruct — Meta
   - Hugging Face: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
   - Ollama: https://ollama.com/library/llama3.3

3. Mistral 7B Instruct v0.3 — Mistral AI
   - Hugging Face: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
   - Ollama: https://ollama.com/library/mistral

4. Gemma 2 9B IT — Google
   - Hugging Face: https://huggingface.co/google/gemma-2-9b-it
   - Ollama: https://ollama.com/library/gemma2

5. Gemma 3 27B IT — Google
   - Hugging Face: https://huggingface.co/google/gemma-3-27b-it
   - Ollama: https://ollama.com/library/gemma3

6. Qwen2.5 7B Instruct — Alibaba
   - Hugging Face: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
   - Ollama: https://ollama.com/library/qwen2.5

7. Qwen2.5-Coder 7B Instruct — Alibaba
   - Hugging Face: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
   - Ollama: https://ollama.com/library/qwen2.5-coder

8. Phi-4 (instruct) — Microsoft
   - Hugging Face: https://huggingface.co/microsoft/phi-4
   - Ollama: https://ollama.com/library/phi4

9. DeepSeek-R1 (reasoning/instruct) — DeepSeek AI
   - Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-R1
   - Ollama: https://ollama.com/library/deepseek-r1

10. Command R (chat) — Cohere
    - Hugging Face: https://huggingface.co/CohereForAI/c4ai-command-r-v01
    - Ollama: https://ollama.com/library/command-r


## Notes
- Meta's Llama-family repos on Hugging Face are gated — request access/accept the license before downloading.
- On Ollama, check the model's tag list page before pulling — exact base/instruct/quantization tag names vary by family and change over time.
- Base models continue text; they don't reliably follow instructions unless fine-tuned or heavily few-shot prompted.
```
