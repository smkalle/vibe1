# ja-support-asr: Japanese customer-support ASR on EmbeddingGemma 2

One notebook, `japanese_customer_support_asr.ipynb`, that turns `google/embeddinggemma-2` into a fast
non-autoregressive Japanese speech-to-text model for contact-center audio. The pretrained audio encoder
stays frozen and only a small CTC "reader" (~22M params) is trained on top of it. The approach comes from
[qnguyen3/gemma-asr](https://github.com/qnguyen3/gemma-asr), which did this for Vietnamese.

What's inside (21 parts):

- Japanese text normalization: NFKC, punctuation, spacing, digits → kanji readings, optional filler removal. It has tests.
- A customer-support domain kit: glossary, keigo phrases, intents, escalation terms.
- Data prep for Common Voice ja, FLEURS ja, JSUT, ReazonSpeech and your own call recordings. Calls are split by `call_id`, so no call lands in both train and test. The output is parquet shards in the same format as gemma-asr.
- A CTC vocabulary built from Gemma tokens, with a check that glossary terms are covered.
- Telephony augmentation: 8 kHz narrowband, G.711 μ-law, noise.
- Training that picks the best checkpoint by CER. Checkpoints load in the original repo's scripts.
- Evaluation: CER, kana-CER, SER, top confusions, product-name recall, number accuracy, and scores on phone-degraded audio.
- Long-audio inference with timestamps, and stereo call transcription (operator and customer on separate channels).
- Post-processing: numbers back to digits, glossary correction, PII masking.
- Rule-based call analytics: intents, escalation risk, keigo compliance, talk ratio.
- Stage-2 domain adaptation, multi-GPU training with the original `torchrun` scripts, and a LitServe endpoint with an MCP bridge.

## Run

```bash
pip install torch "transformers>=5.19" "huggingface_hub[hf_xet]" datasets pyarrow soundfile soxr numpy jiwer pandas matplotlib
hf auth login            # embeddinggemma-2 is gated: accept the license on Hugging Face first
jupyter lab japanese_customer_support_asr.ipynb
```

If no GPU is found, the notebook switches to **DRY_RUN** mode. It replaces the encoder with a tiny random one,
the tokenizer with a character tokenizer, and the datasets with synthetic audio, so every cell runs on a CPU in
about 3 minutes. Use `JA_ASR_DRY_RUN=0` (and a GPU) for real training.

The notebook writes a reusable package (`ja_asr/`), its data and checkpoints (`ja_asr_work/`) and the serving
files (`serving/`) next to itself. All three are git-ignored.
