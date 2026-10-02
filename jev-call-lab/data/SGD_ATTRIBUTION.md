# Attribution: sgd_*.json

`sgd_train.json`, `sgd_dev.json`, `sgd_test.json` and `sgd_utterance_vocab.json` are derived from the
**Schema-Guided Dialogue (SGD) dataset** by Google Research:
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

> Rastogi, Zang, Sunkara, Gupta, Khaitan. "Towards Scalable Multi-domain Conversational Agents:
> The Schema-Guided Dialogue Dataset." AAAI 2020.

The original is licensed under **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/), and
these derived files are shared under the same license.

**What changed:** only single-service `Services_1..4` dialogues are kept. Each is reduced by
`reduce_sgd.py` to structural events (dialogue acts, slot names, service-call outcomes). Utterance text
and slot values are dropped, and timings are synthetic. `sgd_utterance_vocab.json` is the lowercase
word list of those utterances, used only by eval E2 to prove no word leaks into a Jev state.
