# LILAB_Intern-Project
Implementation and analysis of foundational NLP models (Word2Vec, ELMo, ...)  as part of LILAB internship project. Includes training, evaluation, and ablation studies.

## Projects

- [Word2Vec](word2vec_pytorch_server/): from-scratch PyTorch SGNS and CBOW training and evaluation.
- [ELMo](elmo_pytorch_server/): depth 1/2 biLM ablation, CoNLL-2003 NER, static/public embedding baselines, and a measured 24-hour run budget.

See each package's README for server setup and execution. ELMo's current test coverage and pending runtime checks are documented in [VALIDATION.md](elmo_pytorch_server/VALIDATION.md).
