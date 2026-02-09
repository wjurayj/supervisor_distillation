
# On-policy distillation teaches recursive language models edge-cloud collaboration.
# Self-distilling recursive language models

# Deep research system

Supervisor model (under certain conditions) distills behavior into worker model

Different settings:
- Few-shot demonstrations pointing to what information to relay in summary
- Many-shot demonstrations for gradient-based imitation learning
- On-policy distillation (maybe after SFT stage?), highlighting large/small
- RL with compression objective (reconstruction + succinctness)

## Related reading:

### Large-small model collaboration principles
- [Minions: edge-cloud collaboration](https://openreview.net/pdf?id=qGDlzt3dKz), and [Recursive Language Models](https://arxiv.org/pdf/2512.24601) adding REPL
- [Mutual information for compressor-predictors](https://arxiv.org/pdf/2512.21720)
- [Resource-Adaptive Subtask Routing for Edge-Cloud Inference](https://www.arxiv.org/pdf/2512.22137v3)

### Gradient updates to LLM based on input data
- [Self-adapting language models](https://arxiv.org/pdf/2506.10943), c.f. [LoRA without regret](https://thinkingmachines.ai/blog/lora/)
- [Self-Improving Model Collaboration Systems](https://arxiv.org/pdf/2602.05182)
    - on-policy distillation is preferable to fixed set of SFT demonstrations

### On-policy distillation
- [On-policy distillation](https://arxiv.org/abs/2306.13649), [Thinky Blog](https://thinkingmachines.ai/blog/on-policy-distillation/), and [Qwen3 Report](https://arxiv.org/pdf/2505.09388)
- [Universal Logit Loss for cross-tokenizer distillation](https://arxiv.org/pdf/2402.12030) and [TRL implementation called "GOLD"](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
- [RL via self-distillation](https://arxiv.org/pdf/2601.20802), and accompanying work [self-distillation for continual learning](https://arxiv.org/pdf/2601.19897)


### Uncertainty in multi-agent systems:
- [On uncertainty in LLM MAS](https://arxiv.org/pdf/2602.04234)
    - initial step certainty is major determinant of performance
- [Uncertainty propagation in knowledge distillation](https://arxiv.org/pdf/2601.18909)
- [Model cascades with confidence tuning](https://arxiv.org/pdf/2502.19335) by Google
    - GateKeeper loss: ...


### Aesthetically similar, but not directly influencing:
- [Phase Transition for Budgeted Multi-Agent Synergy](https://arxiv.org/pdf/2601.17311)
    - theoretical exploration of chain, star, and tree structures for multi-agent coordination
- [Self-steering language models](https://arxiv.org/pdf/2504.07081) (Jacob Andreas)
    - LLMs writing probabilistic programs to define sampling procedure for controllable generation, similar architecture but different goals
- [Stacked LLM policies for Web Actions](https://arxiv.org/abs/2310.03720) (Yoav Artzi)
