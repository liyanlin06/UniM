<div align="center">

# UniM

## UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark

[![Project Page](https://img.shields.io/badge/Project-Page-0A66C2?style=for-the-badge&logo=googlechrome&logoColor=white)](https://any2any-mllm.github.io/unim/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.05075-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.05075)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/yanlinli/UniM)
[![GitHub Repo stars](https://img.shields.io/github/stars/liyanlin06/UniM?style=for-the-badge)](https://github.com/liyanlin06/UniM)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-1F6FEB?style=for-the-badge)](https://arxiv.org/abs/2603.05075)

Official repository for **UniM**, the first **unified any-to-any interleaved multimodal benchmark** for evaluating both multimodal understanding and multimodal generation under a single paradigm.

</div>

---

![UniM Teaser](https://any2any-mllm.github.io/unim/static/assets/teaser.png)

## Overview

Real-world multimodal systems must handle **arbitrarily combined, interleaved multimodal inputs** and produce outputs in **any interleaved multimedia form**. UniM is designed to benchmark this setting in a unified way.

UniM contains:

- **31K** high-quality instances
- **30** diverse domains
- **7** representative modalities: `text`, `image`, `audio`, `video`, `document`, `code`, and `3D`

Beyond the dataset itself, UniM also provides:

- a unified **evaluation suite**
- structured **prediction and scoring protocols**
- an agentic baseline, **UniMA**

## Highlights

- **Unified any-to-any setting**: jointly studies multimodal understanding and generation instead of separating them into narrow sub-benchmarks.
- **Rich interleaving**: supports arbitrarily interleaved combinations of modalities rather than simple text-image pairs.
- **Broad modality coverage**: spans seven modalities, including less commonly supported ones such as `document`, `code`, and `3D`.
- **Comprehensive evaluation**: measures semantic quality, structural correctness, and cross-modal coherence together.

## Benchmark at a Glance

| Aspect | Description |
| --- | --- |
| Benchmark name | `UniM` |
| Paper | `UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark` |
| Venue | `CVPR 2026` |
| Scale | `31K` instances |
| Domains | `30` |
| Modalities | `7` |
| Baseline | `UniMA` |
| Project page | [any2any-mllm.github.io/unim](https://any2any-mllm.github.io/unim/) |
| Dataset | [huggingface.co/datasets/yanlinli/UniM](https://huggingface.co/datasets/yanlinli/UniM) |

## Evaluation Suite

UniM evaluates models along three complementary dimensions:

1. **Semantic Correctness & Generation Quality**
   Measures whether generated outputs are semantically aligned with the reference and whether the generated content is of reasonable perceptual and structural quality.
2. **Response Structure Integrity**
   Checks whether a model follows the required response format, especially the expected modality types and item counts.
3. **Interleaved Coherence**
   Evaluates whether different generated modalities form a coherent interleaved response in terms of logic, structure, tone, and style.

The evaluation suite includes the following core metrics:

- `SQCS`: Semantic-Quality Coupled Score
- `StS`: Strict Structure Score
- `LeS`: Lenient Structure Score
- `ICS`: Interleaved Coherence Score

![UniM Evaluation Suite](https://any2any-mllm.github.io/unim/static/assets/eval.png)

## Repository Layout

This repository is being organized around three main components:

```text
UniM/
├── README.md
├── data/
│   └── ...
├── evaluation/
│   ├── scripts/
│   ├── metrics/
│   ├── prompts/
│   ├── examples/
│   └── README.md
├── models/
│   └── ...
├── assets/
│   └── ...
└── docs/
    └── ...
```

The public release will focus on:

- **Data**: benchmark samples, metadata, and access instructions
- **Evaluation**: scripts, metrics, judge prompts, and reproducible scoring pipelines
- **Method code**: reference code for our own baseline and related examples

## Release Status

The repository is currently being cleaned up for public release.

- `data/`: prepared
- `evaluation/`: under active cleanup and documentation
- `method code`: will be released after repository finalization

This means some detailed instructions are intentionally still marked as coming soon while we standardize the public-facing codebase.

## Coming Soon

We will gradually add:

- environment setup instructions
- evaluation dependencies
- runnable evaluation commands
- input and output format examples
- baseline inference or reproduction code
- leaderboard or result submission details

## Quick Links

- Project page: [https://any2any-mllm.github.io/unim/](https://any2any-mllm.github.io/unim/)
- Paper: [https://arxiv.org/abs/2603.05075](https://arxiv.org/abs/2603.05075)
- Dataset: [https://huggingface.co/datasets/yanlinli/UniM](https://huggingface.co/datasets/yanlinli/UniM)
- Code repository: [https://github.com/liyanlin06/UniM](https://github.com/liyanlin06/UniM)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=liyanlin06/UniM&type=Date)](https://star-history.com/#liyanlin06/UniM&Date)

## Citation

If you find UniM useful in your research, please consider citing:

```bibtex
@article{li2026unim,
  title={UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark},
  author={Li, Yanlin and Guo, Minghui and Zhang, Kaiwen and Zhang, Shize and Zhao, Yiran and Li, Haodong and Zhou, Congyue and Zheng, Weijie and Yan, Yushen and Wu, Shengqiong and Ji, Wei and Cui, Lei and Wei, Furu and Fei, Hao and Lee, Mong-Li and Hsu, Wynne},
  journal={arXiv preprint arXiv:2603.05075},
  year={2026}
}
```

## License

License information will be added together with the full public release.

## Acknowledgement

This repository accompanies the UniM benchmark and its associated project page. More detailed documentation will be added as the evaluation and method code are finalized for open-source release.
