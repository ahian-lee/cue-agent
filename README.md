# CUE-Agent

[中文](README_CN.md) · [Research results](RESULTS.md#english) · [Brilliant Researcher](https://github.com/ahian-lee/brilliant-researcher)

CUE-Agent studies when a tool-using agent should reuse a past experience. It compares two continuations from the same context, one with the experience and one with the original policy, and estimates the signed difference in utility.

This standalone Python package contains the CUE-v0 research prototype, paired outcome metrics and a synthetic demo. Run it locally without a model service, credentials, the BRI workbench or an external task dataset.

## Quick start

```sh
python -m pip install .
python -m cue_agent
```

The demo fits a positive estimate on synthetic pairs and prints an abstention. Applying experience requires a validated lower bound for the conditional mean benefit. The prototype's point estimates and diagnostic intervals do not establish that bound.

```python
from cue_agent import CUEv0

selector = CUEv0().fit([
    {"pair_id": f"demo-{i}", "features": {}, "delta": 1.0}
    for i in range(6)
])
result = selector.decide({})
print(result["decision"])
```

Supply paired results from the same decision context. Each `delta` is experience utility minus original-policy utility. Use `None` for an unknown outcome. The package exposes `project_features` for converting visible task facts into inputs and `evaluate_policy` for summarizing paired outcomes.

## Research status

Evidence remains insufficient to validate a selector or claim a whole-task gain on the recorded pilot. The release contains the experimental implementation and synthetic tests, with separate results for live pilots and offline replay. See [Research results](RESULTS.md#english).

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```
