# CUE-Agent

[English](README.md) · [研究结果](RESULTS.md) · [Brilliant Researcher](https://github.com/ahian-lee/brilliant-researcher)

CUE-Agent 研究一个问题：Agent 遇到新任务时，过去的经验究竟该不该用？它从同一个决策点比较两种续跑，一种使用经验，另一种沿原策略执行，再估算经验带来的收益或损失。

独立仓库提供 CUE-v0 选择器原型、成对结果评估和合成演示。只需本地 Python 环境，无需模型服务、密钥、BRI 看板或外部任务数据。

## 快速开始

```sh
python -m pip install .
python -m cue_agent
```

演示会从合成样本估出正收益，但最终仍会弃权。只有经过验证的条件平均收益下界能支持应用经验。当前原型的点估计和诊断区间尚未达到这个条件。

使用自己的数据时，给同一决策点的两种续跑提供成对结果。将「使用经验的效用减去原策略的效用」填入 `delta`，未知结果用 `None` 表示。可通过 `project_features` 将可见事实转成输入，并用 `evaluate_policy` 汇总成对结果。

实际试点仍缺足够证据来训练和验证可用的收益选择器。仓库中的测试使用合成输入，历史模型小试和离线回放结论见[研究结果](RESULTS.md)。

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```
