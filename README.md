<div align="center">
<img src="https://cloud.tsinghua.edu.cn/f/3bb3e445a3ce4dc8baa1/?dl=1" width=100% style="vertical-align: bottom;">
<h3>Diffusion-Based Planning for Autonomous Driving with Flexible Guidance</h3>

[Yinan Zheng](https://github.com/ZhengYinan-AIR)\*, Ruiming Liang\*, Kexin Zheng\*, [Jinliang Zheng](https://github.com/2toinf), Liyuan Mao, [Jianxiong Li](https://facebear-ljx.github.io/), Weihao Gu, Rui Ai, [Shengbo Eben Li](https://scholar.google.com/citations?user=Dxiw1K8AAAAJ&hl=zh-CN), [Xianyuan Zhan](https://zhanzxy5.github.io/zhanxianyuan/), [Jingjing Liu](https://air.tsinghua.edu.cn/en/info/1046/1194.htm)

 Paper & code comming soon
</div>

The official implementation of **Diffusion Planner**, which **represents a pioneering effort in fully harnessing the power of diffusion models for high-performance motion planning, without overly relying on refinement**.

<div style="display: flex; justify-content: center; align-items: center; gap: 2%;">

  <img src="https://cloud.tsinghua.edu.cn/f/f829237ae715475e9441/?dl=1" width="32%" alt="Video 1">

  <img src="https://cloud.tsinghua.edu.cn/f/5d24e1a792d04854af9e/?dl=1" width="32%" alt="Video 2">

  <img src="https://cloud.tsinghua.edu.cn/f/5b66086b2ca445d3b5f4/?dl=1" width="32%" alt="Video 3">

</div>


## Table of Contents

- [Methods](#methods)
- [Closed-loop Performance on nuPlan](#closed-loop-performance-on-nuplan)
   - [Learning-based Methods](#learning-based-methods)
   - [Rule-based / Hybrid Methods](#rule-based-hybrid-methods)
   - [Qualitative Results](#qualitative-results)


## Methods

**Diffusion Planner** leverages the expressive and flexible diffusion model to enhance autonomous planning:
* DiT-based architecture focusing on the fusion of noised future vehicle trajectories and conditional information
* Joint modeling of key participants' statuses, unifying motion prediction and closed-loop planning as future trajectory generation
* Fast inference during diffusion sampling, achieving around 20Hz for real-time performance

<image src="https://cloud.tsinghua.edu.cn/f/75af1e961ad74e89bf27/?dl=1" width=100%>

## Closed-loop Performance on nuPlan
### Learning-based Methods

| Methods                            | Val14 (NR) | Val14 \(R\) | Test14-hard (NR) | Test14-hard \(R\) | Test14 (NR) | Test14 \(R\) |
| ---------------------------------- | ---------- | ----------- | ---------------- | ----------------- | ----------- | ------------ |
| PDM-Open*                          | 53.53      | 54.24       | 33.51            | 35.83             | 52.81       | 57.23        |
| UrbanDriver                        | 68.57      | 64.11       | 50.40            | 49.95             | 51.83       | 67.15        |
| GameFormer w/o refine.             | 13.32      | 8.69        | 7.08             | 6.69              | 11.36       | 9.31         |
| PlanTF                             | 84.72      | 76.95       | 69.70            | 61.61             | 85.62       | 79.58        |
| PLUTO w/o refine.*                 | 88.89      | 78.11       | 70.03            | 59.74             | **89.90**   | 78.62        |
| Diffusion-es w/o LLM               | 50.00      | -           | -                | -                 | -           | -            |
| STR2-CPKS-800M w/o refine.*        | 8.80       | -           | 10.99            | -                 | -           | -            |
| Diffusion Planner (ours)           | **89.76**  | **82.56**   | **75.67**        | **68.56**         | **89.22**   | **83.36**    |

*: Using pre-searched reference lines or additional proposals as model inputs provides prior knowledge.

---

### Rule-based / Hybrid Methods

| Methods                              | Val14 (NR) | Val14 \(R\) | Test14-hard (NR) | Test14-hard \(R\) | Test14 (NR) | Test14 \(R\) |
| ------------------------------------ | ---------- | ----------- | ---------------- | ----------------- | ----------- | ------------ |
| **Expert (Log-replay)**              | 93.53      | 80.32       | **85.96**        | 68.80             | 94.03       | 75.86        |
| IDM                                  | 75.60      | 77.33       | 56.15            | 62.26             | 70.39       | 74.42        |
| PDM-Closed                           | 92.84      | 92.12       | 65.08            | 75.19             | 90.05       | 91.63        |
| PDM-Hybrid                           | 92.77      | 92.11       | 65.99            | 76.07             | 90.10       | 91.28        |
| GameFormer                           | 79.94      | 79.78       | 68.70            | 67.05             | 83.88       | 82.05        |
| PLUTO                                | 92.88      | 76.88       | 80.08            | 76.88             | 92.23       | 90.29        |
| Diffusion-es                         | 92.00      | -           | -                | -                 | -           | -            |
| STR2-CPKS-800M                       | 93.91      | 92.51       | 77.54            | **82.02**         | -           | -            |
| Diffusion Planner w/ refine (ours)   | **94.26**  | **92.90**   | 78.87            | **82.00**         | **94.80**   | **91.75**    |

---

###  QualitativeResults

<image src="https://cloud.tsinghua.edu.cn/f/01e54aa90ab44c48b49d/?dl=1" width=100%>

**Future trajectory generation visualization**. A frame from a challenging narrow road turning scenario in the closed-loop test, including the **future planning** of the ego vehicle (*PlanTF* and *PLUTO w/o refine.* showing multiple **candidate trajectories**), **predictions** for neighboring vehicles, and the **ground truth** ego trajectory.


## Acknowledgement
Diffusion Planner is greatly inspired by the following outstanding contributions to the open-source community: [nuplan-devkit](https://github.com/motional/nuplan-devkit), [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner), [tuplan_garage](https://github.com/autonomousvision/tuplan_garage), [planTF](https://github.com/jchengai/planTF), [pluto](https://github.com/jchengai/pluto), [StateTransformer](https://github.com/Tsinghua-MARS-Lab/StateTransformer), [DiT](https://github.com/facebookresearch/DiT), [dpm-solver](https://github.com/LuChengTHU/dpm-solver)
