<h1 align="center">Metis-RISE: RL Incentivizes and SFT Enhance Multimodal Reasoning Model Learning</h1>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2506.13056-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.13056)&ensp;[![Code License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

</h5>



## 💡 Overview

We introduces **Metis-RISE** (**R**L **I**ncentivizes and **S**FT **E**nhances), a hybrid training paradigm that strategically sequences RL and SFT to significantly advance multimodal reasoning in MLLMs. By prioritizing RL-driven exploration, Metis-RISE incentivizes the model to unlock latent reasoning skills and avoids premature convergence often seen in SFT-first approaches. Subsequently, targeted SFT stages enhance these capabilities by efficiently addressing inconsistent reasoning through self-distilled trajectories and rectifying fundamental capability absence via expert knowledge injection.

<img src="assets/framework.png" alt="Metis-RISE Framework Overview" style="width:850px; max-width:100%;">

## 📢 News
- **[2025-06-16]** We release [Metis-RISE: RL Incentivizes and SFT Enhance Multimodal Reasoning Model Learning](https://arxiv.org/abs/2506.13056) on Arxiv! 🎉🎉🎉


## 📊 Results
We evaluate both **Metis-RISE-7B** and **Metis-RISE-72B** on the comprehensive [**OpenCompass Multimodal Reasoning Leaderboard**](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning/?m=REALTIME).
Both of them achieve state-of-the-art performance among similar-sized models, with the 72B version ranking fourth overall on the full leaderboard (as of June 16, 2025), validating the effectiveness and scalability of the Metis-RISE framework for enhancing multimodal reasoning.

<table>
<thead>
<tr>
<th align="left"><strong>Model</strong></th>
<th align="center"><strong>Avg.</strong></th>
<th align="center"><strong>MathVista</strong></th>
<th align="center"><strong>MathVision</strong></th>
<th align="center"><strong>MathVerse</strong></th>
<th align="center"><strong>DynaMath</strong></th>
<th align="center"><strong>WeMath</strong></th>
<th align="center"><strong>LogicVista</strong></th>
</tr>
</thead>
<tbody>
<tr style="background-color: #f0f0f0;">
<td colspan="8" align="center"><strong><em>Proprietary Models</em></strong></td>
</tr>
<tr>
<td>Seed1.5-VL</td>
<td align="center">73.3</td>
<td align="center">86.8</td>
<td align="center">67.3</td>
<td align="center">79.3</td>
<td align="center">56.1</td>
<td align="center">77.5</td>
<td align="center">72.7</td>
</tr>

<tr>
<td>Gemini-2.5-Pro</td>
<td align="center">72.5</td>
<td align="center">80.9</td>
<td align="center">69.1</td>
<td align="center">76.9</td>
<td align="center">56.3</td>
<td align="center">78.0</td>
<td align="center">73.8</td>
</tr>

<tr>
<td>Doubao-1.5-Pro</td>
<td align="center">61.6</td>
<td align="center">78.6</td>
<td align="center">51.5</td>
<td align="center">64.7</td>
<td align="center">44.9</td>
<td align="center">65.7</td>
<td align="center">64.2</td>
</tr>
<tr>
<td>Gemini-2.0-Pro</td>
<td align="center">56.6</td>
<td align="center">71.3</td>
<td align="center">48.1</td>
<td align="center">67.3</td>
<td align="center">43.3</td>
<td align="center">56.5</td>
<td align="center">53.2</td>
</tr>
<tr>
<td>ChatGPT-4o-202504</td>
<td align="center">54.8</td>
<td align="center">71.6</td>
<td align="center">43.8</td>
<td align="center">49.9</td>
<td align="center">48.5</td>
<td align="center">50.6</td>
<td align="center">64.4</td>
</tr>
<tr>
<td>Gemini-2.0-Flash</td>
<td align="center">50.6</td>
<td align="center">70.4</td>
<td align="center">43.6</td>
<td align="center">47.8</td>
<td align="center">42.1</td>
<td align="center">47.4</td>
<td align="center">52.3</td>
</tr>
<tr>
<td>Claude 3.7 Sonnet</td>
<td align="center">50.4</td>
<td align="center">66.8</td>
<td align="center">41.9</td>
<td align="center">46.7</td>
<td align="center">39.7</td>
<td align="center">49.3</td>
<td align="center">58.2</td>
</tr>
<tr>
<td>GLM-4v-Plus-202501</td>
<td align="center">49.2</td>
<td align="center">73.5</td>
<td align="center">51.1</td>
<td align="center">40.7</td>
<td align="center">27.5</td>
<td align="center">47.7</td>
<td align="center">54.4</td>
</tr>
<tr style="background-color: #f0f0f0;">
<td colspan="8" align="center"><strong><em>≤10B Models</em></strong></td>
</tr>
<tr>
<td>Kimi-VL-A3B-Instruct</td>
<td align="center">35.8</td>
<td align="center">66.0</td>
<td align="center">21.8</td>
<td align="center">34.1</td>
<td align="center">18.0</td>
<td align="center">32.3</td>
<td align="center">42.7</td>
</tr>
<tr>
<td>Qwen2.5-VL-7B</td>
<td align="center">40.1</td>
<td align="center">68.1</td>
<td align="center">25.4</td>
<td align="center">41.1</td>
<td align="center">21.8</td>
<td align="center">36.2</td>
<td align="center">47.9</td>
</tr>
<tr>
<td>InternVL3-8B</td>
<td align="center">41.4</td>
<td align="center">70.5</td>
<td align="center">30.0</td>
<td align="center">38.5</td>
<td align="center">25.7</td>
<td align="center">39.5</td>
<td align="center">44.5</td>
</tr>
<tr>
<td>VLAA-Thinker-7B</td>
<td align="center">42.5</td>
<td align="center">68.0</td>
<td align="center">26.4</td>
<td align="center">48.2</td>
<td align="center">22.4</td>
<td align="center">41.5</td>
<td align="center">48.5</td>
</tr>
<tr style="background-color: #e6f3ff;">
<td><strong>Metis-RISE-7B</strong></td>
<td align="center"><strong>46.4</strong></td>
<td align="center"><strong>75.8</strong></td>
<td align="center"><strong>28.7</strong></td>
<td align="center"><strong>51.0</strong></td>
<td align="center"><strong>27.7</strong></td>
<td align="center"><strong>45.2</strong></td>
<td align="center"><strong>49.7</strong></td>
</tr>
<tr style="background-color: #f0f0f0;">
<td colspan="8" align="center"><strong><em>>10B Models</em></strong></td>
</tr>
<tr>
<td>InternVL3-14B</td>
<td align="center">46.0</td>
<td align="center">74.4</td>
<td align="center">34.0</td>
<td align="center">43.7</td>
<td align="center">30.3</td>
<td align="center">41.3</td>
<td align="center">52.1</td>
</tr>
<tr>
<td>Ovis2-34B</td>
<td align="center">47.9</td>
<td align="center">76.1</td>
<td align="center">31.9</td>
<td align="center">50.1</td>
<td align="center">27.5</td>
<td align="center">51.9</td>
<td align="center">49.9</td>
</tr>
<tr>
<td>QVQ-72B-Preview</td>
<td align="center">46.9</td>
<td align="center">70.3</td>
<td align="center">34.9</td>
<td align="center">48.2</td>
<td align="center">30.7</td>
<td align="center">39.0</td>
<td align="center">58.2</td>
</tr>
<tr>
<td>LLaVA-OneVision-72B</td>
<td align="center">34.7</td>
<td align="center">67.1</td>
<td align="center">25.3</td>
<td align="center">27.2</td>
<td align="center">15.6</td>
<td align="center">32</td>
<td align="center">40.9</td>
</tr>
<tr>
<td>Qwen2.5-VL-72B</td>
<td align="center">50.3</td>
<td align="center">74.2</td>
<td align="center">39.3</td>
<td align="center">47.3</td>
<td align="center">35.9</td>
<td align="center">49.1</td>
<td align="center">55.7</td>
</tr>
<tr>
<td>InternVL3-78B</td>
<td align="center">51.0</td>
<td align="center">79.0</td>
<td align="center">38.8</td>
<td align="center">51.0</td>
<td align="center">35.1</td>
<td align="center">46.1</td>
<td align="center">55.9</td>
</tr>
<tr style="background-color: #e6f3ff;">
<td><strong>Metis-RISE-72B</strong></td>
<td align="center"><strong>56.6</strong></td>
<td align="center"><strong>80.4</strong></td>
<td align="center"><strong>42.7</strong></td>
<td align="center"><strong>59.8</strong></td>
<td align="center"><strong>42.5</strong></td>
<td align="center"><strong>55.1</strong></td>
<td align="center"><strong>58.8</strong></td>
</tr>
</tbody>
</table>


## 🚀 (Coming Soon) Code & Models

We plan to release the code in the near future. Stay tuned!

## 🙏 Acknowledgement
We sincerely appreciate [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for providing high-quality training framework.

## 📖 Citation

Coming soon
