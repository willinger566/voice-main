# 本轮新增语料与可追溯记录

本文件记录历史训练来源。按精简部署要求，下载语料、派生清单和特征缓存已清理；当前仅保留六个网页示例、三个跨语料验证音频、数据计数及许可/校验记录。

## 实际整理数量

| 来源 | 原始录音 | 训练 | 验证 | 测试 | 划分 |
|---|---:|---:|---:|---:|---|
| 原 CASIA | 1,200 | 840 | 180 | 180 | 保留旧划分，三个集合共享四名说话人 |
| CSEMOTIONS 六类子集 | 3,539 | 2,065 | 776 | 698 | 训练 6 个、验证 2 个、测试 2 个说话人 ID，互斥 |
| RAVDESS 六类语音子集 | 1,055 | 703 | 176 | 176 | 训练演员 01–16，验证 17–20，测试 21–24 |
| CREMA-D 高一致率声音标注子集 | 4,049 | 2,890 | 597 | 562 | 训练演员 1001–1065，验证 1066–1078，测试 1079–1091 |
| 合计 | 9,843 | 6,498 | 1,729 | 1,616 | 增广与长音频切片不计为新的独立录音 |

CSEMOTIONS 原数据共 4,160 条，排除 621 条 playfulness。RAVDESS 原语音包 1,440 条，排除 384 条 calm/disgust，另剔除训练集内一条完全重复录音。未把相邻情绪强行改成目标标签。未将项目内按情绪重复存放的 CASIA 副本作为额外数据。

## 发布方与下载验证

- [CSEMOTIONS 作者发布仓库](https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS)，相关 [Marco-Voice 项目](https://github.com/ATH-MaaS/Marco-Voice)。下载时固定仓库 revision，并核对 8 个 Parquet 分片的发布 SHA256。数据卡与实际字段中存在说话人数表述差异，本项目使用文件中实际的 10 个 speaker ID，未凭网页说明猜测身份。
- [RAVDESS 官方 Zenodo 记录](https://zenodo.org/records/1188976)，仅下载 `Audio_Speech_Actors_01-24.zip`，MD5 为 `bc696df654c87fed845eb13823edef8a`。[论文](https://doi.org/10.1371/journal.pone.0196391)：Livingstone SR, Russo FA (2018), The Ryerson Audio-Visual Database of Emotional Speech and Song.

来源与校验记录现存于 `licenses/{数据集}/source.json`。CSEMOTIONS 的原始 README、NOTICE 也已保存；大体积语料本体已删除。

CREMA-D 使用[原发布方](https://github.com/CheyneyComputerScience/CREMA-D)的 `tabulatedVotes.csv`（queryType=1）与 `summaryTable.csv/VoiceVote` 交叉核对，固定 Git commit 与 CSV SHA256；音频使用 [公开 Parquet 镜像](https://huggingface.co/datasets/cfahlgren1/crema-d)，固定 revision，4 个分片共约606 MB，逐个验证 SHA256。仅纳入声音多数票唯一、至少6人评分、一致率≥60%的五类声音（无 surprise）。排除低一致率/并列/不支持类别3391条、重复1条及无法对齐原始文件名的1条。这是经过筛选的子集成绩，不能当作完整 CREMA-D 基准。原始表演目标标签不替代听者的声音标注。其发布许可为 ODbL-1.0 / DbCL-1.0，保存原许可和署名信息。

## 实际使用范围

CSEMOTIONS 数据卡标为 Apache-2.0，但其 NOTICE 还注明引入 ESD，并包含研究用途限制；RAVDESS 标注 CC BY-NC-SA 4.0。因此本轮数据及派生检查点按研究/非商业用途记录，不能仅依据 CSEMOTIONS 卡片将整个组合视为纯 Apache 商业语料。

[ESD 官方仓库](https://github.com/HLTSingapore/Emotional-Speech-Data)要求提交许可申请，[EmotionTalk 官方仓库](https://huggingface.co/datasets/BAAI/Emotiontalk)要求登录并接受访问条件。本轮没有绕过这些要求直接获取它们的受限文件，也没有对外发送邮件或提交个人信息。

## 防止泄漏

- 训练时的 manifest 保存来源、说话人、原录音 ID、划分、标签、路径及 16 kHz PCM 指纹；精简时已删除，计数/划分和其哈希保留在 `results/dataset_report.json`。
- 所有数据合并检查跨集合的完全重复 PCM；发现冲突则停止。原 CASIA 的说话人重叠明确保留并单独报告。
- 长录音按与服务相同的规则切片，保留原 source_id；同一录音的所有片段继承同一划分。
- 噪声、速度、短混响扰动仅用于 CASIA 训练录音；验证/测试不做增广。
- 共提取 9,952 条片段/增广特征，但按原始录音聚合评估，不能把这些特征行宣传为 9,952 条独立录音。
- 可训练骨干前的冻结层缓存只包含训练与验证，不包含测试片段。
- 候选、早停和温度校准只使用验证集；选定发布检查点后，才运行固定原测试和新增外部测试。原测试上一轮已使用，因此称固定回归集，不能称新盲测。
- emotion2vec 原始预训练数据是否包含这些语料无法完整核实；本轮可证明的是项目新增训练和验证没有混入外部测试说话人，不能据此声称骨干从未接触过任何相关音频。

V4训练时保留V3前三个语料的所有划分；该训练目录已清理。V3测试已评估，此后这些集合称历史回归集；新增CREMA测试演员只在V4选择完成后评估。V4特征共14001行，包含切片/增广，不是14001条独立录音。CREMA听感筛选后中性类别占多数，必须同时看macro-F1与分类别指标，不能只看总准确率。
