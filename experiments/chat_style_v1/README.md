# 聊天风格分身：第一轮训前基线

本地维护本目录脚本，通过Git同步；服务器运行。私人聊天留在仓库外，不能提交到GitHub。
本实验不改学习源码，不训练、不转换、不覆盖已有权重。当前脚本仅依赖Transformers，不依赖本地尚未提交的MiniMind源码改动。

## 1. 确认服务器拿到脚本，再加载现有模型

从服务器仓库根目录运行（代码推送并拉取完成以后）：

```bash
cd /data/lxj/fanhuilin/minimind
conda activate fhl_minimind
CUDA_VISIBLE_DEVICES=0 python experiments/chat_style_v1/baseline.py --model-dir ./minimind-3
```

按config.json声明的模型类原生加载，缺失/额外/形状错误则停止，不用strict=False掩盖问题。
此处不是自定义MiniMind网络；原生模型能加载并不证明两种实现完全等价。
下载源、模型训练阶段仍须补充原始链接或下载记录，不能由qwen3或目录名推断。

## 2. 固定问题进入模板，再生成完整回答

5个新编同学聊天场景，不使用私人对话原文，也不作为训练样本。
system定义角色，user提供问题；模板追加assistant开头，再编码为[1,T]。
generate内部循环forward，返回Prompt与新回答；切掉前T个ID后解码。
关闭采样，最多96个新Token；结果记录实际格式化文本，检查思考开关是否被模板采用。

### 第二次：仅改变系统提示的对照

第一次original提示、五个问题与生成设置保持在代码中，不覆盖旧结果。
第二次concise明确要求先接住话意、一到两句口语、不列清单、不编造经历：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/chat_style_v1/baseline.py --model-dir ./minimind-3 --system-variant concise
```

不加该参数仍运行original。两版结果以不同版本名及时间戳保存，终端显示是否以EOS结束。
system和user都是模型本次读取的输入；修改system不更新参数，不等于微调。
比较重点是接话是否正确、回复是否简短和相关，而不只是短了多少；不要把更短误当作更正确。
这五题经过提示调试后属于开发集，后续判断泛化需要另留未用于调试的新问题。

## 3. 保存结果，用于微调前后对照

results目录保存时间戳JSON（Git忽略），包含配置、软件版本、输入ID、回答ID及是否EOS结束。
这里只记录配置哈希，不冒充完整权重校验。尚未取得服务器运行结果前，不能宣称模型已通过检查。
每条回答人工检查：是否接得上话、自然简短、是否编造历史、是否胡乱输出或到长度上限仍不停止。
baseline用于前后对照，不是大规模能力评测；基础质量差时先处理基础模型，不靠两个样本强行修复。

后续LoRA比较必须保持同一基础权重、Tokenizer、模板、问题和生成设置；如果改用自定义MiniMind，先重新跑其未微调基线，再同实现比较，避免混淆实现差异和训练收益。
