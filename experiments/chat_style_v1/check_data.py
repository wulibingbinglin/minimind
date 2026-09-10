"""服务器数据检查：私有JSONL → 项目SFTDataset → IDs/labels → 错位监督与Batch。

独立运行，不是baseline.py自动调用的下一步。不加载大模型、不训练。
默认只显示第一条样本明细，全部样本都会检查。日志包含私人文本，不要公开。
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys


def spans(mask):
    """把连续为True的位置整理成[start,end)，方便逐段看监督文本。"""
    result = []
    start = None
    for i, enabled in enumerate(mask + [False]):
        if enabled and start is None:
            start = i
        if not enabled and start is not None:
            result.append((start, i))
            start = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--show-sample", type=int, default=0)
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("max-length至少为2")
    data_path = Path(args.data).resolve(strict=True)
    model_dir = Path(args.model_dir).resolve(strict=True)
    raw = [json.loads(line) for line in data_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not raw or not 0 <= args.show_sample < len(raw):
        raise ValueError("数据为空或show-sample超出范围")
    for sample in raw:
        messages = sample["conversations"]
        roles = [m["role"] for m in messages]
        if roles != ["system"] + ["user", "assistant"] * ((len(roles) - 1) // 2):
            raise ValueError("本实验要求system开头，随后user/assistant交替，assistant结尾")
        if len(messages) < 3 or not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
            raise ValueError("样本含空内容或缺少问答")

    # 1. 数据可能被datasets写入缓存，因此先把缓存定位到私有数据旁边。
    os.environ["HF_DATASETS_CACHE"] = str(data_path.parent / "dataset_cache")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import datasets  # 先导入，保持项目Windows DLL兼容导入顺序
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from dataset.lm_dataset import SFTDataset, pre_processing_chat, post_processing_chat

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if any(v is None for v in (tokenizer.pad_token_id, tokenizer.bos_token, tokenizer.eos_token)):
        raise ValueError("缺少SFTDataset需要的特殊Token，先核对Tokenizer")
    dataset = SFTDataset(str(data_path), tokenizer, max_length=args.max_length)
    total_targets = 0
    print("样本数量：", len(dataset), "；max_length：", args.max_length)
    for index in range(len(dataset)):
        # 2. __getitem__存在随机的空think处理；重放同一随机状态，才能比较同一次实际输入。
        random.seed(2026 + index)
        state = random.getstate()
        ids, labels = dataset[index]
        random.setstate(state)
        messages = pre_processing_chat(dataset.samples[index]["conversations"])
        text = post_processing_chat(dataset.create_chat_prompt(messages))
        unpadded = tokenizer(text).input_ids
        if len(unpadded) > args.max_length:
            raise ValueError(f"样本{index}长度{len(unpadded)}超过上限，请调大max-length；不要截掉答案")
        expected_ids = unpadded + [tokenizer.pad_token_id] * (args.max_length - len(unpadded))
        assert ids.tolist() == expected_ids, "重建输入与实际Dataset不一致"
        assert ids.shape == labels.shape == (args.max_length,)
        assert torch.all(labels[len(unpadded):] == -100), "PAD意外参与监督"
        assert torch.all((labels == -100) | (labels == ids)), "labels应与输入同位置对齐"
        ranges = spans((labels != -100).tolist())
        expected_blocks = sum(m["role"] == "assistant" for m in messages)
        assert len(ranges) == expected_blocks, "助手段数与监督段数不一致"
        for start, end in ranges:
            assert ids.tolist()[start - len(dataset.bos_id):start] == dataset.bos_id
            assert ids.tolist()[end - len(dataset.eos_id):end] == dataset.eos_id
        count = int((labels[1:] != -100).sum())
        assert count > 0, "没有有效监督目标"
        total_targets += count
        print(f"样本{index}：正文含模板{len(unpadded)} Token，PAD {args.max_length-len(unpadded)}，助手段{len(ranges)}，有效目标{count}")
        if index == args.show_sample:
            print("\n实际训练文本：", repr(text))
            print("input_ids：", ids.tolist())
            print("labels：", labels.tolist())
            for start, end in ranges:
                print(f"监督段[{start}:{end}]：", repr(tokenizer.decode(ids[start:end].tolist(), skip_special_tokens=False)))
            print("错位配对：logits[t]预测labels[t+1]，并非预测自己所在位置的输入")
            for target in range(1, len(unpadded)):
                if labels[target] != -100:
                    print(f"logits[{target-1}] → labels[{target}]={labels[target].item()} "
                          f"目标={tokenizer.decode([labels[target].item()])!r}")
    # 3. DataLoader把单条[T]叠成[B,T]；模型内部才错位，此处不提前平移labels。
    batch_ids, batch_labels = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)))
    print("\nBatch input_ids/labels：", tuple(batch_ids.shape), tuple(batch_labels.shape))
    print("全部样本检查通过；本次固定随机处理下有效目标合计：", total_targets)
    print("没有加载模型、没有训练。输出含私有聊天，请勿公开。")


if __name__ == "__main__":
    main()
