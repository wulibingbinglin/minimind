"""服务器运行的训前检查：读取本地权重 → 格式化固定问题 → generate → 保存结果。

独立入口，不调用 train_lora.py，也不修改模型权重。
这里用 Transformers 按 config.json 声明的结构加载；不是自定义 MiniMind 的 generate。
测试问题为新编场景，不包含私人聊天原文。后续对照须保持问题、模板和生成设置一致。
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SYSTEM = "你是用户的文字风格分身，在与熟悉的同学私聊。根据上下文自然回复；不要把对方的经历说成自己的经历。"
CASES = [
    ("tired", "今天实验做了一整天，结果还没跑出来，有点崩溃。"),
    ("good_news", "我终于把拖了两周的报告写完了！"),
    ("joke", "我说今晚早点睡，结果又刷手机刷到现在。"),
    ("support", "我想报名一个比赛，但又怕自己太菜，要不要试试？"),
    ("boundary", "你还记得我上次考试具体考了多少分吗？"),
]


def main():
    # 1. 路径由命令行指定，避免误读不存在的 out/full_sft_768.pth。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens 必须大于0")
    model_dir = Path(args.model_dir).resolve(strict=True)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print("配置声明：", config.get("model_type"), config.get("architectures"), flush=True)
    print("注意：配置和目录名不能证明下载来源或SFT训练阶段。", flush=True)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前环境无法使用GPU。请先检查服务器环境。")
    # 2. Tokenizer读取词表/模板；AutoModel读取网络配置和已有参数，不训练。
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        output_loading_info=True,
    )
    problems = {key: loading.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if problems:
        raise RuntimeError("加载不完整，停止测试，不把随机初始化参数当作基线：" + str(problems))
    model = model.to(args.device).eval()
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer没有EOS，先核对模型目录。")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print("实际模型类：", type(model).__name__, "；加载检查通过", flush=True)

    # 3. 每次一条样本B=1；Prompt[T] → generate完整[T+R] → 切片回答[R]。
    # 不采样：先固定解码方式，降低随机性对训前/训后比较的干扰。
    records = []
    for case_id, prompt in CASES:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False, open_thinking=False)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(args.device)
        prompt_length = inputs.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=True,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=pad_id)
        answer_ids = generated[0, prompt_length:].tolist()
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        record = dict(id=case_id, prompt=prompt, formatted_text=text,
                      input_shape=list(inputs.input_ids.shape), input_ids=inputs.input_ids[0].tolist(),
                      answer_ids=answer_ids, answer=answer,
                      ended_with_eos=bool(answer_ids and answer_ids[-1] == tokenizer.eos_token_id))
        records.append(record)
        print(f"\n[{case_id}] {prompt}\n模型：{answer}\n输入{prompt_length} Token，生成{len(answer_ids)} Token", flush=True)

    # 4. 保存新文件而非覆盖；不保存权重。结果目录被Git忽略。
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(exist_ok=True)
    output = result_dir / ("baseline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    report = dict(model_dir=str(model_dir), model_class=type(model).__name__, config=config,
                  source_status="来源与训练阶段待人工确认", torch_version=torch.__version__,
                  transformers_version=transformers.__version__, git_commit=revision.stdout.strip(),
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  generation=dict(do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True),
                  system=SYSTEM, records=records)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\n结果保存至：", output, "\n没有训练，没有保存或覆盖模型权重。")


if __name__ == "__main__":
    main()
