import torch
import os
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, Audio
import argparse
import queue

# 全局日志队列（由 app.py 传入）
asr_log_queue = None

def set_log_queue(q):
    """设置日志队列"""
    global asr_log_queue
    asr_log_queue = q

def log(msg):
    """统一日志输出：写入队列 + 打印（降级）"""
    full_msg = f"[ASR训练] {msg}"
    if asr_log_queue is not None:
        try:
            asr_log_queue.put(full_msg)
        except:
            pass  # 队列已关闭
    print(full_msg)  # 同时保留控制台输出

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="微调 Whisper 模型")
    parser.add_argument("--model_path", required=True, help="预训练模型路径")
    parser.add_argument("--data_dir", required=True, help="数据集路径")
    parser.add_argument("--output_dir", required=True, help="保存微调模型的路径")
    parser.add_argument("--batch_size", type=int, default=8, help="每设备训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志步数")
    parser.add_argument("--save_total_limit", type=int, default=2, help="保存模型数量限制")
    parser.add_argument("--fp16", type=str, default="true", help="是否启用 FP16 训练")
    args = parser.parse_args()

    # 检查GPU可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"使用设备: {device}")

    # 加载处理器和模型
    processor = WhisperProcessor.from_pretrained(args.model_path)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model = model.to(device)
    log(f"模型加载完成: {args.model_path}")

    # 加载数据集
    def read_trn_file(trn_file):
        try:
            with open(trn_file, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                if lines and len(lines) >= 1:
                    return lines[0].strip()
                else:
                    return None
        except Exception as e:
            log(f"读取 {trn_file} 时出错：{e}")
            return None

    audio_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".wav")]
    transcriptions = []
    valid_audio_files = []

    for audio_file in audio_files:
        trn_file = audio_file + ".trn"
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            if sr != 16000:
                log(f"警告：{audio_file} 采样率 {sr}，期望 16000，跳过。")
                continue
            text = read_trn_file(trn_file)
            if text:
                transcriptions.append(text)
                valid_audio_files.append(audio_file)
            else:
                log(f"警告：{trn_file} 无效或为空，跳过 {audio_file}")
        except Exception as e:
            log(f"处理 {audio_file} 时出错：{e}，跳过。")

    if len(valid_audio_files) != len(transcriptions):
        log(f"错误：音频 ({len(valid_audio_files)}) 和转录 ({len(transcriptions)}) 数量不匹配！")
        exit(1)

    data = {
        "audio": valid_audio_files,
        "text": transcriptions
    }
    dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=16000))
    log(f"加载数据集完成，有效样本数：{len(dataset)}")

    # 预处理数据
    def preprocess(batch):
        try:
            audio = batch["audio"]
            input_features = processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ).input_features.squeeze()
            labels = processor.tokenizer(
                batch["text"], return_tensors="pt", padding=False
            ).input_ids.squeeze()
            return {
                "input_features": input_features.tolist(),
                "labels": labels.tolist()
            }
        except Exception as e:
            log(f"预处理 {batch['audio']['path']} 时出错：{e}")
            return None

    dataset = dataset.map(preprocess, remove_columns=["audio", "text"], num_proc=1)
    dataset = dataset.filter(lambda x: x["input_features"] is not None and x["labels"] is not None, num_proc=1)
    log(f"预处理后有效样本数：{len(dataset)}")

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16.lower() == "true",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to=[]  # 关闭 wandb 等
    )

    # 数据整理器
    def data_collator(data):
        input_features = [torch.tensor(f["input_features"]) for f in data if f["input_features"] is not None]
        labels = [torch.tensor(f["labels"]) for f in data if f["labels"] is not None]

        for i, (feat, lab) in enumerate(zip(input_features, labels)):
            if not isinstance(feat, torch.Tensor) or not isinstance(lab, torch.Tensor):
                raise TypeError(f"样本 {i}: 预期 Tensor，收到 input_features {type(feat)}, labels {type(lab)}")
            if feat.dim() != 2 or lab.dim() != 1:
                raise ValueError(f"样本 {i}: input_features 维度 {feat.dim()}, labels 维度 {lab.dim()}")

        labels_padded = processor.tokenizer.pad(
            {"input_ids": labels},
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            "input_features": torch.stack(input_features),
            "labels": labels_padded["input_ids"],
            "attention_mask": labels_padded["attention_mask"]
        }

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 开始训练（实时日志由 Trainer 内部输出）
    log(f"开始训练，epochs={args.num_train_epochs}，batch_size={args.batch_size}")
    trainer.train()
    log("训练完成！")

    # 保存模型
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log(f"微调模型已保存至 {args.output_dir}")

if __name__ == "__main__":
    main()