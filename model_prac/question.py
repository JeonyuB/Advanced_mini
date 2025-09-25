#!/usr/bin/env python3
import torch
from prac import TinyGPT, GPTConfig, CharTokenizer  # prac.py에서 가져오기

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 체크포인트 로드 --------------------
    ckpt_path = "tinyllm_qa_1.pth"
    ckpt = torch.load(ckpt_path, map_location=device)

    config = GPTConfig(
        vocab_size=ckpt['config']['vocab_size'],
        n_embd=ckpt['config']['n_embd'],
        n_head=ckpt['config']['n_head'],
        n_layer=ckpt['config']['n_layer'],
        block_size=ckpt['config']['block_size'],
        dropout=ckpt['config']['dropout']
    )

    model = TinyGPT(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # -------------------- 토크나이저 재생성 --------------------
    tokenizer = CharTokenizer("")  # 초기화 후 vocab 덮어쓰기
    tokenizer.stoi = ckpt['tokenizer']['stoi']
    tokenizer.itos = ckpt['tokenizer']['itos']
    tokenizer.vocab_size = len(tokenizer.stoi)

    # -------------------- 질문+답 생성 --------------------
    start_text = "Q: "
    start_ids = tokenizer.encode(start_text).unsqueeze(0).to(device)

    with torch.no_grad():
        # max_new_tokens와 top_k를 늘려서 더 안정적인 생성
        out_ids = model.generate(start_ids, max_new_tokens=500, temperature=1.0, top_k=50)
        out_text = tokenizer.decode(out_ids[0])

    # -------------------- raw 출력 확인 --------------------
    print("=== raw generated text ===")
    print(out_text)
    print("==========================")

    # -------------------- Q/A 분리 --------------------
    qa_pairs = []
    # Q:로 split 후 A:가 있는 경우만 추출
    entries = out_text.split("Q:")
    for entry in entries:
        parts = entry.split("A:")
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            if question and answer:
                qa_pairs.append((question, answer))

    # -------------------- 출력 --------------------
    if qa_pairs:
        for i, (q, a) in enumerate(qa_pairs, 1):
            print(f"#{i} Q: {q}")
            print(f"    A: {a}")
            print("-"*40)
    else:
        print("생성된 Q/A 쌍이 없습니다. 모델 학습 데이터나 step을 확인하세요.")

if __name__ == "__main__":
    main()
