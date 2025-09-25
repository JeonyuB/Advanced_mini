import pandas as pd
if __name__ == "__main__":
    #
    # # 데이터셋
    # dataset = [
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "바다"},
    #     {"question": "다음 중 '행복'의 의미에 가까운 것은?", "answer": "기쁨"},
    #     {"question": "다음 단어와 반대 의미를 가진 것은?", "answer": "추위"},
    #     {"question": "다음 문장과 같은 의미를 가진 단어는?", "answer": "달리다"},
    #     {"question": "다음 중 '슬픔'과 가장 가까운 것은?", "answer": "우울"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "밝음"},
    #     {"question": "다음 단어와 유사한 의미를 가진 것은?", "answer": "신속"},
    #     {"question": "다음 중 '큰'의 반대말은?", "answer": "작은"},
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "하늘"},
    #     {"question": "다음 중 '기쁨'과 가장 관련 있는 단어는?", "answer": "행복"},
    #     {"question": "다음 단어와 반대 의미를 가진 것은?", "answer": "뜨거움"},
    #     {"question": "다음 문장과 같은 의미를 가진 단어는?", "answer": "걷다"},
    #     {"question": "다음 단어와 가장 가까운 의미를 가진 것은?", "answer": "빠르다"},
    #     {"question": "다음 중 '차갑다'의 반대말은?", "answer": "따뜻하다"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "무겁다"},
    #     {"question": "다음 중 '슬프다'와 유사한 단어는?", "answer": "우울하다"},
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "나무"},
    #     {"question": "다음 중 '행복'과 관련 없는 것은?", "answer": "슬픔"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "가볍다"},
    #     {"question": "다음 문장과 같은 의미를 가진 단어는?", "answer": "달리다"},
    #     {"question": "다음 단어와 가장 가까운 의미를 가진 것은?", "answer": "밝다"},
    #     {"question": "다음 중 '높다'의 반대말은?", "answer": "낮다"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "빠르다"},
    #     {"question": "다음 중 '슬픔'과 유사한 단어는?", "answer": "애수"},
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "산"},
    #     {"question": "다음 중 '기쁨'과 관련 없는 단어는?", "answer": "화남"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "어둡다"},
    #     {"question": "다음 문장과 같은 의미를 가진 단어는?", "answer": "달다"},
    #     {"question": "다음 단어와 가장 가까운 의미를 가진 것은?", "answer": "강하다"},
    #     {"question": "다음 중 '차갑다'의 반대말은?", "answer": "뜨겁다"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "가볍다"},
    #     {"question": "다음 중 '행복'과 유사한 단어는?", "answer": "기쁨"},
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "물"},
    #     {"question": "다음 중 '슬픔'과 관련 없는 단어는?", "answer": "웃음"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "차갑다"},
    #     {"question": "다음 문장과 같은 의미를 가진 단어는?", "answer": "달리다"},
    #     {"question": "다음 단어와 가장 가까운 의미를 가진 것은?", "answer": "빠르다"},
    #     {"question": "다음 중 '높다'의 반대말은?", "answer": "낮다"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "무겁다"},
    #     {"question": "다음 중 '슬프다'와 유사한 단어는?", "answer": "우울하다"},
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "나무"},
    #     {"question": "다음 중 '행복'과 관련 없는 것은?", "answer": "슬픔"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "가볍다"},
    #     {"question": "다음 문장과 같은 의미를 가진 단어는?", "answer": "걷다"},
    #     {"question": "다음 단어와 가장 가까운 의미를 가진 것은?", "answer": "밝다"},
    #     {"question": "다음 중 '높다'의 반대말은?", "answer": "낮다"},
    #     {"question": "다음 단어와 의미가 반대인 것은?", "answer": "빠르다"},
    #     {"question": "다음 중 '슬픔'과 유사한 단어는?", "answer": "애수"},
    #     {"question": "다음 단어와 유의어가 아닌 것은?", "answer": "산"},
    #     {"question": "다음 중 '기쁨'과 관련 없는 단어는?", "answer": "화남"},
    # ]

    # # DataFrame 변환
    # df = pd.DataFrame(dataset)
    #
    # # 엑셀 저장 (encoding 제거)
    # df.to_excel("qa_dataset.xlsx", index=False)
    # print("qa_dataset.xlsx 파일 생성 완료!")

    # 엑셀 파일 읽기
    df = pd.read_excel("data/qa_dataset.xlsx")

    # CSV로 저장
    df.to_csv("data/qa_dataset.csv", index=False, encoding="utf-8-sig")
    print("data/qa_dataset.csv 파일 생성 완료!")

