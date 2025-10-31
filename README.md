# 안동댐 수질 예측

## 모델
- Vanilla LSTM
- Stacked LSTM

## 디렉토리
- data: 데이터 폴더
- etc: 프로젝트 진행 과정 폴더
- point_data: 포인트별 데이터 구분 폴더
- summary: 포인트별 요약 데이터 폴더
- weights: 가중치 저장 폴더
- src: 스크립트 폴더

## 파일
- codetest.ipynb: 테스트용 파일
- main.py: Lightning 학습 파일
- model.py: 모델 클래스 파일
- dataset.py: 데이터셋 클레스 파일
- utils.py: 유틸리티 파일

---

## 할 일
- [x] 학습 코드 작성
- [x] 저장 코드 작성
- [ ] 시각화
- [ ] 빈 날짜 처리 방법
- [ ] 피쳐 선정 & 비선정 비교