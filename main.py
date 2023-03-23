import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.figure(figsize=(7, 7))

enterprise = pd.read_csv('로봇산업_기업_현황_20221130123620.csv')  # 기업현황
sales = pd.read_csv('로봇산업_매출_현황_20221130123612.csv')  # 매출현황
production = pd.read_csv('로봇산업_생산_현황_20221130123628.csv')  # 생산현황

total_production = [1, 10, 19, 25, 32, 36, 42]  # 총생산량
x = enterprise['2020'][1:]  # 기업명
y = sales['2020'][1:]  # 매출액
x = np.array(x)
y = np.array(y)  # 매출액

for i in range(7):
    plt.text(x[i], y[i], sales['업종별(1)'][i], fontsize=9, horizontalalignment='center', verticalalignment='bottom',
             rotation=45)  # 매출액에 업종별(1) 텍스트 추가

x = x.reshape(-1, 1)  # x를 2차원으로 변환
y = y.reshape(-1, 1)  # y를 2차원으로 변환

from sklearn.linear_model import LinearRegression  # 선형회귀

model = LinearRegression()  # 선형회귀 모델 생성
model.fit(x, y)  # 선형회귀 모델 학습
print(model.coef_, model.intercept_)  # 기울기, 절편 출력
print(model.score(x, y))  # 결정계수 출력

plt.rc('font', family='NanumGothic')  # 한글 폰트 설정
plt.rc('axes', unicode_minus=False)  # 한글 폰트 설정
plt.scatter(x, y)  # 산점도 그리기
plt.plot(x, model.predict(x))  # 선형회귀 직선 그리기
plt.show()  # 그래프 출력
