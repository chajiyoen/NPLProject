import pandas as pd
import urllib.request #인터넷을 이용하여 데이터 요청
import matplotlib.pyplot as plt
import re #정규 표현
from konlpy.tag import Okt # 한국어 불용어 처리시
from tqdm import tqdm #진행율 바 표기
import numpy as np

#seperate title [1. 리뷰파일 다운로드]  ===========
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

#seperate title [2.판다스로 데이터 확인 ]  ===========
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
print(train_data.info())
print(test_data.info())

# >>>> 출력결과
# >>>><class 'pandas.core.frame.DataFrame'>
# >>>>RangeIndex: 150000 entries, 0 to 149999
# >>>>Data columns (total 3 columns):
# >>>> #   Column    Non-Null Count   Dtype 
# >>>>---  ------    --------------   ----- 
# >>>> 0   id        150000 non-null  int64 
# >>>> 1   document  149995 non-null  object
# >>>> 2   label     150000 non-null  int64 
# >>>>dtypes: int64(2), object(1)
# >>>><class 'pandas.core.frame.DataFrame'>
# >>>>RangeIndex: 50000 entries, 0 to 49999
# >>>>Data columns (total 3 columns):
# >>>>---  ------    --------------  ----- 
# >>>>0   id        50000 non-null  int64 
# >>>> 1   document  49997 non-null  object
# >>>> 2   label     50000 non-null  int64 
# >>>>dtypes: int64(2), object(1)
# >>>>memory usage: 1.1+ MB
# >>>>None

# *개발자 분석 내용: document 필드의 수량이 다른 필드의 수량과 틀리므로 결측 데이터가 존재한다.

#seperate title [3.결측데이터 수량 확인 및 제거]  ===========
print("훈련데이터 결측수량:",train_data["document"].isna().sum())
print("테스트데이터 결측수량:",test_data["document"].isna().sum())
train_data = train_data.dropna(axis=0,subset="document")
test_data = test_data.dropna(axis=0,subset="document")
print("훈련데이터 결측수량:",train_data["document"].isna().sum())
print("테스트데이터 결측수량:",test_data["document"].isna().sum())


#>>> 출력결과
훈련데이터 결측수량: 5
테스트데이터 결측수량: 3
훈련데이터 결측수량: 0
테스트데이터 결측수량: 0

# *개발자 분석 내용: 최초에 훈련데이터에 5개의 결측데이턱 관측되었고
#                  테스트데이터에 3개의 결측데이터가 관측되어
#                  pandas 의 dropna 명령으로 제거하였다. 

#seperate title [4.중복데이터 확인 및 제거]  ===========
print(train_data["document"].count()) #총데이터 수량
print(train_data["document"].nunique()) #유니크한 데이터의 수량
print("중복된 데이터의 수:",train_data["document"].count()-train_data["document"].nunique())
#>>> 149995
#>>> 146182
# *개발자 분석 내용: 총 데이터 수량과 유니크 데이터 수량의 차이가 있음은 
#                  중복된 데이터가 존재하고 있다.

print(test_data["document"].count()) 
print(test_data["document"].nunique())
print("중복된 데이터의 수:",test_data["document"].count()-test_data["document"].nunique())
# drop_duplicates() 동일한 값 제거
train_data=train_data.drop_duplicates(subset="document")
test_data=test_data.drop_duplicates(subset="document")
#>>>중복된 데이터의 수: 3813
#>>>49997
#>>>49157
#>>>중복된 데이터의 수: 840
# *개발자 분석 내용:테스트 데이터 또한 중복 내용이 존재하고 있으나 훈련 대상 데이터가 아니지만 중복데이터 제거를 하지 않겠다.

#ref 참조
print("중복된 데이터의 수:",train_data["document"].count()-train_data["document"].nunique())
print("중복된 데이터의 수:",test_data["document"].count()-test_data["document"].nunique())
#>>>중복된 데이터의 수: 0
#>>>중복된 데이터의 수: 0
# *개발자 분석 내용: 모든 데이터의 중복이 제거되어 중복데이터 수가 0으로 표기 되었다.


#seperate title [5. 한글을 제외한 문자제거와 형태소별로 분류]  ===========
print(train_data[:5])
#>>>id          0
#>>>document    0
#>>>label       0
#>>>dtype: int64
#>>>         id                                      document  label
#>>>0   9976970                           아 더빙.. 진짜 짜증나네요 목소리      0
#>>>1   3819312             흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
#>>>2  10265843                           너무재밓었다그래서보는것을추천한다      0
#>>>3   9045019             교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정      0
#>>>4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...      1

# *개발자 분석 내용: ...? 영문 등은 감성분석에 불필욯므로 제거 대상이다.
# 정규표현식을 이용한 한글과 공백을 제외한 모든 단어는 제거

train_data["document"]=\
  train_data["document"].replace(r"[^\sㄱ-ㅎㅏ-ㅣ가-힣]","",regex=True) #한글이 아닌것
print(train_data[:5])
#>>>         id                                       document  label
#>>>0   9976970                              아 더빙 진짜 짜증나네요 목소리      0
#>>>1   3819312                    흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나      1
#>>>2  10265843                             너무재밓었다그래서보는것을추천한다      0
#>>>3   9045019                   교도소 이야기구먼 솔직히 재미는 없다평점 조정      0
#>>>4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화스파이더맨에서 늙어보이기만 했던 커스틴 던...      1
test_data["document"]=\
  test_data["document"].replace(r"[^\sㄱ-ㅎㅏ-ㅣ가-힣]","",regex=True) #한글이 아닌것





