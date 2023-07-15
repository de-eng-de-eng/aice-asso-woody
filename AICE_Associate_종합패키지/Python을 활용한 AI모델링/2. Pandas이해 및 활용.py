#!/usr/bin/env python
# coding: utf-8

# # Chapter 2. Pandas 이해 및 활용

# ## DataFrame 살펴보기

# # __1. DataFrame이 뭔가요?__
# 
# >  - DataFrame은 2차원(col과 row을 가짐)테이블 데이터 구조를 가지는 자료형
# >  - Data Analysis, Machine Learning에서 data 변형을 위해 가장 많이 사용
# >  - **주의** : Series나 DataFrame은 대소문자가 구분되므로 Series, DataFrame으로 사용

# In[1]:


# pandas import
import pandas as pd
import numpy as np
# !pip install Ipython
from IPython.display import Image


# ### <b>1-1. DataFrame 만들어 보기</b>

# > Dictionary 형으로 생성 

# In[2]:


a1 = pd.DataFrame({"a" : [1,2,3], "b" : [4,5,6], "c" : [7,8,9]})
a1


# >  List 형태로 데이터 프레임 생성

# In[3]:


a2 = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], ["a","b","c"])
a2


# ### <b>1-2. 파일을 읽어서 DataFrame생성하기</b>
# > - pandas.read_csv 함수 사용
# >  - 대부분의 업무에서는 분석하고자 하는 Datat가 존재할 것
# >  - 이를 읽어 들이는 것부터 데이터 분석의 시작!
# >  - 이번 실습에서 읽을 파일 : sc_cust_info_txn_v1.5.csv

# In[4]:


# kt 데이터 파일을 활용
# 파일을 수정하고 저장 자체를 MS Office에서 하여서 encoding을 cp949로 해주어야 함
cust = pd.read_csv('./sc_cust_info_txn_v1.5.csv', encoding = "cp949")
cust


# - DataFrame 데이터 살펴보기
# >DataFrame의 구조 (인덱스와 컬럼)
#  - 인덱스(Index) : 행의 레이블에 대한 정보를 보유하고 있음
#  - 컬럼(Columns) : 열의 레이블에 대한 정보를 보유하고 있음
#  - 인덱스와 컬럼 자체는 중복값일 수 없음

# In[5]:


cust


# ### <b>1-3. 데이터 살펴보기</b>
# - head, tail 함수사용하기
# > - 데이터 전체가 아닌, 일부(처음부터, 혹은 마지막부터)를 간단히 보기 위한 함수 (default: 5줄)
#  - **head, tail을 왜 사용할까?**
#    - 광대한 데이터를 다룰 수 있는 Pandas의 특성상 특정변수에 제대로 데이터가 들어갔는지 간략히 확인
#    - 데이터 자료형의 확인
#    - 각 레이블에 맞는 데이터 매칭 확인

# In[6]:


# 상위 3개
cust.head(n=3)


# In[7]:


#하위 10개 
cust.tail(n=10)


# - DataFrame 기본 함수로 살펴보기
# > - **shape :** 속성 반환값은 튜플로 존재하며 row과 col의 개수를 튜플로 반환함(row,col순)
#   - columns : 해당 DataFrame을 구성하는 컬럼명을 확인할 수 있음
#   - **info**  : 데이터 타입, 각 아이템의 개수 등 출력
#   - describe : 데이터 컬럼별 요약 통계량을 나타냄, 숫자형 데이터의 통계치 계산
#             (count:데이터의 갯수 / mean:평균값 / std:표준편차 / min:최소값 / 4분위 수 / max:최대값) 
#   - dtypes : 데이터 형태의 종류(Data Types)

# In[8]:


# shape : 데이터를 파악하는데 중요함
cust.shape


# In[9]:


# DataFrame의 columns들을 보여줌
cust.columns


# In[10]:


# 데이터 타입 및 각 아이템등의 정보를 보여줌
cust.info()


# In[11]:


# DataFrame의 기본적인 통계정보를 보여줌
cust.describe()


# In[12]:


# DataFrame의 데이터 종류
cust.dtypes


# ### <b>1-4. read_csv 함수 파라미터 살펴보기</b>
# > - 함수에 커서를 가져다 두고 shift+tab을 누르면 해당 함수의 parameter 볼 수 있음
# > - sep - 각 데이터 값을 구별하기 위한 구분자(separator) 설정 
# > - index_col : index로 사용할 column 설정
# > - usecols : 실제로 dataframe에 로딩할 columns만 설정
# > - usecols은 index_col을 포함하여야 함

# In[13]:


cust2 = pd.read_csv('./sc_cust_info_txn_v1.5.csv', index_col='cust_class', usecols=['cust_class', 'r3m_avg_bill_amt', 'r3m_B_avg_arpu_amt', 'r6m_B_avg_arpu_amt'])
cust2


# # __2. Data 조회하기__
# DataFrame에서 data를 __조회, 수정__해보고 이를 이해해본다. 

# ### <b>1-1. 데이터 추출하기</b>

# #### <b>1) column 선택하기</b>
# 
# > - 기본적으로 [ ]는 column을 추출 : 특정한 col을기준으로 모델링을 하고자 하는 경우
# > - 컬럼 인덱스일 경우 인덱스의 리스트 사용 가능
# >   - 리스트를 전달할 경우 결과는 Dataframe
# >   - 하나의 컬럼명을 전달할 경우 결과는 Series

# #### <b>2) 하나의 컬럼 선택하기</b>
# 
# > - Series 형태로 가지고 올 수도, DataFrame형태로 가지고 올 수 있음

# In[14]:


cust.cust_class = cust['cust_class']


# In[15]:


cust.cust_class


# In[16]:


# cf : series 형태로 가지고 오기(hcust.cust_class = cust['cust_class'])
cust['cust_class']


# In[17]:


# cf : Dataframe형태로 가지고 오기
cust[['cust_class']]


# #### <b>3) 복수의 컬럼 선택하기</b>

# In[18]:


# 'cust_class' , 'age' 'r3m_avg_bill_amt'등 3개의 col 선택하기
cust[['cust_class', 'age', 'r3m_avg_bill_amt']]


# #### <b>4) DataFrame slicing</b>
# 
# >- 특정 **행 범위**를 가지고 오고 싶다면 [ ]를 사용
# >- DataFrame의 경우 기본적으로 [ ] 연산자가 **column 선택**에 사용되지만 **slicing은 row 레벨**로 지원

# In[19]:


# 7,8,9행을 가지고 옴 (인덱스 기준)
cust[7:10]


# #### <b>5) row 선택하기</b>
# 
#  - DataFrame에서는 기본적으로 **[ ]을 사용하여 column을 선택**
#  > __row 선택(두가지 방법이 존재)__
#  > - **loc** : Dataframe에 존재하는 **인덱스를 그대로 사용** (인덱스 기준으로 행 데이터 읽기)
#  > - **iloc** : Datafrmae에 존재하는 인덱스 상관없이 **0 based index로 사용** (행 번호 기준으로 행 데이터 읽기)
#  > - 이 두 함수는 ,를 사용하여 column 선택도 가능
# 

# In[20]:


cust.info()


# In[21]:


# arange함수는 10부터 19에서 끝나도록 간격을 1로 반환한다.
cp=np.arange(10,20)
cp


# In[22]:


#index를 100부터 달아주기 
cust.index = np.arange(100, 10030)
cust


# In[23]:


cust.tail()


# In[24]:


#한개의 row만 가지고 오기
cust.loc[[289]]


# In[25]:


#여러개의 row 가지고 오기
cust.loc[[102, 202, 302]]


# In[26]:


#iloc과비교(위와 같은 값을 가지고 오려면...)    (직접 타이핑 해보세요)
cust.iloc[[2, 102, 202]]


# - row, column 동시에 선택하기
#  > loc, iloc 속성을 이용할 때, 콤마를 이용하여 row와 col 다 명시 가능

# In[27]:


# 100, 200, 300 대상으로 cust_class, sex_type, age, r3m_avg_bill_amt, r3m_A_avg_arpu_amt  col 가지고 오기(loc사용)
cust.loc[[100, 200, 300], ['cust_class', 'sex_type', 'age', 'r3m_avg_bill_amt', 'r3m_A_avg_arpu_amt']]   # row, col


# In[28]:


# 같은 형태로 iloc사용하기 (index를 level로 가지고 오기)
# 100, 200, 300 대상으로 cust_class, sex_type, age, r3m_avg_bill_amt, r3m_A_avg_arpu_amt  col 가지고 오기(iloc사용)
cust.iloc[[0, 100, 200], [3, 4, 5, 9, 10]]


# In[29]:


# 100, 200, 300 대상으로 cust_class, sex_type, age, r3m_avg_bill_amt, r3m_A_avg_arpu_amt  col 가지고 오기(loc사용 : 에러발생 함) 
cust.loc[[100, 200, 300], [3, 4, 5, 9, 10]]   # row, col


# #### __6) boolean selection 연산으로 row 선택하기 (= 컬럼 조건문으로 행 추출하기)__
# 
#  - 해당 조건에 맞는 row만 선택
#  - 조건을 명시하고 조건을 명시한 형태로 inedxing 하여 가지고 옴

#  - ex: 남자이면서 3개월 평균 청구 금액이 50000 이상이면서 100000 미만인 사람만 가지고오기 

# In[30]:


#조건을 전부다  [ ]안에 넣어 주면 됨
extract = cust[(cust['sex_type']=='M') & (cust['r3m_avg_bill_amt']>=50000) & (cust['r3m_avg_bill_amt']< 100000)]
extract.head()


# In[31]:


# 조건문이 너무 길어지거나 복잡해지면...아래와 같은 방식으로 해도 무방함
# 남자이면서 
sex = cust['sex_type']=='M'
# 3개월 평균 청구 금액이 50000 이상이면서 100000 미만
bill = (cust['r3m_avg_bill_amt']>=50000) & (cust['r3m_avg_bill_amt']< 100000)

cust[sex & bill].head()


# #### <b>7) 정리 </b>
# 
# - 기본적인 대괄호는 col을 가지고 오는 경우 사용, 하지만 slicing은 row를 가지고 온다.
# - row를 가지고 오는 경우는 loc과 iloc을 사용하는데, loc과 iloc은 컬럼과 row를 동시에 가지고 올 수 있다.

# In[32]:


import matplotlib.pyplot as plt      #matplotlib.pyplot import   


# ### <b>1-2. 데이터 추가하기</b>

# #### <b>1) 새 column 추가하기</b>
# 
# >- 데이터 전처리 과정에서 빈번하게 발생하는 것
# >- insert 함수 사용하여 원하는 위치에 추가하기

# In[33]:


# r3m_avg_bill_amt 두배로 새로운 col만들기
cust['r3m_avg_bill_amt2'] = cust['r3m_avg_bill_amt'] * 2
cust.head()


# In[34]:


# 기존에 col을 연산하여 새로운 데이터 생성
cust['r3m_avg_bill_amt3'] = cust['r3m_avg_bill_amt2'] + cust['r3m_avg_bill_amt']
cust.head()


# In[35]:


# 새로은 col들은 항상맨뒤에 존재 원하는 위치에 col을 추가하고자 하는 경우
# 위치를 조절 하고 싶다면(insert함수 사용)
cust.insert(10, 'r3m_avg_bill_amt10', cust['r3m_avg_bill_amt'] *10)  # 0부터 시작하여 10번째 col에 insert
cust.head()


# #### __2) column 삭제하기__
# 
# >- drop 함수 사용하여 삭제
# >- axis는 삭제를 가로(행)기준으로 할 것인지, 세로(열)기준으로 할 것인지 명시하는 'drop()'메소드의 파라미터임 
# >- 리스트를 사용하면 멀티플 col 삭제 가능

# In[36]:


# axis : dataframe은 차원이 존재 함으로 항상 0과 1이 존재 
# (0은 행레벨, 1을 열 레벨)
cust.drop('r3m_avg_bill_amt10', axis=1)


# In[37]:


#원본 데이터를 열어 보면 원본 데이터는 안 지워진 상태
cust.head()


# In[38]:


# 원본 데이터를 지우고자 한다면... 
# 방법1 : 데이터를 지우고 다른 데이터 프레임에 저장
cust1=cust.drop('r3m_avg_bill_amt10', axis=1)
cust1.head()


# In[39]:


# 원본 자체를 지우고자 한다면...
# 방법 2 : inplace 파라미터를 할용 True인 경우 원본데이터에 수행
cust.drop('r3m_avg_bill_amt10', axis=1, inplace=True)


# In[40]:


# 원본확인
cust


# <br><br>

# # Chapter 2. Pandas 이해 및 활용

# ## DataFrame 변형하기

# # __1. group by 이해하기__
# 

# ### <b>1-1. 데이터 묶기</b>

# In[41]:


# pandas import
import pandas as pd
import numpy as np
# !pip install Ipython
from IPython.display import Image


# #### <b>1) 그룹화(groupby)</b>
# 
#   + 같은 값을 하나로 묶어 통계 또는 집계 결과를얻기위해 사용하는 것
#   + 아래의 세 단계를 적용하여 데이터를 그룹화(groupping) / 특정한 col을 기준으로 데이터를 그룹핑 하여 통계에 활용하는 것
#     - 데이터 분할(split) : 어떠한 기준을 바탕으로 데이터를 나누는 일
#     - operation 적용(applying) : 각 그룹에 어떤 함수를 독립적으로 적용시키는 일
#     - 데이터 병합(cobine) : 적용되어 나온 결과들을 통합하는 일
#   + 데이터 분석에 있어 사용빈도가 높음
#   + groupby의 결과는 dictionary형태임
# 

# In[42]:


Image("groupby_agg.jpg")


# In[43]:


cust = pd.read_csv('./sc_cust_info_txn_v1.5.csv', encoding = "cp949")
cust


# > ##### __1-1) groupby의 groups 속성__
# > - 각 그룹과 그룹에 속한 index를 dict 형태로 표현

# In[44]:


# 파라미터 값으로 col의 리스트나 col을 전달
# 출력은 우선 dataframe이라고 하는 객체임(그룹을 생성까지 한 상태)

gender_group = cust.groupby('sex_type')
gender_group


# In[45]:


# groups를 활용하여 그룹의 속성을 살펴보기
gender_group.groups


# >##### __1-2) groupby 내부 함수 활용하기__
#  - 그룹 데이터에 적용 가능한 통계 함수(NaN은 제외하여 연산)
#  - count : 데이터 개수
#  - size : 집단별 크기
#  - sum  : 데이터의 합
#  - mean, std, var : 평균, 표준편차, 분산
#  - min, max : 최소, 최대값

# In[46]:


# count 함수 확인
gender_group.count()


# In[47]:


# mean 함수 확인
gender_group.mean()


# In[48]:


# max값 확인하기
gender_group.max()


# In[49]:


# 특정 col만 보는 경우 : gender별 r3m_avg_bill_amt의 평균
gender_group.mean()[['r3m_avg_bill_amt']]


# > ##### __1-3) 인덱스 설정(groupby) 후 데이터 추출하기__
# - 성별 r3m_avg_bill_amt의 평균

# In[50]:


# groupby한 상태에서 가지고 오는 경우 
gender_group.mean()[['r3m_avg_bill_amt']]


# In[51]:


# groupby하기 전 원 DataFrame에서 가지고 오는 경우(같은 의미)
cust.groupby('sex_type').mean()[['r3m_avg_bill_amt']]


# >##### __1-4) 복수 columns을 기준으로 Groupping 하기__
# >- groupby에 column 리스트를 전달할 수 있고 복수개의 전달도 가능함
# >- 통계함수를 적용한 결과는 multiindex를 갖는 DataFrame

# In[52]:


# cust_class 와 sex_type으로 index를 정하고 이에따른 r3m_avg_bill_amt의 평균을 구하기
cust.groupby(['cust_class', 'sex_type']).mean()[['r3m_avg_bill_amt']]


# In[53]:


# 위와 동일하게 groupby한 이후에 평균 구하기
multi_group=cust.groupby(['cust_class', 'sex_type'])
multi_group.mean()[['r3m_avg_bill_amt']]


# In[54]:


# INDEX는 DEPTH가 존재함 (loc을 사용하여 원하는 것만 가지고 옴)
cust.groupby(['cust_class', 'sex_type']).mean().loc[[("D","M")]]


# > ##### __1-5) index를 이용한 group by__
# > - index가 있는 경우, groupby 함수에 level 사용 가능
# > - level은 index의 depth를 의미하며, 가장 왼쪽부터 0부터 증가
# > - **set_index** 함수
# >  - column 데이터를 index 레벨로 변경하는 경우 사용
# >  - 기존의 행 인덱스를 제거하고 데이터 열 중 하나를 인덱스로 설정
# > - **reset_index** 함수 
# >  - 인덱스 초기화
# >  - 기존의 행 인덱스를 제거하고 인덱스를 데이터 열로 추가

# In[55]:


# heat DataFrame 다시 한번 확인 합니다.
cust.head()


# > ##### __1-6) MultiIndex를 이용한 groupping__

# In[56]:


# set_index로 index셋팅(멀티도 가능)
cust.set_index(['cust_class','sex_type'])


# In[57]:


# reset_index활용하여 기존 DataFrame으로 변환  (set_index <-> reset_index)   
cust.set_index(['cust_class','sex_type']).reset_index()


# In[58]:


# 멀티 인덱스 셋팅 후 인덱스 기준으로 groupby하기
# 'sex'와 'cp'를 기준으로 index를셋팅하고 index를 기준으로 groupby하고자 하는경우
# groupby의 level은 index가 있는 경우에 사용 
cust.set_index(['cust_class','sex_type']).groupby(level=[0]).mean()


# In[59]:


cust.set_index(['cust_class','sex_type']).groupby(level=[0,1]).mean()


# > ##### __1-7) aggregate(집계) 함수 사용하기__
# > groupby 결과에 집계함수를 적용하여 그룹별(mean, max등) 데이터 확인 가능

# In[60]:


#그룹별로 한번에 데이터를 한번에 보는 경우
cust.set_index(['cust_class','sex_type']).groupby(level=[0,1]).aggregate([np.mean, np.max])


# # __2. pivot / pivot_table 함수 활용__

# ### <b>2-1. pivot 은 뭘까?</b>
# > - dataframe의 형태를 변경
# > - 여러 분류로 섞인 **행 데이터를 열 데이터로 회전** 시키는 것
# >   - pivot의 사전적의미 : (축을 중심으로)회전하다, 회전시키다. 
# > - pivot형태 : pandas.pivot(index, columns, values) 로 사용할 컴럼을 명시

# In[61]:


import numpy as np
import pandas as pd


# In[62]:


data = pd.DataFrame({'cust_id': ['cust_1', 'cust_1', 'cust_1', 'cust_2', 'cust_2', 'cust_2', 'cust_3', 'cust_3', 'cust_3'],
                  'prod_cd': ['p1', 'p2', 'p3', 'p1', 'p2', 'p3', 'p1', 'p2', 'p3'],
                  'grade' : ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
                  'pch_amt': [30, 10, 0, 40, 15, 30, 0, 0, 10]})
data


# In[63]:


# 행(row)는 고객ID(cust_id), 열(col)은 상품코드(prod_cd), 값은 구매금액(pch_amt)을 pivot릏 활용하여 만들어보기
# (직접 타이핑해보세요)
data.pivot(index = 'cust_id', columns ='prod_cd', values ='pch_amt')


# In[64]:


data.pivot('cust_id', 'prod_cd', 'pch_amt')


# ### <b>2-2. Pivot_table은 뭘까?</b>
# > pivot_table형태 : pandas.pivot_table(data, index, columns, aggfunc)

# In[65]:


# pivot_table을 활용하여 위와 동일하게 만들기
data.pivot_table(index = 'cust_id', columns ='prod_cd', values ='pch_amt')


# ### <b>2-3. Pivot과 Pivot_table의 차이는뭘까 ?</b>
# 
# > pivot은 안되고 pivot_table만을 사용해야 하는 경우가 있음

# #### <b>1) index가 2개 이상인 경우</b>

# <font color=red><b>실습 VOD 동영상에서는 아래 코드 실행시 에러가 발생하지만</b></font><br>
# <font color=blue><b>저희 실습은 Pandas 버젼업 되어 에러없이 아래 코드가 실행됩니다.</b></font>

# In[66]:


data.pivot(index = ['cust_id','grade'], columns ='prod_cd', values ='pch_amt')


# In[67]:


data.pivot_table(index = ['cust_id','grade'], columns ='prod_cd', values ='pch_amt')


# #### <b>2) columns가 2개 이상인 경우</b>
# 

# <font color=red><b>실습 VOD 동영상에서는 아래 코드 실행시 에러가 발생하지만</b></font><br>
# <font color=blue><b>저희 실습은 Pandas 버젼업 되어 에러없이 아래 코드가 실행됩니다.</b></font>

# In[68]:


data.pivot(index = 'cust_id', columns =['grade','prod_cd'], values ='pch_amt')


# In[69]:


data.pivot_table(index = 'cust_id', columns =['grade','prod_cd'], values ='pch_amt')


# #### <b>3) 중복 값이 있는 경우</b>
# 
# > - pivot은 중복 값이 있는 경우 valueError를 반환함
# > - pivot_table은 aggregation 함수를 활용하여 처리

# In[70]:


#index로 쓰인 grade가 중복이 있음
data.pivot(index='grade', columns='prod_cd', values='pch_amt')  


# In[71]:


#index로 쓰인 grade가 중복이 있음
data.pivot_table(index='grade', columns='prod_cd', values='pch_amt')  


# ### <b>2-4. pivot_table의 추가 살펴 보기</b>
# 
# >pivot_table은 aggregation 함수를 활용하여 처리

# In[72]:


# aggfunc를 sum으로 구하기 (직접 타이핑 해보세요)
data.pivot_table(index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.sum)


# In[73]:


# 위와 같은결과(참고로 알아 두셔요)
pd.pivot_table(data, index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.sum)


# In[74]:


# aggfunc를 mean으로 구하기(default가 mean임) (직접 타이핑 해보세요)
data.pivot_table(index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.mean)


# # __3. stack, unstack 함수 활용__

# In[75]:


df = pd.DataFrame({
    '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
    '요일': ['월요일', '화요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
    '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
    '강수확률': [80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]})

df


# ### <b>3-1. stack & unstack</b>
# 
# > - **stack** : 컬럼 레벨에서 인덱스 레벨로 dataframe 변경
# >  - 즉, 데이터를 row 레벨로 쌓아올리는 개념으로 이해하면 쉬움
# > - **unstack** : 인덱스 레벨에서 컬럼 레벨로 dataframe 변경
# >  - stack의 반대 operation
# > - 둘은 역의 관계에 있음

# In[76]:


# '지역'과 '요일'두개로 인덱스를 설정하고 별도의 DataFrame으로 설정 하기 - (직접 타이핑 해보세요)
new_df = df.set_index(['지역', '요일'])
new_df


# In[77]:


# 첫번째 레벨의 인덱스(지역)를 컬럼으로 이동 / 인덱스도 레벨이 있음 
new_df.unstack(0)


# In[78]:


# 두번째 레벨의 인덱스를 컬럼으로 이동
new_df.unstack(1)


# In[79]:


new_df


# In[80]:


# new_df.unstack(0)상태에서 첫번째 레벨의 컬럼(강수량과 강수확률)을 인덱스로 이동(stack(0))
new_df.unstack(0).stack(0)


# In[81]:


new_df.unstack(0).stack(1)


# <br><br>

# # Chapter 2. Pandas 이해 및 활용

# ## DataFrame 병합하기

# # __1. concat함수 활용__
# 
# 두 개 이상의 데이터프레임을 하나로 합치는 데이터 병합(merge)이나 연결(concatenate)을 지원합니다. 

# In[83]:


# pandas import
import pandas as pd
import numpy as np
# !pip install Ipython
from IPython.display import Image


# ### <b>1-1. concat 함수 사용하여 DataFrame 병합하기</b>
# 
# > - pandas.concat 함수  (배열결합 : concatenate)
# >  - 데이터의 속성 형태가 동일한 데이터 셋 끼리 합칠때 사용 (DataFrame을 물리적으로 붙여주는 함수)
# > - 열 or 행 레벨로 병합하는 것

# #### <b>1) column명이 같은 경우</b>
# >  ignore_index, axis 활용

# In[84]:


df1 = pd.DataFrame({'key1' : [0,1,2,3,4], 'value1' : ['a', 'b', 'c','d','e']}, index=[0,1,2,3,4])
df2 = pd.DataFrame({'key1' : [3,4,5,6,7], 'value1' : ['c','d','e','f','g']}, index=[3,4,5,6,7])


# In[85]:


df1


# In[86]:


df2


# > ##### __concat함수 옵션__
# 
# > - **ignore_index** : 기존 index를 무시하고자 하는 경우
# >   - False : 기존 index유지(default) / True : 기존 index무시(index재배열)
# > - **axis**
# >   - 0 : 위+아래로 합치기(row레벨) / 1 : 왼쪽+오른쪽으로 합치기(col레벨)

# In[87]:


# ignore_index에 대한이해  (직접 타이핑 해보세요)
pd.concat([df1, df2], ignore_index=False)


# In[89]:


# ignore_index에 대한이해  (직접 타이핑 해보세요)
pd.concat([df1, df2], ignore_index=True)


# In[88]:


# axis=0,1 비교해 보기  (직접 타이핑 해보세요)
pd.concat([df1, df2], axis =1)


# In[90]:


# axis=0,1 비교해 보기  (직접 타이핑 해보세요)
pd.concat([df1, df2], axis =0)


# #### <b>2) column명이 다른 경우</b>
# 
# > * concat함수중에 join에 대한이해
# > * join 방식은 outer의 경우 합집합, inner의 경우 교집합을 의미

# In[91]:


df3 = pd.DataFrame({'a':['a0','a1','a2', 'a3'], 'b':['b0','b1','b2','b3'], 'c':['c0','c1','c2','c3']}, index = [0,1,2,3])
df4 = pd.DataFrame({'a':['a2','a3','a4', 'a5'], 'b':['b2','b3','b4','b5'], 'c':['c2','c3','c4','c5'], 'd':['d1','d2','d3','d4']}, index = [2,3,4,5])


# In[92]:


df3


# In[93]:


df4


# In[94]:


pd.concat([df3, df4], join='outer')


# In[95]:


pd.concat([df3, df4], join='inner')


# #### <b>3) index 중복 여부 확인</b>
# 
# > * concat함수중에 verify_integrity에 대한 이해
# > * verify_integrity=False가 default임으로 error발생을 하지 않음
# > * verify_integrity=True인 경우 error 발생

# In[96]:


df5 = pd.DataFrame({'A':['A0','A1','A2'], 'B':['B0','B1','B2'], 'C':['C0','C1','C2'], 'D':['D0','D1','D2']}, index=['I0','I1','I2'])
df6 = pd.DataFrame({'A':['AA2','A3','A4'], 'B':['BB2','B3','B4'], 'C':['CC2','C3','C4'], 'D':['DD2','D3','D4']}, index=['I2','I3','I4'])


# In[97]:


df5


# In[98]:


df6


# In[99]:


pd.concat([df5, df6], verify_integrity=False)


# In[100]:


# index중복이있는 경우 error가 남
pd.concat([df5, df6], verify_integrity=True)


# # __2. merge & join 함수 활용__
# 

# ### <b>2-1. DataFrame merge</b>
# 
# > - Database의 Table들을 Merge/Join하는 것과 유사함
# > - 특정한 column(key)을 기준으로 병합
# >   - join 방식: how 파라미터를 통해 명시(특정한 col을 바탕으로 join 하는 것)
#      - inner: 기본 merge방법, 일치하는 값이 있는 경우 (Merge할 테이블의 데이터가 모두 있는 경우만 가지고 옴)
#      - left: left outer join (왼쪽을 기준으로 오른쪽을 채움 - 오른쪽에 데이터 없으면 NaN)
#      - right: right outer join
#      - outer: full outer join  (Left와 Right를 합한 것)

# In[101]:


customer = pd.DataFrame({'cust_id' : np.arange(6), 
                    'name' : ['철수', '영희', '길동', '영수', '수민', '동건'], 
                    '나이' : [40, 20, 21, 30, 31, 18]})

orders = pd.DataFrame({'cust_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9], 
                    'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'], 
                    'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})


# In[102]:


customer


# In[103]:


orders


# #### <b>1) merge함수의 on 옵션</b>
# 
# >  - join 대상이 되는 column 명시

# In[104]:


# 기본적인 Merge방식은 inner임 (직접 타이핑 해보세요)
# customer의 cust_id 5번과 orders의 cust_id 9번이 없는 것을 확인
pd.merge(customer, orders, on='cust_id')  


# In[105]:


# merge하고자 하는 컬럼 명칭을 on에 명시한다. 
# 여러개인 경우 리스트  일치하는 것만 가지고 옴
pd.merge(customer, orders, on='cust_id', how='inner') 


# In[106]:


# 왼쪽 테이블을 기준으로 Merge (여기서는 customer기준)
pd.merge(customer, orders, on='cust_id', how='left')


# In[107]:


# 오른쪽 테이블을 기준으로 Merge (여기서는 orders기준)
pd.merge(customer, orders, on='cust_id', how='right')


# In[108]:


# outer : Left와 Right를 합친 것
pd.merge(customer, orders, on='cust_id', how='outer')


# #### <b>2) index 기준으로 join하기</b>
# 
# 

# In[109]:


#cust_id를 기준으로 인덱스 생성하기 (set_index활용)  (직접 타이핑 해보세요)
cust1 = customer.set_index('cust_id')
order1 = orders.set_index('cust_id')


# In[110]:


cust1


# In[111]:


order1


# In[112]:


# on을 명시할 필요 없이 index를 merge 하고자 하는 경우 
pd.merge(cust1, order1, left_index=True, right_index=True)
#inner와 동일한 형태


# #### <b>연습문제1) 가장 많이 팔린 아이템은?</b>
#   - Hint : merge, groupby, sort_values 활용
#   - 아이템이 중요함으로 orders DF이 중요

# In[113]:


# 1. customer, orders 를 merge (how는?)
# 2. group_by이용하여 item 을 grouping후 sum
pd.merge(customer, orders, on='cust_id', how='right').groupby('item').sum()


# In[114]:


# 3. sort_values를 이용하여 quantity를기준으로 sort + 내림차순으로 정렬
pd.merge(customer, orders, on='cust_id', how='right').groupby('item').sum().sort_values(by='quantity', ascending=False)


# #### <b>연습문제2) 영희가 가장 많이 구매한 아이템은?</b>
#    - 1.우선 사람과 아이템 별로 sum을 해서(groupby시 "이름"과 "아이템"기준으로 합을 구하고)
#    - 2.loc을 활용하여 (영희의 row의 quantity만 확인)

# In[115]:


pd.merge(customer, orders, on='cust_id', how='inner').groupby(['name', 'item']).sum().loc['영희']


# ### <b>2-2. join 함수</b>
# 
#  - index가 있는 경우 사용(행 인덱스를 기준으로 결합)
#  - 내부적으로 pandas.merge 함수를 기반으로 만들어짐
#  - 기본적으로 index를 사용하여 left join
#  - 형태 : Dataframe1.join(Dataframe2. how='left')

# In[116]:


cust1.join(order1, how='inner')

