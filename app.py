import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# -----------------------------------------------------------------------------
# 1. 환경 설정 (한글 폰트)
# -----------------------------------------------------------------------------
# Streamlit Cloud(리눅스)와 로컬(윈도우) 환경 모두 대응
if platform.system() == 'Linux':
    plt.rc('font', family='NanumGothic') # Streamlit Cloud 기본 한글 폰트
else:
    plt.rc('font', family='Malgun Gothic') # 윈도우 로컬 테스트용
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링 (Back-end Logic)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생산량 예측 AI", layout="wide")

# (1) 데이터 준비 
# [실제 운영 시] 엑셀 파일을 같은 폴더에 넣고 아래 주석을 해제해서 쓰세요.
# df = pd.read_excel("참치생산지표3.xlsx") 

# [현재 배포용] 데모 데이터 생성 (엑셀 없이도 작동되게 함)
np.random.seed(123)
n = 30
df = pd.DataFrame({
    'production': np.random.normal(100, 10, n),
    'yield': np.random.uniform(80, 95, n),
    'productivity': np.random.uniform(1.0, 2.0, n),
    'workforce': np.random.choice(range(40, 61), n),
    'hour': np.random.choice(range(160, 201), n)
})

# (2) 전처리: 이상치 제거 로직 (R 코드의 [-c(17,20,23)] 반영)
# Python 인덱스는 0부터 시작하므로 16, 19, 22를 제거
drop_indices = [16, 19, 22]
df_clean = df.drop(drop_indices, errors='ignore').reset_index(drop=True)

# (3) 모델 학습 (OLS 회귀분석)
X = df_clean[['yield', 'productivity', 'workforce', 'hour']]
y = df_clean['production']
X = sm.add_constant(X) # 상수항 추가
model = sm.OLS(y, X).fit()

# -----------------------------------------------------------------------------
# 3. 사용자 인터페이스 (Front-end)
# -----------------------------------------------------------------------------
st.title("🐟 참치 생산 실적 예측 시뮬레이터")
st.markdown("과거 28개월 데이터를 기반으로 **투입 조건에 따른 예상 생산량**을 산출합니다.")
st.divider()

col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("🛠️ 생산 조건 입력")
    st.info("오늘의 작업 계획을 입력하세요.")
    
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce, 명)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour, 시간)", 160, 200, 180, step=1)
    
    with st.expander("모델 통계 상세 보기 (Summary)"):
        st.code(str(model.summary()))

with col_result:
    st.subheader("📈 AI 예측 결과")
    
    # 입력값으로 예측 수행
    input_data = pd.DataFrame({
        'const': 1.0,
        'yield': [input_yield], 
        'productivity': [input_prod], 
        'workforce': [input_wf], 
        'hour': [input_hour]
    })
    
    # 예측 및 신뢰구간 계산
    predictions = model.get_prediction(input_data)
    pred_df = predictions.summary_frame(alpha=0.05) # 95% 신뢰구간
    
    pred_val = pred_df['mean'][0]
    lower_val = pred_df['obs_ci_lower'][0]
    upper_val = pred_df['obs_ci_upper'][0]
    
    # 핵심 지표 표시
    m1, m2, m3 = st.columns(3)
    m1.metric("최소 예상 (보수적)", f"{lower_val:.1f} 톤")
    m2.metric("🎯 예측 생산량", f"{pred_val:.1f} 톤", delta="Target")
    m3.metric("최대 예상 (긍정적)", f"{upper_val:.1f} 톤")
    
    # 그래프 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 바 차트 (예측값)
    ax.bar(['예상 생산량'], [pred_val], color='#2ecc71', alpha=0.7, width=0.3)
    
    # 에러바 (신뢰구간)
    ax.errorbar(['예상 생산량'], [pred_val], 
                yerr=[[pred_val - lower_val], [upper_val - pred_val]], 
                fmt='o', color='red', ecolor='gray', elinewidth=3, capsize=10, 
                label='95% 예측 신뢰구간')
    
    # 텍스트 및 레이블
    ax.text(0, pred_val + 2, f"{pred_val:.1f} 톤", ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(lower_val * 0.8, upper_val * 1.2)
    ax.set_ylabel('생산량 (톤)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    
    st.pyplot(fig)
