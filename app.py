import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# -----------------------------------------------------------------------------
# 1. 환경 설정 (✅ 한글 폰트 깨짐 해결 - 웹에서 폰트 다운로드 방식)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_korean_font():
    # 네이버 나눔고딕 폰트를 다운로드하여 적용합니다. (Streamlit Cloud 호환성 높음)
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    font_path = "NanumGothic-Regular.ttf"
    
    if not os.path.exists(font_path):
        import urllib.request
        urllib.request.urlretrieve(font_url, font_path)
        
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    return font_name

# 폰트 적용 및 마이너스 기호 깨짐 방지
font_name = get_korean_font()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링 (Back-end Logic)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생산량 예측 AI", layout="wide")

# (1) 데이터 준비 (데모용)
np.random.seed(123)
n = 30
df = pd.DataFrame({
    'production': np.random.normal(100, 10, n),
    'yield': np.random.uniform(80, 95, n),
    'productivity': np.random.uniform(1.0, 2.0, n),
    'workforce': np.random.choice(range(40, 61), n),
    'hour': np.random.choice(range(160, 201), n)
})

# (2) 전처리
drop_indices = [16, 19, 22]
df_clean = df.drop(drop_indices, errors='ignore').reset_index(drop=True)

# (3) 모델 학습
X = df_clean[['yield', 'productivity', 'workforce', 'hour']]
y = df_clean['production']
X = sm.add_constant(X)
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
    
    # 예측 수행
    input_data = pd.DataFrame({
        'const': 1.0,
        'yield': [input_yield], 
        'productivity': [input_prod], 
        'workforce': [input_wf], 
        'hour': [input_hour]
    })
    
    predictions = model.get_prediction(input_data)
    pred_df = predictions.summary_frame(alpha=0.05)
    
    pred_val = pred_df['mean'][0]
    lower_val = pred_df['obs_ci_lower'][0]
    upper_val = pred_df['obs_ci_upper'][0]
    
    # 핵심 지표 표시
    m1, m2, m3 = st.columns(3)
    m1.metric("최소 예상 (보수적)", f"{lower_val:.1f} 톤", help="95% 신뢰구간 하한값")
    m2.metric("🎯 예측 생산량", f"{pred_val:.1f} 톤", delta="Target", help="가장 유력한 예측값")
    m3.metric("최대 예상 (긍정적)", f"{upper_val:.1f} 톤", help="95% 신뢰구간 상한값")
    
    # ✅ 그래프 설명 추가
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>
        <strong>💡 그래프 해석 가이드:</strong><br>
        • <strong>초록색 막대:</strong> AI가 예측한 가장 가능성 높은 생산량입니다.<br>
        • <strong>빨간색 선(I):</strong> 95% 신뢰구간(안전 범위)입니다. 실제 생산량이 이 빨간 선 범위 내에 있을 확률이 높다는 것을 의미합니다. (하단 점: 최소치, 상단 점: 최대치)
    </div>
    """, unsafe_allow_html=True)

    # ✅ 그래프 크기 조절 (figsize 변경)
    fig, ax = plt.subplots(figsize=(10, 3)) # 높이를 5에서 3으로 줄임
    
    # 바 차트 (예측값)
    ax.bar(['예상 생산량'], [pred_val], color='#2ecc71', alpha=0.7, width=0.3)
    
    # 에러바 (신뢰구간)
    ax.errorbar(['예상 생산량'], [pred_val], 
                yerr=[[pred_val - lower_val], [upper_val - pred_val]], 
                fmt='o', color='red', ecolor='gray', elinewidth=3, capsize=10, 
                label='95% 예측 신뢰구간')
    
    # 텍스트 및 레이블
    ax.text(0, pred_val + (upper_val - lower_val)*0.05, f"{pred_val:.1f} 톤", ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(lower_val * 0.9, upper_val * 1.1) # Y축 범위 여백 조정
    ax.set_ylabel('생산량 (톤)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # ax.legend() # 범례는 설명 박스로 대체하여 주석 처리

    # 그래프 여백 조정 (꽉 차게)
    plt.tight_layout()
    
    st.pyplot(fig)
