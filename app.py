import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# -----------------------------------------------------------------------------
# 1. 환경 설정 (폰트 및 스타일)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_korean_font():
    font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    font_path = "NanumGothic-Regular.ttf"
    if not os.path.exists(font_path):
        import urllib.request
        urllib.request.urlretrieve(font_url, font_path)
    fm.fontManager.addfont(font_path)
    return fm.FontProperties(fname=font_path).get_name()

font_name = get_korean_font()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 설정 (깔끔한 디자인)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = font_name # Seaborn 적용 후 폰트 재설정

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생산량 예측 대시보드", layout="wide")

# (1) 데이터 준비
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
# 3. 사이드바 (입력 컨트롤) - 공간 절약
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 시뮬레이션 설정")
    st.info("조건을 변경하면 실시간으로 반영됩니다.")
    
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce, 명)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour, 시간)", 160, 200, 180, step=1)
    
    st.markdown("---")
    with st.expander("ℹ️ 모델 통계 정보"):
        st.caption(f"R-squared: {model.rsquared:.3f}")
        st.caption("Data Source: 2020.01 ~ 2022.04")

# -----------------------------------------------------------------------------
# 4. 메인 대시보드 (결과 시각화)
# -----------------------------------------------------------------------------
st.title("📊 참치 생산 실적 예측 대시보드")
st.markdown("##### AI 기반 생산량 예측 및 공정 변수 진단")

# (1) 예측 계산
input_data = pd.DataFrame({'const': 1.0, 'yield': [input_yield], 'productivity': [input_prod], 'workforce': [input_wf], 'hour': [input_hour]})
predictions = model.get_prediction(input_data)
pred_df = predictions.summary_frame(alpha=0.05)
pred_val = pred_df['mean'][0]
lower_val, upper_val = pred_df['obs_ci_lower'][0], pred_df['obs_ci_upper'][0]

# (2) 상단: 핵심 지표 (KPI)
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("📉 최소 예상 (Risk)", f"{lower_val:.1f} 톤")
kpi2.metric("🎯 예측 생산량 (Target)", f"{pred_val:.1f} 톤", delta_color="normal")
kpi3.metric("📈 최대 예상 (Max)", f"{upper_val:.1f} 톤")

st.markdown("---")

# (3) 중단: 메인 예측 그래프 (Slim Layout)
c1, c2 = st.columns([3, 1]) # 그래프 공간을 넓게, 설명 공간을 좁게

with c1:
    st.subheader("예측 구간 시각화")
    fig_main, ax = plt.subplots(figsize=(10, 1.5)) # 높이를 매우 낮게 설정 (Slim)
    
    # 그라데이션 느낌의 바 차트
    ax.barh(0, pred_val, color='#00C853', alpha=0.8, height=0.6, label='예측값')
    
    # 에러바 (신뢰구간)
    ax.errorbar(pred_val, 0, xerr=[[pred_val - lower_val], [upper_val - pred_val]], 
                fmt='o', color='#D50000', ecolor='gray', elinewidth=2, capsize=5, markersize=8)
    
    # 텍스트 레이블 (바 끝에 표시)
    ax.text(pred_val + 2, 0, f"{pred_val:.1f} 톤", va='center', fontweight='bold', fontsize=12, color='#1b5e20')
    
    # 스타일링
    ax.set_yticks([]) # Y축 라벨 제거
    ax.set_xlim(lower_val * 0.9, upper_val * 1.1)
    ax.set_xlabel("생산량 (톤)")
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 테두리 제거 (깔끔하게)
    sns.despine(left=True, bottom=False)
    st.pyplot(fig_main)

with c2:
    st.info("""
    **그래프 보는 법**
    * **초록 막대:** 예측값
    * **빨간 점:** 95% 신뢰구간
    """)

# (4) 하단: 입력 변수 진단 (Compact Row Layout)
st.subheader("🔍 투입 조건 진단 (vs 과거 분포)")

# 4개의 컬럼으로 나누어 한 줄에 배치
cols = st.columns(4) 
vars_info = [
    ('yield', '수율 (%)', input_yield, 'Blues'),
    ('productivity', '생산성', input_prod, 'Greens'),
    ('workforce', '인원 (명)', input_wf, 'Oranges'),
    ('hour', '시간 (h)', input_hour, 'Purples')
]

for i, (col_name, title, current_val, color_theme) in enumerate(vars_info):
    with cols[i]:
        # 작은 그래프 생성
        fig, ax = plt.subplots(figsize=(3, 2)) # 아주 작은 사이즈
        
        # KDE Plot (부드러운 곡선 분포)
        sns.kdeplot(data=df_clean, x=col_name, fill=True, color=sns.color_palette(color_theme)[4], alpha=0.3, ax=ax)
        
        # 현재 값 표시 (빨간선)
        ax.axvline(current_val, color='#FF5252', linestyle='--', linewidth=2)
        
        # 스타일링
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticks([]) # Y축 눈금 제거 (깔끔하게)
        
        # 현재 위치 텍스트
        ax.text(current_val, ax.get_ylim()[1]*0.9, "Here", color='#FF5252', ha='center', fontsize=8, fontweight='bold')
        
        sns.despine(left=True) # 왼쪽 테두리 제거
        st.pyplot(fig)
