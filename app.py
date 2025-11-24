import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# -----------------------------------------------------------------------------
# 1. 환경 설정 (한글 폰트 자동 다운로드 및 적용)
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

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생산량 예측 AI", layout="wide")

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

# (2) 전처리 (이상치 제거)
drop_indices = [16, 19, 22]
df_clean = df.drop(drop_indices, errors='ignore').reset_index(drop=True)

# (3) 모델 학습
X = df_clean[['yield', 'productivity', 'workforce', 'hour']]
y = df_clean['production']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# -----------------------------------------------------------------------------
# 3. 사용자 인터페이스 (UI)
# -----------------------------------------------------------------------------
st.title("🐟 참치 생산 실적 예측 시뮬레이터")
st.markdown("과거 데이터를 기반으로 **투입 조건에 따른 예상 생산량**을 산출하고, **입력값의 적정성**을 진단합니다.")
st.divider()

col_input, col_result = st.columns([1, 2])

# --- [좌측] 입력 패널 ---
with col_input:
    st.subheader("🛠️ 생산 조건 입력")
    st.info("오늘의 작업 계획을 입력하세요.")
    
    # 슬라이더 설정
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce, 명)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour, 시간)", 160, 200, 180, step=1)
    
    st.write("---")
    with st.expander("📊 모델 통계 (Summary)"):
        st.code(str(model.summary()))

# --- [우측] 결과 패널 ---
with col_result:
    # 1. 예측 계산
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
    
    # 2. 메인 결과 (Metrics)
    st.subheader("📈 AI 예측 결과")
    m1, m2, m3 = st.columns(3)
    m1.metric("최소 예상", f"{lower_val:.1f} 톤")
    m2.metric("🎯 예측 생산량", f"{pred_val:.1f} 톤", delta="Target")
    m3.metric("최대 예상", f"{upper_val:.1f} 톤")
    
    # 3. 예측 그래프 (Production Graph)
    fig_main, ax_main = plt.subplots(figsize=(10, 2.5))
    ax_main.barh(['생산량'], [pred_val], color='#2ecc71', alpha=0.7, height=0.4)
    ax_main.errorbar([pred_val], ['생산량'], xerr=[[pred_val - lower_val], [upper_val - pred_val]], 
                     fmt='o', color='red', ecolor='gray', elinewidth=3, capsize=5)
    
    # 그래프 꾸미기
    ax_main.set_xlim(lower_val * 0.9, upper_val * 1.1)
    ax_main.set_xlabel('생산량 (톤)')
    ax_main.grid(axis='x', linestyle='--', alpha=0.5)
    ax_main.text(pred_val, 0.3, f"{pred_val:.1f} 톤", ha='center', fontweight='bold', fontsize=12)
    st.pyplot(fig_main)
    
    st.write("---")
    
    # 4. 입력 변수 진단 그래프 (Input Analysis Graphs)
    st.subheader("🔍 투입 조건 진단 (vs 과거 데이터)")
    st.markdown("""
    <div style='font-size: 0.9em; color: gray; margin-bottom: 10px;'>
    • <b>회색 막대:</b> 과거 실제 데이터 분포 &nbsp;&nbsp; | &nbsp;&nbsp; 
    • <b>빨간 선:</b> 현재 입력한 계획 값 &nbsp;&nbsp; | &nbsp;&nbsp; 
    • <b>파란 점선:</b> 과거 평균
    </div>
    """, unsafe_allow_html=True)
    
    # 4개 변수 시각화를 위한 서브플롯 생성
    fig_sub, axes = plt.subplots(2, 2, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 변수 매핑 정보
    vars_info = [
        ('yield', '수율 (%)', input_yield),
        ('productivity', '생산성 지표', input_prod),
        ('workforce', '투입 인원 (명)', input_wf),
        ('hour', '작업 시간 (h)', input_hour)
    ]
    
    # 반복문으로 4개 그래프 그리기
    for idx, (col, title, current_val) in enumerate(vars_info):
        row, col_idx = divmod(idx, 2)
        ax = axes[row, col_idx]
        
        # 히스토그램 (과거 데이터 분포)
        ax.hist(df_clean[col], bins=10, color='lightgray', edgecolor='white', label='과거 분포')
        
        # 현재 입력값 (빨간 실선)
        ax.axvline(current_val, color='#e74c3c', linewidth=2, linestyle='-', label='현재 입력')
        
        # 과거 평균값 (파란 점선)
        mean_val = df_clean[col].mean()
        ax.axvline(mean_val, color='#3498db', linewidth=1.5, linestyle='--', label='과거 평균')
        
        # 디자인
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        # 현재 값이 평균과 많이 차이나면 텍스트로 표시
        if idx == 0: # 첫 번째 그래프에만 범례 표시 (깔끔하게)
            ax.legend(loc='upper right', fontsize=8)

    st.pyplot(fig_sub)
