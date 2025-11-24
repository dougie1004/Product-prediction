import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 및 모바일 전용 디자인 설정 (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Production Dashboard", layout="wide")

st.markdown("""
<style>
    /* 기본(PC) 스타일 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stApp {
        background-color: #f8f9fa;
    }

    /* 📱 모바일 전용 스타일 (화면 폭이 768px 이하일 때 적용) */
    @media (max-width: 768px) {
        /* 1. 상단 여백을 줄여서 제목이 바로 보이게 함 */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 5rem !important;
        }
        /* 2. 제목 폰트 크기 조절 */
        h1 {
            font-size: 1.8rem !important;
        }
        /* 3. KPI 박스 간의 간격 확보 */
        div[data-testid="metric-container"] {
            margin-bottom: 10px;
        }
        /* 4. 그래프 간격 확보 */
        div[data-testid="column"] {
            margin-bottom: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링
# -----------------------------------------------------------------------------
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

# (2) 전처리 & 모델링
drop_indices = [16, 19, 22]
df_clean = df.drop(drop_indices, errors='ignore').reset_index(drop=True)

X = df_clean[['yield', 'productivity', 'workforce', 'hour']]
y = df_clean['production']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

means = df_clean.mean()

# -----------------------------------------------------------------------------
# 3. 사이드바 (모바일에서는 접혀있음)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 시뮬레이션 설정")
    st.info("모바일에서는 왼쪽 상단 화살표(>)를 눌러 설정을 변경하세요.") # 모바일 안내 문구 추가
    st.markdown("---")
    
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour)", 160, 200, 180, step=1)
    
    st.markdown("---")
    st.caption(f"Model Accuracy ($R^2$): **{model.rsquared:.2f}**")

# -----------------------------------------------------------------------------
# 4. 메인 대시보드
# -----------------------------------------------------------------------------
st.title("🏭 생산 실적 예측")
st.markdown("**AI-driven Production Forecasting**")
st.caption("👈 왼쪽 사이드바를 열어 조건을 입력하세요.") # 모바일 사용자를 위한 힌트

# (1) 예측 계산
input_data = pd.DataFrame({'const': 1.0, 'yield': [input_yield], 'productivity': [input_prod], 'workforce': [input_wf], 'hour': [input_hour]})
predictions = model.get_prediction(input_data)
pred_df = predictions.summary_frame(alpha=0.05)

pred_val = pred_df['mean'][0]
lower_val = pred_df['obs_ci_lower'][0]
upper_val = pred_df['obs_ci_upper'][0]

st.divider()

# --- SECTION 1: 핵심 KPI (모바일에서는 자동 세로 정렬됨) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("예측 생산량 (Target)", f"{pred_val:.1f} 톤", delta=f"{pred_val - means['production']:.1f} vs Avg")
with col2:
    st.metric("최소 보장 (Risk Min)", f"{lower_val:.1f} 톤", delta="- Conservative", delta_color="off")
with col3:
    st.metric("최대 가능 (Max)", f"{upper_val:.1f} 톤", delta="+ Optimistic", delta_color="off")
with col4:
    achievement = (pred_val / 100) * 100
    st.metric("목표 달성률 (Ref. 100t)", f"{achievement:.1f}%")

st.markdown("") # 여백

# --- SECTION 2: 메인 차트 ---
# 모바일에서 그래프가 너무 작아지지 않도록 컬럼 비율 조정 안함 (1:1 자동)
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("🎯 예측 계기판")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': " 톤", 'font': {'size': 24, 'color': '#2c3e50'}},
        gauge = {
            'axis': {'range': [lower_val*0.8, upper_val*1.1], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "white",
            'steps': [
                {'range': [lower_val*0.8, lower_val], 'color': '#ffcdd2'},
                {'range': [lower_val, upper_val], 'color': '#f1f8e9'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_val}
        }
    ))
    # 모바일 높이 최적화 (250px)
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)

with c_right:
    st.subheader("📊 예측 범위 상세")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=['생산량'], x=[pred_val],
        orientation='h',
        marker_color='#3498db',
        error_x=dict(type='data', array=[upper_val-pred_val], arrayminus=[pred_val-lower_val], color='#e74c3c', width=6),
        text=[f"{pred_val:.1f} 톤"], 
        textposition='auto',
        hovertemplate='<b>예측값:</b> %{x:.1f} 톤<br>' +
                      '<b>안전 범위:</b> ±' + f"{(upper_val-lower_val)/2:.1f} 톤" + 
                      '<extra></extra>' 
    ))
    fig_bar.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Production (Tons)", range=[lower_val*0.8, upper_val*1.1]),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showticklabels=False),
        hoverlabel=dict(bgcolor="white", font_size=14)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- SECTION 3: 투입 변수 진단 ---
st.subheader("🔍 투입 변수 진단")
st.caption("진한 막대(현재) vs 연한 막대(평균)")

# 모바일에서는 4개가 세로로 자동 정렬됩니다.
cols = st.columns(4)
vars_config = [
    ('yield', '수율 (%)', input_yield, means['yield'], 100),
    ('productivity', '생산성', input_prod, means['productivity'], 2.5),
    ('workforce', '인원 (명)', input_wf, means['workforce'], 70),
    ('hour', '작업시간 (h)', input_hour, means['hour'], 220)
]

for i, (col_name, title, curr, avg, max_range) in enumerate(vars_config):
    with cols[i]:
        fig_bullet = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = curr,
            domain = {'x': [0.1, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 15, 'color': 'gray'}},
            number = {'font': {'size': 20, 'color': '#2c3e50'}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, max_range]},
                'bar': {'color': "#34495e"},
                'bgcolor': "white",
                'steps': [{'range': [0, avg], 'color': "#ecf0f1"}],
                'threshold': {'line': {'color': "#e74c3c", 'width': 3}, 'thickness': 0.75, 'value': avg}
            }
        ))
        fig_bullet.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bullet, use_container_width=True)
