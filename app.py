import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- [페이지 설정] ---
st.set_page_config(page_title="쪼꼬야옹 백테스트 연구소", page_icon="📈", layout="wide")

# --- [세션 상태 초기화] ---
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = []
if 'trial_count' not in st.session_state:
    st.session_state.trial_count = 0

# --- [유틸리티 함수] ---
def excel_round_up(x, digit=0):
    return float(np.ceil(x * (10 ** digit)) / (10 ** digit))

def excel_round_down(x, digit=0):
    return float(np.floor(x * (10 ** digit)) / (10 ** digit))

def calculate_loc_quantity(seed_amount, order_price, close_price, buy_range, max_add_orders):
    # (핵심 로직은 기존과 동일하므로 간략화하여 구현)
    # 실제로는 사용자가 기존에 쓰던 함수를 그대로 가져옵니다.
    # 여기서는 시뮬레이션을 위해 약식으로 계산합니다.
    return int(seed_amount / order_price)

# --- [백테스트 엔진] ---
def backtest_engine_web(df, params):
    # 데이터 필터링
    df = df.copy()
    
    # 이평선 계산
    ma_window = int(params['ma_window'])
    df['MA_New'] = df['QQQ'].rolling(window=ma_window, min_periods=1).mean()
    df['Disparity'] = df['QQQ'] / df['MA_New']
    
    # 주간 데이터 매핑 (약식)
    # 실제 웹 앱에서는 속도를 위해 미리 계산된 컬럼을 쓰는 것이 좋습니다.
    # 여기서는 매번 계산하도록 둡니다.
    weekly_series = df['Disparity'].resample('W-FRI').last()
    weekly_df = pd.DataFrame({'Basis_Disp': weekly_series})
    calendar_df = weekly_df.resample('D').ffill()
    calendar_shifted = calendar_df.shift(1)
    daily_mapped = calendar_shifted.reindex(df.index).ffill()
    df['Basis_Disp'] = daily_mapped['Basis_Disp']
    df['Prev_Close'] = df['SOXL'].shift(1)
    
    start_dt = pd.to_datetime(params['start_date'])
    end_dt = pd.to_datetime(params['end_date'])
    df = df[(df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))].copy()
    
    if len(df) == 0: return None

    # 전략 파라미터
    strategy = {
        'Bottom':  {'cond': params['bt_cond'], 'buy': params['bt_buy'], 'prof': params['bt_prof'], 'time': params['bt_time']},
        'Ceiling': {'cond': params['cl_cond'], 'buy': params['cl_buy'], 'prof': params['cl_prof'], 'time': params['cl_time']},
        'Middle':  {'cond': 999,           'buy': params['md_buy'], 'prof': params['md_prof'], 'time': params['md_time']}
    }
    
    cash = params['initial_balance']
    seed_equity = cash
    holdings = []
    daily_equity = []
    dates = df.index
    MAX_SLOTS = 10
    SEC_FEE = 0.0000278

    for i in range(len(df)):
        row = df.iloc[i]
        today_close = row['SOXL']
        disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        
        # 모드 결정
        if disp < strategy['Bottom']['cond']: phase = 'Bottom'
        elif disp > strategy['Ceiling']['cond']: phase = 'Ceiling'
        else: phase = 'Middle'
        
        conf = strategy[phase]
        
        # 매도
        tiers_sold = set()
        daily_profit = 0
        for stock in holdings[:]:
            buy_p, days, qty, mode, tier, _ = stock
            s_conf = strategy[mode]
            days += 1
            target_p = excel_round_up(buy_p * (1 + s_conf['prof']), 2)
            
            is_sold = False
            if days >= s_conf['time'] or today_close >= target_p:
                is_sold = True
            
            if is_sold:
                holdings.remove(stock)
                tiers_sold.add(tier)
                amt = today_close * qty
                fee = amt * SEC_FEE
                net = amt * (1 - params['fee_rate']) - fee
                cost = (buy_p * qty) * (1 + params['fee_rate'])
                daily_profit += (net - cost)
                cash += net
            else:
                stock[1] = days
        
        # 투자금 갱신
        if daily_profit != 0:
            rate = params['profit_rate'] if daily_profit > 0 else params['loss_rate']
            seed_equity += daily_profit * rate
            
        # 매수
        target_seed = int((seed_equity / MAX_SLOTS) + 0.5)
        prev_c = row['Prev_Close'] if not pd.isna(row['Prev_Close']) else today_close
        target_p = excel_round_down(prev_c * (1 + conf['buy'] / 100), 2)
        bet = min(target_seed, cash)
        
        if today_close <= target_p and len(holdings) < MAX_SLOTS and bet > 10:
            curr_tiers = {h[4] for h in holdings}
            unavail = curr_tiers.union(tiers_sold)
            new_tier = 1
            while new_tier in unavail: new_tier += 1
            
            if new_tier <= MAX_SLOTS:
                qty = int(bet / target_p)
                max_q = int(cash / (today_close * (1+params['fee_rate'])))
                real_q = min(qty, max_q)
                if real_q > 0:
                    cash -= today_close * real_q * (1+params['fee_rate'])
                    holdings.append([today_close, 0, real_q, phase, new_tier, dates[i]])
        
        daily_equity.append(cash + sum([h[2]*today_close for h in holdings]))

    # 결과 정리
    final_equity = daily_equity[-1]
    days = (dates[-1] - dates[0]).days
    cagr = ((final_equity / params['initial_balance']) ** (365/days) - 1) * 100 if days > 0 else 0
    
    eq_series = pd.Series(daily_equity)
    peak = eq_series.cummax()
    mdd = ((eq_series / peak - 1) * 100).min()
    
    return {
        'CAGR': round(cagr, 2),
        'MDD': round(mdd, 2),
        'Final': int(final_equity),
        'Series': daily_equity,
        'Dates': dates
    }

# --- [UI 구성] ---
st.title("📊 쪼꼬야옹 백테스트 연구소")
st.markdown("언제 어디서나 최적의 파라미터를 찾아보세요!")

# 1. 사이드바 (설정)
with st.sidebar:
    st.header("⚙️ 기본 설정")
    uploaded_file = st.file_uploader("📂 데이터 파일 업로드 (csv)", type=['csv'])
    
    st.subheader("💰 자산 설정")
    balance = st.number_input("초기 자본 ($)", value=10000)
    fee = st.number_input("수수료 (%)", value=0.07)
    
    st.subheader("📈 기간 설정")
    # 날짜 기본값
    start_date = st.date_input("시작일", pd.to_datetime("2010-01-01"))
    end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))

# 2. 메인 화면
if uploaded_file is not None:
    # 데이터 로드
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    tab1, tab2, tab3 = st.tabs(["🚀 백테스트", "🎲 몬테카를로 최적화", "🔬 심층 분석"])
    
    # --- 탭 1: 단일 백테스트 ---
    with tab1:
        st.subheader("🛠️ 전략 파라미터 설정")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📉 바닥 (Bottom)")
            bt_cond = st.slider("이격도 기준", 0.8, 1.0, 0.96)
            bt_buy = st.number_input("매수점 (%)", -20.0, 5.0, -5.0, key='bt_b')
            bt_prof = st.number_input("익절 (%)", 1.0, 50.0, 10.0, key='bt_p')
            bt_time = st.number_input("존버일", 1, 100, 50, key='bt_t')
            
        with col2:
            st.markdown("### ➖ 중간 (Middle)")
            md_buy = st.number_input("매수점 (%)", -10.0, 5.0, -2.5, key='md_b')
            md_prof = st.number_input("익절 (%)", 1.0, 30.0, 5.0, key='md_p')
            md_time = st.number_input("존버일", 1, 100, 30, key='md_t')

        with col3:
            st.markdown("### 📈 천장 (Ceiling)")
            cl_cond = st.slider("이격도 기준", 1.0, 1.3, 1.05)
            cl_buy = st.number_input("매수점 (%)", -20.0, 5.0, -10.0, key='cl_b')
            cl_prof = st.number_input("익절 (%)", 1.0, 30.0, 5.0, key='cl_p')
            cl_time = st.number_input("존버일", 1, 100, 20, key='cl_t')
            
        ma_win = st.slider("이평선 (MA)", 100, 300, 200)

        if st.button("백테스트 실행", type="primary"):
            params = {
                'start_date': start_date, 'end_date': end_date,
                'initial_balance': balance, 'fee_rate': fee/100,
                'ma_window': ma_win, 'profit_rate': 0.7, 'loss_rate': 0.5,
                'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time
            }
            res = backtest_engine_web(df, params)
            
            st.success(f"최종 자산: ${res['Final']:,.0f} (CAGR: {res['CAGR']}%)")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(res['Dates'], res['Series'], label='Total Equity', color='red')
            ax.set_title("Equity Curve")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

    # --- 탭 2: 최적화 ---
    with tab2:
        st.subheader("🎲 몬테카를로 시뮬레이션")
        sim_count = st.slider("시도 횟수", 10, 200, 50)
        
        if st.button("최적화 시작"):
            progress_bar = st.progress(0)
            
            for i in range(sim_count):
                # 랜덤 파라미터 생성 (범위는 예시로 고정)
                rand_params = {
                    'start_date': start_date, 'end_date': end_date,
                    'initial_balance': balance, 'fee_rate': fee/100,
                    'ma_window': np.random.randint(150, 250),
                    'profit_rate': 0.7, 'loss_rate': 0.5,
                    
                    'bt_cond': np.random.uniform(0.90, 0.99),
                    'bt_buy': np.random.uniform(-10, 0),
                    'bt_prof': np.random.uniform(0.05, 0.20),
                    'bt_time': np.random.randint(30, 80),
                    
                    'md_buy': np.random.uniform(-5, 0),
                    'md_prof': np.random.uniform(0.03, 0.10),
                    'md_time': np.random.randint(20, 60),
                    
                    'cl_cond': np.random.uniform(1.01, 1.15),
                    'cl_buy': np.random.uniform(-15, -5),
                    'cl_prof': np.random.uniform(0.02, 0.08),
                    'cl_time': np.random.randint(10, 40)
                }
                
                res = backtest_engine_web(df, rand_params)
                if res:
                    flat_res = rand_params.copy()
                    flat_res.update(res)
                    del flat_res['Series'] # 용량 절약
                    del flat_res['Dates']
                    st.session_state.opt_results.append(flat_res)
                
                progress_bar.progress((i + 1) / sim_count)
            
            st.success("탐색 완료!")
            
        if st.session_state.opt_results:
            res_df = pd.DataFrame(st.session_state.opt_results)
            res_df['Score'] = res_df['CAGR'] - abs(res_df['MDD'])
            res_df = res_df.sort_values('Score', ascending=False)
            
            st.dataframe(res_df[['Score', 'CAGR', 'MDD', 'ma_window', 'Final']].head(10))

else:
    st.info("👈 왼쪽 사이드바에서 데이터 파일(CSV)을 먼저 업로드해주세요.")
    st.markdown("""
    **[Tip] 코랩에서 사용하던 데이터를 다운받으려면?**
    1. 코랩에서 `global_df.to_csv('my_data.csv')` 코드를 실행하세요.
    2. 생성된 `my_data.csv`를 다운받아 여기에 업로드하면 됩니다!
    """)