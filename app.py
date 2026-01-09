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
if 'last_backtest_result' not in st.session_state:
    st.session_state.last_backtest_result = None

# --- [유틸리티 함수] ---
def excel_round_up(x, digit=0):
    return float(np.ceil(x * (10 ** digit)) / (10 ** digit))

def excel_round_down(x, digit=0):
    return float(np.floor(x * (10 ** digit)) / (10 ** digit))

def calculate_loc_quantity(seed_amount, order_price):
    return int(seed_amount / order_price)

# --- [핵심 엔진] ---
def backtest_engine_web(df, params):
    # 1. 데이터 준비
    df = df.copy()
    ma_window = int(params['ma_window'])
    df['MA_New'] = df['QQQ'].rolling(window=ma_window, min_periods=1).mean()
    df['Disparity'] = df['QQQ'] / df['MA_New']
    
    weekly_series = df['Disparity'].resample('W-FRI').last()
    weekly_df = pd.DataFrame({'Basis_Disp': weekly_series})
    calendar_df = weekly_df.resample('D').ffill()
    daily_mapped = calendar_df.shift(1).reindex(df.index).ffill()
    df['Basis_Disp'] = daily_mapped['Basis_Disp']
    df['Prev_Close'] = df['SOXL'].shift(1)
    
    start_dt = pd.to_datetime(params['start_date'])
    end_dt = pd.to_datetime(params['end_date'])
    df = df[(df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))].copy()
    
    if len(df) == 0: return None

    # 2. 전략 파라미터
    strategy = {
        'Bottom':  {'cond': params['bt_cond'], 'buy': params['bt_buy'], 'prof': params['bt_prof'], 'time': params['bt_time']},
        'Ceiling': {'cond': params['cl_cond'], 'buy': params['cl_buy'], 'prof': params['cl_prof'], 'time': params['cl_time']},
        'Middle':  {'cond': 999,           'buy': params['md_buy'], 'prof': params['md_prof'], 'time': params['md_time']}
    }
    
    cash = params['initial_balance']
    seed_equity = cash
    holdings = []
    
    # 기록용
    daily_equity = []
    daily_dates = []
    trade_count = 0
    win_count = 0
    
    MAX_SLOTS = 10
    SEC_FEE = 0.0000278

    # 3. 일별 루프
    for i in range(len(df)):
        row = df.iloc[i]
        today_close = row['SOXL']
        disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        
        # 모드 결정
        if disp < strategy['Bottom']['cond']: phase = 'Bottom'
        elif disp > strategy['Ceiling']['cond']: phase = 'Ceiling'
        else: phase = 'Middle'
        
        conf = strategy[phase]
        
        # [매도]
        tiers_sold = set()
        daily_profit = 0
        
        for stock in holdings[:]:
            buy_p, days, qty, mode, tier, _ = stock
            s_conf = strategy[mode]
            days += 1
            target_p = excel_round_up(buy_p * (1 + s_conf['prof']), 2)
            
            is_sold = False
            # 손절일(TimeCut) 또는 익절
            if days >= s_conf['time'] or today_close >= target_p:
                is_sold = True
            
            if is_sold:
                holdings.remove(stock)
                tiers_sold.add(tier)
                amt = today_close * qty
                fee = amt * SEC_FEE
                net = amt * (1 - params['fee_rate']) - fee
                cost = (buy_p * qty) * (1 + params['fee_rate'])
                
                real_profit = net - cost
                daily_profit += real_profit
                cash += net
                
                trade_count += 1
                if real_profit > 0: win_count += 1
            else:
                stock[1] = days
        
        # [투자금 갱신] (일별 합산 복리)
        if daily_profit != 0:
            rate = params['profit_rate'] if daily_profit > 0 else params['loss_rate']
            seed_equity += daily_profit * rate
            
        # [매수]
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
        
        # 자산 기록
        current_eq = cash + sum([h[2]*today_close for h in holdings])
        daily_equity.append(current_eq)
        daily_dates.append(dates[i])

    # 4. 결과 지표 계산
    final_equity = daily_equity[-1]
    total_ret_pct = (final_equity / params['initial_balance'] - 1) * 100
    
    # CAGR
    days_total = (dates[-1] - dates[0]).days
    cagr = ((final_equity / params['initial_balance']) ** (365/days_total) - 1) * 100 if days_total > 0 else 0
    
    # MDD
    eq_series = pd.Series(daily_equity, index=daily_dates)
    peak = eq_series.cummax()
    mdd = ((eq_series / peak - 1) * 100).min()
    
    # 승률
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    
    # 연도별 수익률
    yearly_ret = eq_series.resample('YE').last().pct_change() * 100
    # 첫해 수익률 보정
    yearly_ret.iloc[0] = (eq_series.resample('YE').last().iloc[0] / params['initial_balance'] - 1) * 100

    return {
        'CAGR': round(cagr, 2),
        'MDD': round(mdd, 2),
        'Final': int(final_equity),
        'Return': round(total_ret_pct, 2),
        'WinRate': round(win_rate, 2),
        'Trades': trade_count,
        'Series': eq_series,
        'Yearly': yearly_ret,
        'Params': params
    }

# --- [UI 구성] ---
st.title("📊 쪼꼬야옹 백테스트 연구소")

# 1. 사이드바 (설정)
with st.sidebar:
    st.header("⚙️ 기본 설정")
    uploaded_file = st.file_uploader("📂 데이터 파일 (CSV)", type=['csv'])
    
    st.subheader("💰 자산 설정")
    balance = st.number_input("초기 자본 ($)", value=10000)
    fee = st.number_input("수수료 (%)", value=0.07)
    
    st.subheader("📈 기간 설정")
    start_date = st.date_input("시작일", pd.to_datetime("2010-01-01"))
    end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))

# 2. 메인 화면 로직
if uploaded_file is not None:
    # 데이터 로드
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    tab1, tab2, tab3 = st.tabs(["🚀 백테스트", "🎲 몬테카를로 최적화", "🔬 심층 분석"])
    
    # ==========================
    # 탭 1: 백테스트 (개별 실행)
    # ==========================
    with tab1:
        st.subheader("🛠️ 전략 파라미터 입력")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 📉 바닥 (Bottom)")
            bt_cond = st.number_input("기준 이격도", 0.8, 1.0, 0.96, step=0.01)
            bt_buy = st.number_input("매수점 (%)", -30.0, 30.0, -5.0, step=0.1, key='bt_b')
            bt_prof = st.number_input("익절 (%)", 0.0, 100.0, 10.0, step=0.1, key='bt_p')
            bt_time = st.number_input("존버일", 1, 100, 50, key='bt_t')
            
        with col2:
            st.markdown("##### ➖ 중간 (Middle)")
            md_buy = st.number_input("매수점 (%)", -30.0, 30.0, -2.5, step=0.1, key='md_b')
            md_prof = st.number_input("익절 (%)", 0.0, 100.0, 5.0, step=0.1, key='md_p')
            md_time = st.number_input("존버일", 1, 100, 30, key='md_t')

        with col3:
            st.markdown("##### 📈 천장 (Ceiling)")
            cl_cond = st.number_input("기준 이격도", 1.0, 1.5, 1.05, step=0.01)
            cl_buy = st.number_input("매수점 (%)", -30.0, 30.0, -10.0, step=0.1, key='cl_b')
            cl_prof = st.number_input("익절 (%)", 0.0, 100.0, 5.0, step=0.1, key='cl_p')
            cl_time = st.number_input("존버일", 1, 100, 20, key='cl_t')
            
        ma_win = st.number_input("이평선 (MA)", 50, 300, 200)

        if st.button("백테스트 실행 (Run)", type="primary"):
            current_params = {
                'start_date': start_date, 'end_date': end_date,
                'initial_balance': balance, 'fee_rate': fee/100,
                'ma_window': ma_win, 'profit_rate': 0.7, 'loss_rate': 0.5,
                'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
                'label': '🎯 현재 설정'
            }
            res = backtest_engine_web(df, current_params)
            st.session_state.last_backtest_result = res # 분석 탭을 위해 저장
            
            # 결과 요약
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("최종 자산", f"${res['Final']:,.0f}", f"{res['Return']}%")
            m2.metric("CAGR (연평균)", f"{res['CAGR']}%")
            m3.metric("MDD (최대낙폭)", f"{res['MDD']}%")
            m4.metric("승률 / 횟수", f"{res['WinRate']}%", f"{res['Trades']}회")
            
            # 그래프
            st.line_chart(res['Series'])
            
            # 연도별 수익률 차트
            st.markdown("#### 📅 연도별 수익률")
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['red' if x >= 0 else 'blue' for x in res['Yearly']]
            bars = ax.bar(res['Yearly'].index.year, res['Yearly'], color=colors, alpha=0.7)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', 
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
            st.pyplot(fig)

    # ==========================
    # 탭 2: 몬테카를로 최적화
    # ==========================
    with tab2:
        st.subheader("🎲 최적 파라미터 탐색")
        st.info("💡 범위를 설정하고 '최적화 시작'을 누르면 결과가 누적됩니다.")
        
        # 범위 설정 UI
        c1, c2 = st.columns(2)
        with c1:
            sim_count = st.slider("시도 횟수", 10, 1000, 100, step=10)
            ma_range = st.slider("이평선 범위", 100, 300, (120, 250))
            
            st.markdown("**📉 바닥 모드 범위**")
            bt_buy_r = st.slider("바닥 매수점", -20.0, 20.0, (-10.0, 5.0))
            bt_prof_r = st.slider("바닥 익절", 0.0, 20.0, (5.0, 15.0))
            bt_time_r = st.slider("바닥 존버", 1, 50, (20, 50))
            
        with c2:
            st.markdown("**📈 천장/중간 모드 범위**")
            md_buy_r = st.slider("중간 매수점", -20.0, 20.0, (-5.0, 5.0))
            md_prof_r = st.slider("중간 익절", 0.0, 20.0, (3.0, 10.0))
            md_time_r = st.slider("중간 존버", 1, 50, (10, 40))

            cl_buy_r = st.slider("천장 매수점", -20.0, 20.0, (-15.0, -5.0))
            cl_prof_r = st.slider("천장 익절", 0.0, 20.0, (2.0, 8.0))
            cl_time_r = st.slider("천장 존버", 1, 50, (5, 30))

        # 실행 버튼
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        if col_btn1.button("🚀 최적화 시작"):
            # 현재 탭1의 설정값도 비교군으로 추가
            curr_res = backtest_engine_web(df, {
                'start_date': start_date, 'end_date': end_date,
                'initial_balance': balance, 'fee_rate': fee/100,
                'ma_window': ma_win, 'profit_rate': 0.7, 'loss_rate': 0.5,
                'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
            })
            if curr_res:
                entry = curr_res['Params'].copy()
                entry.update({'ID': 'MySet', 'CAGR': curr_res['CAGR'], 'MDD': curr_res['MDD'], 
                              'Score': curr_res['CAGR'] - abs(curr_res['MDD']), 'Label': '🎯 현재 설정'})
                st.session_state.opt_results.append(entry)

            # 랜덤 시뮬레이션
            prog = st.progress(0)
            for i in range(sim_count):
                st.session_state.trial_count += 1
                r_params = {
                    'start_date': start_date, 'end_date': end_date,
                    'initial_balance': balance, 'fee_rate': fee/100,
                    'profit_rate': 0.7, 'loss_rate': 0.5,
                    'ma_window': np.random.randint(ma_range[0], ma_range[1]),
                    'bt_cond': np.random.uniform(0.90, 0.99),
                    'cl_cond': np.random.uniform(1.01, 1.15),
                    
                    'bt_buy': round(np.random.uniform(bt_buy_r[0], bt_buy_r[1]), 1),
                    'bt_prof': round(np.random.uniform(bt_prof_r[0], bt_prof_r[1])/100, 4),
                    'bt_time': np.random.randint(bt_time_r[0], bt_time_r[1]),
                    
                    'md_buy': round(np.random.uniform(md_buy_r[0], md_buy_r[1]), 1),
                    'md_prof': round(np.random.uniform(md_prof_r[0], md_prof_r[1])/100, 4),
                    'md_time': np.random.randint(md_time_r[0], md_time_r[1]),
                    
                    'cl_buy': round(np.random.uniform(cl_buy_r[0], cl_buy_r[1]), 1),
                    'cl_prof': round(np.random.uniform(cl_prof_r[0], cl_prof_r[1])/100, 4),
                    'cl_time': np.random.randint(cl_time_r[0], cl_time_r[1]),
                }
                
                res = backtest_engine_web(df, r_params)
                if res:
                    entry = r_params.copy()
                    entry.update({
                        'ID': st.session_state.trial_count,
                        'CAGR': res['CAGR'], 'MDD': res['MDD'], 
                        'Score': res['CAGR'] - abs(res['MDD']),
                        'Label': '🎲 랜덤'
                    })
                    st.session_state.opt_results.append(entry)
                prog.progress((i+1)/sim_count)
            st.success("완료!")

        if col_btn2.button("🗑️ 결과 초기화"):
            st.session_state.opt_results = []
            st.session_state.trial_count = 0
            st.rerun()

        # 결과 표시
        if st.session_state.opt_results:
            res_df = pd.DataFrame(st.session_state.opt_results)
            # Score 기준 정렬
            res_df = res_df.sort_values('Score', ascending=False).reset_index(drop=True)
            res_df.index += 1
            res_df.index.name = 'Rank'
            
            # 메인 테이블 출력
            show_cols = ['Label', 'Score', 'CAGR', 'MDD', 'ma_window', 'bt_buy', 'bt_prof']
            st.markdown("##### 🏆 Top 랭킹 (Score순)")
            
            # 스타일링: 내 설정 강조
            def highlight_myset(s):
                return ['background-color: #FFF8DC' if s['Label'] == '🎯 현재 설정' else '' for _ in s]
            
            st.dataframe(res_df[show_cols].style.apply(highlight_myset, axis=1), height=300)
            
            # 상세 보기
            st.markdown("---")
            st.subheader("🔍 상세 파라미터 보기")
            
            # 선택 박스 생성
            options = []
            for idx, row in res_df.head(30).iterrows(): # Top 30만 표시
                lbl = f"[Rank {idx}] {row['Label']} (Score: {row['Score']:.2f} | CAGR: {row['CAGR']}%)"
                options.append(lbl)
                
            selected_opt = st.selectbox("결과를 선택하세요:", options)
            
            if selected_opt:
                rank_idx = int(selected_opt.split(']')[0].replace('[Rank ', ''))
                sel_row = res_df.loc[rank_idx]
                
                code_text = f"""# === [Rank {rank_idx}] {sel_row['Label']} 파라미터 ===
# Score: {sel_row['Score']} | CAGR: {sel_row['CAGR']}% | MDD: {sel_row['MDD']}%

MY_BEST_PARAMS = {{
    'ma_window': {sel_row['ma_window']},
    'bt_cond': {sel_row['bt_cond']:.2f}, 'bt_buy': {sel_row['bt_buy']}, 'bt_prof': {sel_row['bt_prof']*100:.1f}, 'bt_time': {sel_row['bt_time']},
    'md_buy': {sel_row['md_buy']}, 'md_prof': {sel_row['md_prof']*100:.1f}, 'md_time': {sel_row['md_time']},
    'cl_cond': {sel_row['cl_cond']:.2f}, 'cl_buy': {sel_row['cl_buy']}, 'cl_prof': {sel_row['cl_prof']*100:.1f}, 'cl_time': {sel_row['cl_time']}
}}"""
                st.code(code_text, language='python')
                
                # 심층 분석으로 보내기 위한 버튼 (Session State 활용)
                if st.button("이 전략으로 심층 분석하기 ➡️"):
                    sel_row_dict = sel_row.to_dict()
                    # % 단위 복원 등 전처리 필요하면 여기서 수행 (이미 decimal 상태)
                    st.session_state.target_analysis_params = sel_row_dict
                    st.success("심층 분석 탭으로 이동하세요!")

    # ==========================
    # 탭 3: 심층 분석
    # ==========================
    with tab3:
        st.subheader("🔬 전략 정밀 검진")
        
        target = None
        
        # 분석 대상 선택
        src = st.radio("분석 대상:", ["최근 백테스트 결과", "최적화에서 선택한 전략"])
        
        if src == "최근 백테스트 결과":
            if st.session_state.last_backtest_result:
                target = st.session_state.last_backtest_result['Params']
            else:
                st.warning("⚠️ 백테스트 탭에서 먼저 '실행'을 눌러주세요.")
                
        else: # 최적화 선택 전략
            if 'target_analysis_params' in st.session_state:
                target = st.session_state.target_analysis_params
            else:
                st.warning("⚠️ 최적화 탭에서 전략을 선택하고 '심층 분석하기' 버튼을 눌러주세요.")
        
        if target:
            # 분석 실행
            res = backtest_engine_web(df, target)
            
            if res:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("CAGR", f"{res['CAGR']}%")
                k2.metric("MDD", f"{res['MDD']}%")
                k3.metric("승률", f"{res['WinRate']}%")
                k4.metric("거래횟수", f"{res['Trades']}회")
                
                st.markdown("#### 📅 연도별 수익률 상세")
                
                # 연도별 표 + 그래프
                yearly_df = pd.DataFrame(res['Yearly'])
                yearly_df.columns = ['Return %']
                yearly_df.index = yearly_df.index.strftime('%Y')
                
                c_chart, c_table = st.columns([2, 1])
                
                with c_chart:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['red' if x >= 0 else 'blue' for x in res['Yearly']]
                    bars = ax.bar(res['Yearly'].index.year, res['Yearly'], color=colors, alpha=0.7)
                    ax.axhline(0, color='black', linewidth=0.8)
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', 
                                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
                    st.pyplot(fig)
                    
                with c_table:
                    st.dataframe(yearly_df.style.background_gradient(cmap='RdBu_r', vmin=-50, vmax=50), height=400)

else:
    st.info("👈 왼쪽 사이드바에서 데이터 파일을 먼저 업로드해주세요.")