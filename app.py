import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
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
def excel_round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier - 1e-9) / multiplier

def excel_round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 1e-9) / multiplier

# 🟢 [복원] Colab과 동일한 LOC 수량 계산 함수
def calculate_loc_quantity(seed_amount, order_price, close_price, buy_range, max_add_orders):
    if seed_amount is None or order_price is None or order_price <= 0:
        return 0

    # 1. Base Qty
    base_qty = int(seed_amount / order_price)

    # 2. Bot Price (바닥 가격)
    multiplier = (1 + buy_range) if buy_range <= 0 else (1 - buy_range)
    bot_price = math.floor(order_price * multiplier * 100 + 1e-9) / 100

    # 3. Fix Qty
    if bot_price > 0:
        qty_at_bot_float = seed_amount / bot_price
        qty_at_order_float = seed_amount / order_price
        fix_qty = int((qty_at_bot_float - qty_at_order_float) / max_add_orders)
    else:
        fix_qty = 0

    if fix_qty < 0: fix_qty = 0

    final_qty = 0

    # [Step 0] 기본 주문
    current_cum_qty = base_qty
    if current_cum_qty > 0:
        implied_price = seed_amount / current_cum_qty
        if implied_price >= close_price and implied_price >= bot_price:
            final_qty += base_qty

    # [Step 1 ~ Max] 추가 주문
    for i in range(1, max_add_orders + 1):
        step_qty = fix_qty
        current_cum_qty = base_qty + (i * step_qty)

        if current_cum_qty <= 0: continue

        implied_price = seed_amount / current_cum_qty
        if implied_price >= close_price and implied_price >= bot_price:
            final_qty += step_qty

    return final_qty

# --- [핵심 엔진] ---
def backtest_engine_web(df, params):
    # 1. 데이터 준비
    df = df.copy()
    ma_window = int(params['ma_window'])
    df['MA_New'] = df['QQQ'].rolling(window=ma_window, min_periods=1).mean()
    df['Disparity'] = df['QQQ'] / df['MA_New']
    
    # 주간 데이터 매핑 (Colab 로직 일치)
    weekly_series = df['Disparity'].resample('W-FRI').last()
    weekly_df = pd.DataFrame({'Basis_Disp': weekly_series})
    calendar_df = weekly_df.resample('D').ffill()
    calendar_shifted = calendar_df.shift(1) # 하루 밀기
    daily_mapped = calendar_shifted.reindex(df.index).ffill()
    df['Basis_Disp'] = daily_mapped['Basis_Disp']
    df['Prev_Close'] = df['SOXL'].shift(1)
    
    start_dt = pd.to_datetime(params['start_date'])
    end_dt = pd.to_datetime(params['end_date'])
    df = df[(df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))].copy()
    
    if len(df) == 0: return None

    dates = df.index

    # 2. 전략 파라미터
    strategy = {
        'Bottom':  {'cond': params['bt_cond'], 'buy': params['bt_buy'], 'prof': params['bt_prof'], 'time': params['bt_time']},
        'Ceiling': {'cond': params['cl_cond'], 'buy': params['cl_buy'], 'prof': params['cl_prof'], 'time': params['cl_time']},
        'Middle':  {'cond': 999,           'buy': params['md_buy'], 'prof': params['md_prof'], 'time': params['md_time']}
    }
    
    cash = params['initial_balance']
    seed_equity = cash
    holdings = []
    
    # 기록용 로그 리스트
    trade_log = [] 
    daily_log = [] 
    
    daily_equity = []
    daily_dates = []
    trade_count = 0
    win_count = 0
    
    MAX_SLOTS = 10
    SEC_FEE = 0.0000278 # SEC fee (Colab 일치)

    # 3. 일별 반복
    for i in range(len(df)):
        row = df.iloc[i]
        today_close = row['SOXL']
        # force_round 적용
        if params.get('force_round', True): today_close = round(today_close, 2)

        disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        
        # 모드 결정
        if disp < strategy['Bottom']['cond']: phase = 'Bottom'
        elif disp > strategy['Ceiling']['cond']: phase = 'Ceiling'
        else: phase = 'Middle'
        
        conf = strategy[phase]
        
        # 🟢 [중요] 시드 계산: 장 시작 전 seed_equity 기준
        target_seed_float = seed_equity / MAX_SLOTS
        target_seed = int(target_seed_float + 0.5)

        # [매도]
        tiers_sold = set()
        daily_net_profit_sum = 0
        
        for stock in holdings[:]:
            buy_p, days, qty, mode, tier, buy_dt = stock
            s_conf = strategy[mode]
            days += 1
            target_p = excel_round_up(buy_p * (1 + s_conf['prof']), 2)
            
            is_sold = False
            reason = ""
            
            if days >= s_conf['time']: 
                is_sold = True; reason = f"TimeCut({days}d)"
            elif today_close >= target_p: 
                is_sold = True; reason = "Profit"
            
            if is_sold:
                holdings.remove(stock)
                tiers_sold.add(tier)
                
                sell_amt = today_close * qty
                sec_fee_val = round(sell_amt * SEC_FEE, 2)
                
                net_receive = sell_amt * (1 - params['fee_rate']) - sec_fee_val
                buy_cost = (buy_p * qty) * (1 + params['fee_rate'])
                
                real_profit = round(net_receive - buy_cost, 2)
                daily_net_profit_sum += real_profit
                
                # 실제 현금은 즉시 입금
                cash += net_receive
                
                trade_count += 1
                if real_profit > 0: win_count += 1

                trade_log.append({
                    'Date': dates[i], 'Type': 'Sell', 'Tier': tier, 'Phase': mode,
                    'Price': today_close, 'Qty': qty, 'Profit': real_profit, 'Reason': reason
                })
            else:
                stock[1] = days
        
        # [투자금 갱신] 매도 루프 종료 후 일괄 반영 (Colab 로직)
        if daily_net_profit_sum != 0:
            rate = params['profit_rate'] if daily_net_profit_sum > 0 else params['loss_rate']
            seed_equity += daily_net_profit_sum * rate
            
        # [매수]
        prev_c = row['Prev_Close'] if not pd.isna(row['Prev_Close']) else today_close
        target_p = excel_round_down(prev_c * (1 + conf['buy'] / 100), 2)
        
        # 베팅 금액은 (갱신 전 시드)와 (현재 현금) 중 작은 값
        bet = min(target_seed_float, cash)
        if bet < 10: bet = 0
        
        if today_close <= target_p and len(holdings) < MAX_SLOTS and bet > 0:
            curr_tiers = {h[4] for h in holdings}
            unavail = curr_tiers.union(tiers_sold)
            new_tier = 1
            while new_tier in unavail: new_tier += 1
            
            if new_tier <= MAX_SLOTS:
                # 🟢 [복원] 정밀 LOC 수량 계산 (Colab 일치)
                final_qty = 0
                if new_tier == MAX_SLOTS:
                    final_qty = int(bet / target_p)
                else:
                    final_qty = calculate_loc_quantity(
                        seed_amount=bet,
                        order_price=target_p,
                        close_price=today_close,
                        buy_range= -1 * (params['loc_range'] / 100.0), # 음수 변환 주의
                        max_add_orders=int(params['add_order_cnt'])
                    )

                # 현금 부족 시 조정
                max_buyable = int(cash / (today_close * (1 + params['fee_rate'])))
                real_qty = min(final_qty, max_buyable)
                
                if real_qty > 0:
                    buy_amt = today_close * real_qty * (1 + params['fee_rate'])
                    cash -= buy_amt
                    holdings.append([today_close, 0, real_qty, phase, new_tier, dates[i]])
                    
                    trade_log.append({
                        'Date': dates[i], 'Type': 'Buy', 'Tier': new_tier, 'Phase': phase,
                        'Price': today_close, 'Qty': real_qty, 'Profit': 0, 'Reason': 'LOC'
                    })
        
        # 자산 기록
        current_eq = cash + sum([h[2]*today_close for h in holdings])
        daily_equity.append(current_eq)
        daily_dates.append(dates[i])
        
        daily_log.append({
            'Date': dates[i], 'Equity': round(current_eq, 2), 
            'Cash': round(cash, 2), 'SeedEquity': round(seed_equity, 2), 
            'Holdings': len(holdings)
        })

    # 4. 결과 지표 계산
    final_equity = daily_equity[-1]
    total_ret_pct = (final_equity / params['initial_balance'] - 1) * 100
    
    days_total = (dates[-1] - dates[0]).days
    cagr = ((final_equity / params['initial_balance']) ** (365/days_total) - 1) * 100 if days_total > 0 else 0
    
    eq_series = pd.Series(daily_equity, index=daily_dates)
    peak = eq_series.cummax()
    mdd = ((eq_series / peak - 1) * 100).min()
    
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    
    # YE 경고 방지
    try:
        yearly_ret = eq_series.resample('YE').last().pct_change() * 100
        yearly_ret.iloc[0] = (eq_series.resample('YE').last().iloc[0] / params['initial_balance'] - 1) * 100
    except:
        yearly_ret = eq_series.resample('Y').last().pct_change() * 100
        yearly_ret.iloc[0] = (eq_series.resample('Y').last().iloc[0] / params['initial_balance'] - 1) * 100

    return {
        'CAGR': round(cagr, 2),
        'MDD': round(mdd, 2),
        'Final': int(final_equity),
        'Return': round(total_ret_pct, 2),
        'WinRate': round(win_rate, 2),
        'Trades': trade_count,
        'Series': eq_series,
        'Yearly': yearly_ret,
        'Params': params,
        'TradeLog': pd.DataFrame(trade_log),
        'DailyLog': pd.DataFrame(daily_log)
    }

# --- [UI 구성] ---
st.title("📊 쪼꼬야옹 백테스트 연구소")

# 1. 사이드바 (설정)
with st.sidebar:
    st.header("⚙️ 기본 설정")
    uploaded_file = st.file_uploader("📂 데이터 파일 (CSV)", type=['csv'])
    
    st.subheader("💰 자산 및 복리 설정")
    balance = st.number_input("초기 자본 ($)", value=10000)
    fee = st.number_input("수수료 (%)", value=0.07)
    
    profit_rate = st.slider("이익 복리율 (%)", 0, 100, 70) # [cite: 1] 기본값 100이나 사용자가 70으로 씀
    loss_rate = st.slider("손실 복리율 (%)", 0, 100, 50)   # [cite: 1] 기본값 50
    
    st.subheader("📥 LOC 설정")
    # Colab 코드의 기본값 참조 [cite: 1]
    add_order_cnt = st.number_input("추가 주문 횟수", value=4, min_value=1) 
    loc_range = st.number_input("하단 범위 (-%)", value=20.0, min_value=0.0) 
    
    st.subheader("📈 기간 설정")
    start_date = st.date_input("시작일", pd.to_datetime("2014-01-01"))
    end_date = st.date_input("종료일", pd.to_datetime("2025-12-31"))

# 2. 메인 화면 로직
if uploaded_file is not None:
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
            bt_cond = st.number_input("기준 이격도", 0.8, 1.0, 0.90, step=0.01) # [cite: 1] 기본값
            bt_buy = st.number_input("매수점 (%)", -30.0, 30.0, 15.0, step=0.1, key='bt_b') # [cite: 1] 15.0
            bt_prof = st.number_input("익절 (%)", 0.0, 100.0, 2.5, step=0.1, key='bt_p')   # [cite: 1] 2.5
            bt_time = st.number_input("존버일", 1, 100, 10, key='bt_t')                   # [cite: 1] 10
            
        with col2:
            st.markdown("##### ➖ 중간 (Middle)")
            md_buy = st.number_input("매수점 (%)", -30.0, 30.0, -0.01, step=0.1, key='md_b') # [cite: 1] -0.01
            md_prof = st.number_input("익절 (%)", 0.0, 100.0, 2.8, step=0.1, key='md_p')     # [cite: 1] 2.8
            md_time = st.number_input("존버일", 1, 100, 15, key='md_t')                     # [cite: 1] 15

        with col3:
            st.markdown("##### 📈 천장 (Ceiling)")
            cl_cond = st.number_input("기준 이격도", 1.0, 1.5, 1.10, step=0.01) # [cite: 1] 1.10
            cl_buy = st.number_input("매수점 (%)", -30.0, 30.0, -0.1, step=0.1, key='cl_b') # [cite: 1] -0.1
            cl_prof = st.number_input("익절 (%)", 0.0, 100.0, 1.5, step=0.1, key='cl_p')    # [cite: 1] 1.5
            cl_time = st.number_input("존버일", 1, 100, 40, key='cl_t')                    # [cite: 1] 40
            
        ma_win = st.number_input("이평선 (MA)", 50, 300, 200)

        if st.button("백테스트 실행 (Run)", type="primary"):
            current_params = {
                'start_date': start_date, 'end_date': end_date,
                'initial_balance': balance, 'fee_rate': fee/100,
                'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
                'loc_range': loc_range, 'add_order_cnt': add_order_cnt, # 🟢 추가된 파라미터
                'force_round': True, # 🟢 소수점 처리
                'ma_window': ma_win, 
                'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
                'label': '🎯 현재 설정'
            }
            res = backtest_engine_web(df, current_params)
            st.session_state.last_backtest_result = res
            
            # 결과 요약
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("최종 자산", f"${res['Final']:,.0f}", f"{res['Return']}%")
            m2.metric("CAGR (연평균)", f"{res['CAGR']}%")
            m3.metric("MDD (최대낙폭)", f"{res['MDD']}%")
            m4.metric("승률 / 횟수", f"{res['WinRate']}%", f"{res['Trades']}회")
            
            # 로그 다운로드
            c_d1, c_d2 = st.columns(2)
            csv_trade = res['TradeLog'].to_csv(index=False).encode('utf-8-sig')
            c_d1.download_button("📥 매매일지 다운로드", csv_trade, "trade_log.csv", "text/csv")
            csv_daily = res['DailyLog'].to_csv(index=False).encode('utf-8-sig')
            c_d2.download_button("📥 자산일지 다운로드", csv_daily, "daily_log.csv", "text/csv")

            # 그래프
            st.line_chart(res['Series'])
            
            # 연도별 수익률 차트
            st.markdown("#### 📅 연도별 수익률")
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['red' if x >= 0 else 'blue' for x in res['Yearly']]
            bars = ax.bar(res['Yearly'].index.year, res['Yearly'], color=colors, alpha=0.7)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
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
        
        c1, c2 = st.columns(2)
        with c1:
            sim_count = st.slider("시도 횟수", 10, 1000, 100, step=10)
            ma_range = st.slider("이평선 범위", 100, 300, (120, 250))
            
            st.markdown("**📉 바닥 모드 범위**")
            bt_buy_r = st.slider("바닥 매수점", -20.0, 30.0, (10.0, 20.0))
            bt_prof_r = st.slider("바닥 익절", 0.0, 20.0, (1.0, 5.0))
            bt_time_r = st.slider("바닥 존버", 1, 50, (5, 20))
            
        with c2:
            st.markdown("**📈 천장/중간 모드 범위**")
            md_buy_r = st.slider("중간 매수점", -20.0, 20.0, (-5.0, 5.0))
            md_prof_r = st.slider("중간 익절", 0.0, 20.0, (1.0, 5.0))
            md_time_r = st.slider("중간 존버", 1, 50, (10, 30))

            cl_buy_r = st.slider("천장 매수점", -20.0, 20.0, (-10.0, 5.0))
            cl_prof_r = st.slider("천장 익절", 0.0, 20.0, (1.0, 5.0))
            cl_time_r = st.slider("천장 존버", 1, 50, (20, 50))

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        if col_btn1.button("🚀 최적화 시작"):
            # 현재 설정 추가
            curr_res = backtest_engine_web(df, {
                'start_date': start_date, 'end_date': end_date,
                'initial_balance': balance, 'fee_rate': fee/100,
                'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
                'loc_range': loc_range, 'add_order_cnt': add_order_cnt,
                'force_round': True,
                'ma_window': ma_win, 
                'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
                'label': '🎯 현재 설정'
            })
            if curr_res:
                entry = curr_res['Params'].copy()
                entry.update({'ID': 'MySet', 'CAGR': curr_res['CAGR'], 'MDD': curr_res['MDD'], 
                              'Score': curr_res['CAGR'] - abs(curr_res['MDD']), 'Label': '🎯 현재 설정'})
                st.session_state.opt_results.append(entry)

            prog = st.progress(0)
            for i in range(sim_count):
                st.session_state.trial_count += 1
                r_params = {
                    'start_date': start_date, 'end_date': end_date,
                    'initial_balance': balance, 'fee_rate': fee/100,
                    'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
                    'loc_range': loc_range, 'add_order_cnt': add_order_cnt,
                    'force_round': True,
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

        if st.session_state.opt_results:
            res_df = pd.DataFrame(st.session_state.opt_results)
            res_df = res_df.sort_values('Score', ascending=False).reset_index(drop=True)
            res_df.index += 1
            res_df.index.name = 'Rank'
            
            show_cols = ['Label', 'Score', 'CAGR', 'MDD', 'ma_window', 'bt_buy', 'bt_prof']
            st.dataframe(res_df[show_cols], height=300)
            
            options = []
            for idx, row in res_df.head(30).iterrows():
                lbl = f"[Rank {idx}] {row['Label']} (Score: {row['Score']:.2f} | CAGR: {row['CAGR']}%)"
                options.append(lbl)
            
            selected_opt = st.selectbox("결과를 선택하세요:", options)
            if selected_opt:
                rank_idx = int(selected_opt.split(']')[0].replace('[Rank ', ''))
                sel_row = res_df.loc[rank_idx]
                code_text = f"Selected Params:\n{sel_row.to_dict()}"
                st.code(code_text)
                
                if st.button("이 전략으로 심층 분석하기 ➡️"):
                    st.session_state.target_analysis_params = sel_row.to_dict()
                    st.success("심층 분석 탭으로 이동하세요!")

    # ==========================
    # 탭 3: 심층 분석
    # ==========================
    with tab3:
        st.subheader("🔬 전략 정밀 검진")
        
        target = None
        src = st.radio("분석 대상:", ["최근 백테스트 결과", "최적화에서 선택한 전략"])
        
        if src == "최근 백테스트 결과":
            if st.session_state.last_backtest_result:
                target = st.session_state.last_backtest_result['Params']
            else:
                st.warning("⚠️ 백테스트 탭에서 먼저 '실행'을 눌러주세요.")
        else: 
            if 'target_analysis_params' in st.session_state:
                target = st.session_state.target_analysis_params
            else:
                st.warning("⚠️ 최적화 탭에서 전략을 선택하고 '심층 분석하기' 버튼을 눌러주세요.")
        
        if target:
            res = backtest_engine_web(df, target)
            if res:
                st.metric("CAGR", f"{res['CAGR']}%", f"MDD {res['MDD']}%")
                yearly_df = pd.DataFrame(res['Yearly'])
                yearly_df.columns = ['Return %']
                yearly_df.index = yearly_df.index.strftime('%Y')
                st.bar_chart(yearly_df)

else:
    st.info("👈 왼쪽 사이드바에서 데이터 파일을 먼저 업로드해주세요.")