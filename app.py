import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- [기본 설정 값] ---
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dK11y5aTIhDGfpMduNsuSgTDlDoPo-OF6uE5FIePXVg/edit"

# --- [페이지 설정] ---
st.set_page_config(page_title="쪼꼬야옹 백테스트 연구소", page_icon="📈", layout="wide")

# --- [세션 상태 초기화] ---
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = []
if 'trial_count' not in st.session_state:
    st.session_state.trial_count = 0
if 'last_backtest_result' not in st.session_state:
    st.session_state.last_backtest_result = None

# --- [구글 시트 데이터 로드 함수 (독립 데이터 병합)] ---
@st.cache_data(ttl=600)
def load_data_from_gsheet(url):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        sheet = client.open_by_url(url)
        worksheet = sheet.get_worksheet(0)
        rows = worksheet.get_all_values()
        
        if not rows:
            st.error("❌ 시트가 비어있습니다.")
            return None

        # 1. 헤더 위치 찾기 (QQQ, SOXL)
        header_row_idx = -1
        idx_qqq = -1
        idx_soxl = -1
        
        for i, row in enumerate(rows[:20]): # 상위 20줄 검색
            if "QQQ" in row and "SOXL" in row:
                header_row_idx = i
                idx_qqq = row.index("QQQ")
                idx_soxl = row.index("SOXL")
                break
        
        if header_row_idx == -1:
            st.error("❌ 시트에서 'QQQ'와 'SOXL' 헤더를 찾을 수 없습니다.")
            return None

        # 2. 데이터 추출 함수 (날짜, 가격)
        def extract_series(data_rows, col_idx, name):
            # 헤더 아래(Date, Close) 다음 행부터 데이터 시작
            # QQQ/SOXL 헤더 -> 그 아래 Date/Close 헤더 -> 그 아래 실제 데이터
            start_row = header_row_idx + 2 
            
            extracted = []
            for r in data_rows[start_row:]:
                if len(r) > col_idx + 1:
                    d = r[col_idx]     # Date
                    p = r[col_idx + 1] # Close
                    if d and p: # 빈값 제외
                        extracted.append([d, p])
            
            df_temp = pd.DataFrame(extracted, columns=['Date', name])
            
            # 날짜 정제
            df_temp['Date'] = df_temp['Date'].astype(str).str.strip()
            df_temp['Date'] = df_temp['Date'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
            df_temp['Date'] = df_temp['Date'].str.replace('.', '-')
            
            # 연도 보정
            def fix_year(date_str):
                try:
                    parts = date_str.split('-')
                    if len(parts) == 3 and len(parts[0]) == 2:
                        return f"20{parts[0]}-{parts[1]}-{parts[2]}"
                    return date_str
                except: return date_str
            
            df_temp['Date'] = df_temp['Date'].apply(fix_year)
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
            
            # 가격 정제
            df_temp[name] = df_temp[name].astype(str).str.replace(',', '').str.replace('$', '')
            df_temp[name] = pd.to_numeric(df_temp[name], errors='coerce')
            
            df_temp.dropna(inplace=True)
            return df_temp

        # 3. QQQ와 SOXL 각각 추출
        df_qqq = extract_series(rows, idx_qqq, 'QQQ')
        df_soxl = extract_series(rows, idx_soxl, 'SOXL')

        # 4. 날짜 기준 병합 (Inner Join: 둘 다 데이터가 있는 날만)
        df_merged = pd.merge(df_qqq, df_soxl, on='Date', how='inner')
        
        df_merged.set_index('Date', inplace=True)
        df_merged.sort_index(inplace=True)
        
        if len(df_merged) == 0:
            st.error("❌ 날짜가 일치하는 데이터가 없습니다.")
            return None
            
        return df_merged

    except Exception as e:
        st.error(f"구글 시트 로드 실패: {e}")
        return None

# --- [유틸리티 함수] ---
def excel_round_up(n, decimals=0):
    if pd.isna(n) or n == np.inf or n == -np.inf: return 0
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier - 1e-9) / multiplier

def excel_round_down(n, decimals=0):
    if pd.isna(n) or n == np.inf or n == -np.inf: return 0
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 1e-9) / multiplier

def calculate_loc_quantity(seed_amount, order_price, close_price, buy_range, max_add_orders):
    if seed_amount is None or order_price is None or order_price <= 0: return 0
    if pd.isna(seed_amount) or pd.isna(order_price) or pd.isna(close_price): return 0

    base_qty = int(seed_amount / order_price)
    multiplier = (1 + buy_range) if buy_range <= 0 else (1 - buy_range)
    bot_price = excel_round_down(order_price * multiplier, 2)

    fix_qty = 0
    if bot_price > 0:
        qty_at_bot = seed_amount / bot_price
        qty_at_order = seed_amount / order_price
        fix_qty = int((qty_at_bot - qty_at_order) / max_add_orders)
    if fix_qty < 0: fix_qty = 0

    final_qty = 0
    if base_qty > 0:
        implied_price = seed_amount / base_qty
        if implied_price >= close_price and implied_price >= bot_price:
            final_qty += base_qty

    for i in range(1, max_add_orders + 1):
        step_qty = fix_qty
        current_cum_qty = base_qty + (i * step_qty)
        if current_cum_qty <= 0: continue
        implied_price = seed_amount / current_cum_qty
        if implied_price >= close_price and implied_price >= bot_price:
            final_qty += step_qty

    return final_qty

# --- [백테스트 엔진] ---
def backtest_engine_web(df, params):
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

    dates = df.index
    strategy = {
        'Bottom':  {'cond': params['bt_cond'], 'buy': params['bt_buy'], 'prof': params['bt_prof'], 'time': params['bt_time']},
        'Ceiling': {'cond': params['cl_cond'], 'buy': params['cl_buy'], 'prof': params['cl_prof'], 'time': params['cl_time']},
        'Middle':  {'cond': 999,           'buy': params['md_buy'], 'prof': params['md_prof'], 'time': params['md_time']}
    }
    
    cash = params['initial_balance']
    seed_equity = cash
    holdings = []
    
    trade_log = [] 
    daily_log = [] 
    daily_equity = []
    daily_dates = []
    trade_count = 0
    win_count = 0
    
    MAX_SLOTS = 10
    SEC_FEE = 0.0000278

    for i in range(len(df)):
        row = df.iloc[i]
        today_close = row['SOXL']
        if pd.isna(today_close) or today_close <= 0: continue
        if params.get('force_round', True): today_close = round(today_close, 2)

        disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        
        if disp < strategy['Bottom']['cond']: phase = 'Bottom'
        elif disp > strategy['Ceiling']['cond']: phase = 'Ceiling'
        else: phase = 'Middle'
        
        conf = strategy[phase]
        target_seed_float = seed_equity / MAX_SLOTS
        target_seed = int(target_seed_float + 0.5)

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
                cash += net_receive
                trade_count += 1
                if real_profit > 0: win_count += 1
                trade_log.append({
                    'Date': dates[i], 'Type': 'Sell', 'Tier': tier, 'Phase': mode,
                    'Price': today_close, 'Qty': qty, 'Profit': real_profit, 'Reason': reason
                })
            else:
                stock[1] = days
        
        if daily_net_profit_sum != 0:
            rate = params['profit_rate'] if daily_net_profit_sum > 0 else params['loss_rate']
            seed_equity += daily_net_profit_sum * rate
            
        prev_c = row['Prev_Close'] if not pd.isna(row['Prev_Close']) else today_close
        if pd.isna(prev_c): prev_c = today_close
            
        target_p = excel_round_down(prev_c * (1 + conf['buy'] / 100), 2)
        bet = min(target_seed_float, cash)
        if bet < 10: bet = 0
        
        if today_close <= target_p and len(holdings) < MAX_SLOTS and bet > 0:
            curr_tiers = {h[4] for h in holdings}
            unavail = curr_tiers.union(tiers_sold)
            new_tier = 1
            while new_tier in unavail: new_tier += 1
            
            if new_tier <= MAX_SLOTS:
                final_qty = 0
                if new_tier == MAX_SLOTS:
                    final_qty = int(bet / target_p)
                else:
                    final_qty = calculate_loc_quantity(
                        seed_amount=bet,
                        order_price=target_p,
                        close_price=today_close,
                        buy_range= -1 * (params['loc_range'] / 100.0),
                        max_add_orders=int(params['add_order_cnt'])
                    )
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
        
        current_eq = cash + sum([h[2]*today_close for h in holdings])
        daily_equity.append(current_eq)
        daily_dates.append(dates[i])
        daily_log.append({
            'Date': dates[i], 'Equity': round(current_eq, 2), 
            'Cash': round(cash, 2), 'SeedEquity': round(seed_equity, 2), 
            'Holdings': len(holdings)
        })

    if not daily_equity: return None

    final_equity = daily_equity[-1]
    total_ret_pct = (final_equity / params['initial_balance'] - 1) * 100
    days_total = (dates[-1] - dates[0]).days
    cagr = ((final_equity / params['initial_balance']) ** (365/days_total) - 1) * 100 if days_total > 0 else 0
    eq_series = pd.Series(daily_equity, index=daily_dates)
    peak = eq_series.cummax()
    mdd = ((eq_series / peak - 1) * 100).min()
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0
    
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
        'DailyLog': pd.DataFrame(daily_log),
	'CurrentHoldings': holdings,  # <--- 이 줄을 꼭 추가해주세요! (현재 보유 종목 리스트)
        'LastData': df.iloc[-1]
    }

# --- [UI 구성] ---
st.title("📊 쪼꼬야옹 백테스트 연구소")

with st.sidebar:
    st.header("⚙️ 기본 설정")
    sheet_url = st.text_input("🔗 구글 시트 주소 (URL)", value=DEFAULT_SHEET_URL)
    st.caption("※ 시트에 'Date', 'SOXL', 'QQQ' 데이터가 있어야 합니다.")
    st.subheader("💰 자산 및 복리 설정")
    balance = st.number_input("초기 자본 ($)", value=10000)
    fee = st.number_input("수수료 (%)", value=0.07)
    profit_rate = st.slider("이익 복리율 (%)", 0, 100, 70)
    loss_rate = st.slider("손실 복리율 (%)", 0, 100, 50)
    st.subheader("📥 LOC 설정")
    add_order_cnt = st.number_input("추가 주문 횟수", value=4, min_value=1) 
    loc_range = st.number_input("하단 범위 (-%)", value=20.0, min_value=0.0) 
    st.subheader("📈 기간 설정")
    start_date = st.date_input("시작일", pd.to_datetime("2014-01-01"))
    end_date = st.date_input("종료일", pd.to_datetime("2025-12-31"))

if sheet_url:
    df = load_data_from_gsheet(sheet_url)
    
    if df is not None:
        # [수정] 탭에 "대시보드"를 맨 앞에 추가합니다.
        tab0, tab1, tab2, tab3 = st.tabs(["📢 대시보드", "🚀 백테스트", "🎲 몬테카를로 최적화", "🔬 심층 분석"])
        
        # 탭 1: 백테스트
        with tab1:
            st.subheader("🛠️ 전략 파라미터 입력")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### 📉 바닥 (Bottom)")
                bt_cond = st.number_input("기준 이격도", 0.8, 1.0, 0.90, step=0.01)
                bt_buy = st.number_input("매수점 (%)", -30.0, 30.0, 15.0, step=0.1, key='bt_b')
                bt_prof = st.number_input("익절 (%)", 0.0, 100.0, 2.5, step=0.1, key='bt_p')
                bt_time = st.number_input("존버일", 1, 100, 10, key='bt_t')
            with col2:
                st.markdown("##### ➖ 중간 (Middle)")
                md_buy = st.number_input("매수점 (%)", -30.0, 30.0, -0.01, step=0.1, key='md_b')
                md_prof = st.number_input("익절 (%)", 0.0, 100.0, 2.8, step=0.1, key='md_p')
                md_time = st.number_input("존버일", 1, 100, 15, key='md_t')
            with col3:
                st.markdown("##### 📈 천장 (Ceiling)")
                cl_cond = st.number_input("기준 이격도", 1.0, 1.5, 1.10, step=0.01)
                cl_buy = st.number_input("매수점 (%)", -30.0, 30.0, -0.1, step=0.1, key='cl_b')
                cl_prof = st.number_input("익절 (%)", 0.0, 100.0, 1.5, step=0.1, key='cl_p')
                cl_time = st.number_input("존버일", 1, 100, 40, key='cl_t')
            ma_win = st.number_input("이평선 (MA)", 50, 300, 200)

            if st.button("백테스트 실행 (Run)", type="primary"):
                current_params = {
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
                }
                res = backtest_engine_web(df, current_params)
                if res:
                    st.session_state.last_backtest_result = res
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("최종 자산", f"${res['Final']:,.0f}", f"{res['Return']}%")
                    m2.metric("CAGR (연평균)", f"{res['CAGR']}%")
                    m3.metric("MDD (최대낙폭)", f"{res['MDD']}%")
                    m4.metric("승률 / 횟수", f"{res['WinRate']}%", f"{res['Trades']}회")
                    
                    c_d1, c_d2 = st.columns(2)
                    csv_trade = res['TradeLog'].to_csv(index=False).encode('utf-8-sig')
                    c_d1.download_button("📥 매매일지 다운로드", csv_trade, "trade_log.csv", "text/csv")
                    csv_daily = res['DailyLog'].to_csv(index=False).encode('utf-8-sig')
                    c_d2.download_button("📥 자산일지 다운로드", csv_daily, "daily_log.csv", "text/csv")

                    st.line_chart(res['Series'])
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
                else:
                    st.error("백테스트 결과가 없습니다. (날짜 범위 또는 데이터 확인 필요)")

        # 탭 2: 몬테카를로
        with tab2:
            st.header("🎲 최적 파라미터 탐색기")
            st.info("💡 범위를 입력하면 AI가 그 안에서 최고의 조합을 찾아냅니다.")
            
            with st.container(border=True):
                c_base1, c_base2 = st.columns(2)
                with c_base1:
                    sim_count = st.number_input("🚀 시도 횟수 (Trial)", min_value=10, max_value=10000, value=100, step=10)
                with c_base2:
                    st.write("📊 이평선 범위 (MA Window)")
                    c_ma1, c_ma2 = st.columns(2)
                    ma_min = c_ma1.number_input("최소 MA", 50, 300, 120)
                    ma_max = c_ma2.number_input("최대 MA", 50, 300, 250)

            st.subheader("🎛️ 모드별 파라미터 범위 설정")
            col_bt, col_md, col_cl = st.columns(3)
            
            with col_bt:
                with st.container(border=True):
                    st.markdown("#### 📉 바닥 (Bottom)")
                    st.markdown("---")
                    bt_cond_min = st.number_input("B-이격 최소", 0.8, 1.0, 0.90, step=0.01)
                    bt_cond_max = st.number_input("B-이격 최대", 0.8, 1.0, 0.99, step=0.01)
                    c_b1, c_b2 = st.columns(2)
                    bt_buy_min = c_b1.number_input("B-매수 최소", -50.0, 50.0, 10.0, step=0.1)
                    bt_buy_max = c_b2.number_input("B-매수 최대", -50.0, 50.0, 20.0, step=0.1)
                    c_p1, c_p2 = st.columns(2)
                    bt_prof_min = c_p1.number_input("B-익절 최소", 0.0, 100.0, 1.0, step=0.1)
                    bt_prof_max = c_p2.number_input("B-익절 최대", 0.0, 100.0, 5.0, step=0.1)
                    c_t1, c_t2 = st.columns(2)
                    bt_time_min = c_t1.number_input("B-존버 최소", 1, 100, 5)
                    bt_time_max = c_t2.number_input("B-존버 최대", 1, 100, 20)

            with col_md:
                with st.container(border=True):
                    st.markdown("#### ➖ 중간 (Middle)")
                    st.markdown("---")
                    st.info("바닥과 천장 사이 구간")
                    st.write("") 
                    st.write("") 
                    c_b1, c_b2 = st.columns(2)
                    md_buy_min = c_b1.number_input("M-매수 최소", -50.0, 50.0, -5.0, step=0.1)
                    md_buy_max = c_b2.number_input("M-매수 최대", -50.0, 50.0, 5.0, step=0.1)
                    c_p1, c_p2 = st.columns(2)
                    md_prof_min = c_p1.number_input("M-익절 최소", 0.0, 100.0, 3.0, step=0.1)
                    md_prof_max = c_p2.number_input("M-익절 최대", 0.0, 100.0, 10.0, step=0.1)
                    c_t1, c_t2 = st.columns(2)
                    md_time_min = c_t1.number_input("M-존버 최소", 1, 100, 10)
                    md_time_max = c_t2.number_input("M-존버 최대", 1, 100, 30)

            with col_cl:
                with st.container(border=True):
                    st.markdown("#### 📈 천장 (Ceiling)")
                    st.markdown("---")
                    cl_cond_min = st.number_input("C-이격 최소", 1.0, 1.5, 1.01, step=0.01)
                    cl_cond_max = st.number_input("C-이격 최대", 1.0, 1.5, 1.15, step=0.01)
                    c_b1, c_b2 = st.columns(2)
                    cl_buy_min = c_b1.number_input("C-매수 최소", -50.0, 50.0, -10.0, step=0.1)
                    cl_buy_max = c_b2.number_input("C-매수 최대", -50.0, 50.0, 5.0, step=0.1)
                    c_p1, c_p2 = st.columns(2)
                    cl_prof_min = c_p1.number_input("C-익절 최소", 0.0, 100.0, 1.0, step=0.1)
                    cl_prof_max = c_p2.number_input("C-익절 최대", 0.0, 100.0, 5.0, step=0.1)
                    c_t1, c_t2 = st.columns(2)
                    cl_time_min = c_t1.number_input("C-존버 최소", 1, 100, 20)
                    cl_time_max = c_t2.number_input("C-존버 최대", 1, 100, 50)

            st.markdown("---")
            col_btn1, col_btn2 = st.columns([1, 4])
            
            if col_btn1.button("🚀 최적화 시작", type="primary", use_container_width=True):
                st.session_state.opt_results = [r for r in st.session_state.opt_results if r.get('Label') != '🎯 현재 설정']

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
                status_text = st.empty()
                
                for i in range(sim_count):
                    st.session_state.trial_count += 1
                    status_text.text(f"⏳ 탐색 중... ({i+1}/{sim_count})")
                    
                    r_params = {
                        'start_date': start_date, 'end_date': end_date,
                        'initial_balance': balance, 'fee_rate': fee/100,
                        'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
                        'loc_range': loc_range, 'add_order_cnt': add_order_cnt,
                        'force_round': True,
                        'ma_window': np.random.randint(ma_min, ma_max + 1),
                        'bt_cond': round(np.random.uniform(bt_cond_min, bt_cond_max), 2),
                        'bt_buy': round(np.random.uniform(bt_buy_min, bt_buy_max), 1),
                        'bt_prof': round(np.random.uniform(bt_prof_min, bt_prof_max)/100, 4),
                        'bt_time': np.random.randint(bt_time_min, bt_time_max + 1),
                        'md_buy': round(np.random.uniform(md_buy_min, md_buy_max), 1),
                        'md_prof': round(np.random.uniform(md_prof_min, md_prof_max)/100, 4),
                        'md_time': np.random.randint(md_time_min, md_time_max + 1),
                        'cl_cond': round(np.random.uniform(cl_cond_min, cl_cond_max), 2),
                        'cl_buy': round(np.random.uniform(cl_buy_min, cl_buy_max), 1),
                        'cl_prof': round(np.random.uniform(cl_prof_min, cl_prof_max)/100, 4),
                        'cl_time': np.random.randint(cl_time_min, cl_time_max + 1),
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
                status_text.text("✅ 탐색 완료!")
                time.sleep(1)
                status_text.empty()
                prog.empty()

            if col_btn2.button("🗑️ 결과 초기화"):
                st.session_state.opt_results = []
                st.session_state.trial_count = 0
                st.rerun()

            if st.session_state.opt_results:
                st.markdown("### 🏆 Top 랭킹 (Score 기준)")
                res_df = pd.DataFrame(st.session_state.opt_results)
                res_df = res_df.sort_values('Score', ascending=False).reset_index(drop=True)
                res_df.index += 1
                res_df.index.name = 'Rank'
                
                show_cols = ['Label', 'Score', 'CAGR', 'MDD', 'ma_window', 'bt_buy', 'bt_prof']
                def highlight_myset(s):
                    return ['background-color: #FFF8DC' if s['Label'] == '🎯 현재 설정' else '' for _ in s]
                st.dataframe(res_df[show_cols].style.apply(highlight_myset, axis=1), height=300, use_container_width=True)
                
                st.markdown("---")
                c_sel1, c_sel2 = st.columns([3, 1])
                with c_sel1:
                    options = []
                    for idx, row in res_df.head(50).iterrows():
                        lbl = f"[Rank {idx}] {row['Label']} (Score: {row['Score']:.2f} | CAGR: {row['CAGR']}%)"
                        options.append(lbl)
                    selected_opt = st.selectbox("🔍 결과 선택 (상세 파라미터 확인)", options)
                
                with c_sel2:
                    st.write("") 
                    st.write("")
                    if st.button("👉 심층 분석하기", type='primary'):
                        if selected_opt:
                            rank_idx = int(selected_opt.split(']')[0].replace('[Rank ', ''))
                            sel_row = res_df.loc[rank_idx]
                            st.session_state.target_analysis_params = sel_row.to_dict()
                            st.toast("✅ 전략이 선택되었습니다! '심층 분석' 탭으로 이동하세요.")

                if selected_opt:
                    rank_idx = int(selected_opt.split(']')[0].replace('[Rank ', ''))
                    sel_row = res_df.loc[rank_idx]
                    code_text = f"""# === [Rank {rank_idx}] {sel_row['Label']} 파라미터 ===
# Score: {sel_row['Score']:.2f} | CAGR: {sel_row['CAGR']}% | MDD: {sel_row['MDD']}%

MY_BEST_PARAMS = {{
    'ma_window': {sel_row['ma_window']},
    'bt_cond': {sel_row['bt_cond']:.2f}, 'bt_buy': {sel_row['bt_buy']}, 'bt_prof': {sel_row['bt_prof']*100:.1f}, 'bt_time': {sel_row['bt_time']},
    'md_buy': {sel_row['md_buy']}, 'md_prof': {sel_row['md_prof']*100:.1f}, 'md_time': {sel_row['md_time']},
    'cl_cond': {sel_row['cl_cond']:.2f}, 'cl_buy': {sel_row['cl_buy']}, 'cl_prof': {sel_row['cl_prof']*100:.1f}, 'cl_time': {sel_row['cl_time']}
}}"""
                    st.code(code_text, language='python')

        # 탭 3: 심층 분석
        with tab3:
            st.subheader("🔬 전략 정밀 검진")
            target = None
            src = st.radio("분석 대상:", ["최근 백테스트 결과", "최적화에서 선택한 전략"], horizontal=True)
            
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
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("CAGR", f"{res['CAGR']}%")
                    k2.metric("MDD", f"{res['MDD']}%")
                    k3.metric("승률", f"{res['WinRate']}%")
                    k4.metric("거래횟수", f"{res['Trades']}회")
                    
                    st.markdown("#### 📅 연도별 수익률 상세")
                    
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
                    
                    yearly_df = pd.DataFrame(res['Yearly'])
                    yearly_df.columns = ['Return %']
                    yearly_df.index = yearly_df.index.strftime('%Y')
                    st.dataframe(yearly_df.style.background_gradient(cmap='RdBu_r', vmin=-50, vmax=50), use_container_width=True)
 # --- [탭 0: 실전 투자 대시보드] ---
        with tab0:
            st.header("📢 오늘의 투자 브리핑")
            
            # 대시보드를 보려면 백테스트가 한 번은 돌아가야 현재 상태를 알 수 있습니다.
            # 가장 최근 설정값(없으면 기본값)으로 백테스트를 실행합니다.
            dash_params = {
                'start_date': start_date, 'end_date': end_date,
                'initial_balance': balance, 'fee_rate': fee/100,
                'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
                'loc_range': loc_range, 'add_order_cnt': add_order_cnt,
                'force_round': True, 'ma_window': ma_win, 
                'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time
            }
            
            # 조용히 백테스트 실행하여 최신 상태 가져오기
            res = backtest_engine_web(df, dash_params)
            
            if res:
                last_row = res['LastData']
                last_date = last_row.name.strftime('%Y-%m-%d')
                current_holdings = res['CurrentHoldings']
                
                # 1. 상단 요약 정보
                st.info(f"📅 기준 날짜: **{last_date}** (데이터 마지막 업데이트)")
                
                k1, k2, k3, k4 = st.columns(4)
                current_cash = res['DailyLog'].iloc[-1]['Cash']
                total_equity = res['DailyLog'].iloc[-1]['Equity']
                
                # 현재 구간(Phase) 판단
                disp = last_row['Basis_Disp']
                if disp < dash_params['bt_cond']: curr_phase = "📉 바닥 (Bottom)"
                elif disp > dash_params['cl_cond']: curr_phase = "📈 천장 (Ceiling)"
                else: curr_phase = "➖ 중간 (Middle)"

                k1.metric("현재 총 자산", f"${total_equity:,.0f}")
                k2.metric("보유 현금 (주문가능)", f"${current_cash:,.0f}")
                k3.metric("현재 이격도", f"{disp:.4f}")
                k4.metric("현재 구간", curr_phase)
                
                st.markdown("---")

                # 2. 오늘의 매수/매도 주문 (핵심)
                c_buy, c_sell = st.columns(2)
                
                with c_buy:
                    st.subheader("🛒 오늘 매수할 주문 (LOC)")
                    
                    # 매수 타겟 계산
                    if "바닥" in curr_phase: 
                        target_rate = dash_params['bt_buy']
                    elif "천장" in curr_phase: 
                        target_rate = dash_params['cl_buy']
                    else: 
                        target_rate = dash_params['md_buy']
                        
                    target_price = excel_round_down(last_row['SOXL'] * (1 + target_rate/100.0), 2)
                    
                    # 매수 수량 계산 (현금 범위 내)
                    # 시드 분할 로직 (최대 슬롯 10개 가정)
                    target_seed = total_equity / 10
                    bet_amount = min(target_seed, current_cash)
                    
                    if len(current_holdings) >= 10:
                        st.warning("🚫 보유 슬롯이 꽉 찼습니다 (10/10). 추가 매수 금지.")
                    elif bet_amount < 10:
                        st.warning("🚫 주문 가능 현금이 부족합니다.")
                    else:
                        st.success(f"**LOC 매수**를 준비하세요.")
                        st.markdown(f"""
                        - **매수 기준가**: ${target_price} ({target_rate}%)
                        - **LOC 하단(-{dash_params['loc_range']}%)**: ${excel_round_down(target_price * (1 - dash_params['loc_range']/100), 2)}
                        - **예상 투입 금액**: ${bet_amount:,.0f}
                        """)
                        st.caption("※ 실제 수량은 장마감 직전 가격에 따라 LOC 로직으로 결정됩니다.")

                with c_sell:
                    st.subheader("💰 매도 대기 물량 (지정가)")
                    if not current_holdings:
                        st.write("보유 중인 종목이 없습니다.")
                    else:
                        sell_list = []
                        for h in current_holdings:
                            # holdings 구조: [buy_price, days, qty, mode, tier, buy_dt]
                            buy_p, days, qty, mode, tier, buy_dt = h
                            
                            # 익절 목표가 계산
                            if mode == 'Bottom': prof_rate = dash_params['bt_prof']
                            elif mode == 'Ceiling': prof_rate = dash_params['cl_prof']
                            else: prof_rate = dash_params['md_prof']
                            
                            target_sell_p = excel_round_up(buy_p * (1 + prof_rate), 2)
                            curr_return = (last_row['SOXL'] - buy_p) / buy_p * 100
                            
                            sell_list.append({
                                'Tier': tier,
                                '매수일': buy_dt.strftime('%Y-%m-%d'),
                                '보유일': f"{days}일",
                                '매수가': f"${buy_p}",
                                '수량': qty,
                                '🎯 매도목표가': f"${target_sell_p}",
                                '현재수익률': f"{curr_return:.2f}%"
                            })
                        st.dataframe(pd.DataFrame(sell_list), hide_index=True, use_container_width=True)

                st.markdown("---")

                # 3. 최근 매매 기록 (어제 체결 & 최근 1달)
                st.subheader("📜 최근 매매 일지")
                
                trade_log_df = res['TradeLog']
                if not trade_log_df.empty:
                    # 날짜 내림차순 정렬
                    trade_log_df = trade_log_df.sort_values('Date', ascending=False)
                    
                    # 어제(가장 최근 데이터 날짜) 체결 내역
                    last_trade_date = trade_log_df.iloc[0]['Date']
                    if last_trade_date == last_row.name:
                        st.write(f"🔔 **최근 체결 알림 ({last_trade_date.strftime('%Y-%m-%d')})**")
                        recent_trades = trade_log_df[trade_log_df['Date'] == last_trade_date]
                        st.dataframe(recent_trades, hide_index=True, use_container_width=True)
                    else:
                        st.write(f"🔔 가장 최근 데이터 날짜 ({last_date})에는 체결된 내역이 없습니다.")
                    
                    with st.expander("🗓️ 최근 30일간 매매 전체 보기"):
                        month_ago = last_row.name - pd.Timedelta(days=30)
                        recent_month_log = trade_log_df[trade_log_df['Date'] >= month_ago]
                        st.dataframe(recent_month_log, hide_index=True, use_container_width=True)
                else:
                    st.info("아직 체결된 매매 기록이 없습니다.")

else:
    st.warning("👈 왼쪽 사이드바에 구글 시트 주소를 입력하거나, CSV 파일을 업로드해주세요.")