import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
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
        df_merged = pd.merge(df_qqq, df_soxl, on='Date', how='left')
        
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

# [백테스트 엔진] 수정본
def backtest_engine_web(df, params):
    df = df.copy()
    
    # ------------------------------------------------------------------
    # [데이터 전처리]
    # ------------------------------------------------------------------
    df['QQQ'] = pd.to_numeric(df['QQQ'], errors='coerce')
    ma_win = int(params['ma_window'])
    
    # 이평선 계산
    df['MA_Daily'] = df['QQQ'].rolling(window=ma_win, min_periods=1).mean()
    df['Log_Start_Price'] = df['QQQ'].shift(ma_win - 1)

    # ------------------------------------------------------------------
    # [3. 주간 데이터(Weekly) 추출 방식 개선] - 휴장일 대응 로직 적용
    # ------------------------------------------------------------------
    weekly_resampled = df[['QQQ', 'MA_Daily', 'Log_Start_Price']].resample('W-FRI').last()
    weekly_resampled.columns = ['QQQ_Fri', 'MA_Fri', 'Start_Price_Fri']
    weekly_resampled['Disp_Fri'] = weekly_resampled['QQQ_Fri'] / weekly_resampled['MA_Fri']
    
    daily_expanded = weekly_resampled.resample('D').ffill()
    daily_shifted = daily_expanded.shift(1)
    df_mapped = daily_shifted.reindex(df.index)
    
    df['Basis_Disp']      = df_mapped['Disp_Fri'].fillna(1.0)
    df['Log_Ref_Date']    = daily_shifted['QQQ_Fri'].reindex(df.index).index 
    df['Log_QQQ_Fri']     = df_mapped['QQQ_Fri']
    df['Log_MA_Fri']      = df_mapped['MA_Fri']
    df['Log_Start_Price'] = df_mapped['Start_Price_Fri']
    df['Prev_Close'] = df['SOXL'].shift(1)
    
    start_dt = pd.to_datetime(params['start_date'])
    end_dt = pd.to_datetime(params['end_date'])
    df = df.sort_index()
    df = df[(df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))].copy()
    df = df.dropna(subset=['SOXL'])  

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

    # [수정] for문 앞에 공백 4칸을 추가하여 함수 내부로 들여쓰기 했습니다.
    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        
        # [추가] 오늘 아침에 가진 돈을 기록해둡니다. (장중 매도로 늘어나도 이건 변하지 않음)
        start_cash = cash
        
        today_close = row['SOXL']
        if pd.isna(today_close) or today_close <= 0: continue
        if params.get('force_round', True): 
            today_close = round(today_close, 2)
        
        disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        
        if disp < params['bt_cond']: phase = 'Bottom'
        elif disp > params['cl_cond']: phase = 'Ceiling'
        else: phase = 'Middle'

        conf = strategy[phase]
        tiers_sold = set()
        daily_net_profit_sum = 0
        
        # 1. 매도 로직
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
                
                # 순수익 합산 (투자금 갱신은 매수 이후로 미룸)
                daily_net_profit_sum += real_profit
                cash += net_receive
                
                trade_count += 1
                if real_profit > 0: win_count += 1
                trade_log.append({
                    'Date': dates[i], 'Type': 'Sell', 'Tier': tier, 'Phase': mode, 
                    'Ref_Date': row['Log_Ref_Date'].strftime('%Y-%m-%d') if pd.notnull(row['Log_Ref_Date']) else '-',
                    'QQQ_Fri': row['Log_QQQ_Fri'], 'MA_Calc': row['Log_MA_Fri'], 'Disp': disp,
                    'Start_P': row['Log_Start_Price'], 'Price': today_close, 'Qty': qty, 
                    'Profit': real_profit, 'Reason': reason
                })
            else:
                stock[1] = days
        
        # 2. 매수 로직
        prev_c = row['Prev_Close'] if not pd.isna(row['Prev_Close']) else today_close
        if pd.isna(prev_c): prev_c = today_close
        
        # [중요] 매수 목표가 반올림 적용 (776개 -> 779개로 교정됨)
        target_p = excel_round_down(prev_c * (1 + conf['buy'] / 100), 2)
        
        if today_close <= target_p and len(holdings) < MAX_SLOTS:
            curr_tiers = {h[4] for h in holdings}
            unavail = curr_tiers.union(tiers_sold)
            new_tier = 1
            while new_tier in unavail: new_tier += 1
            
            if new_tier <= MAX_SLOTS:
                weight_pct = 10.0
                if 'tier_weights' in params:
                    try:
                        weight_pct = params['tier_weights'].loc[f'Tier {new_tier}', phase]
                    except:
                        weight_pct = 10.0
                
                target_seed = seed_equity * (weight_pct / 100.0)
                
                # [수정] 당일 매도로 생긴 현금(cash)이 아니라 아침 현금(start_cash) 한도 내에서만 배팅
                bet = min(target_seed, start_cash)
                
                # [수수료 안전 마진] 수수료가 0이라도 수식은 유지 (안전성 확보)
                bet_net_fee = bet / (1 + params['fee_rate'])
                
                if bet >= 10:
                    final_qty = 0
                    if new_tier == MAX_SLOTS:
                        final_qty = int(bet_net_fee / target_p)
                    else:
                        final_qty = calculate_loc_quantity(
                            seed_amount=bet_net_fee,
                            order_price=target_p,
                            close_price=today_close,
                            buy_range= -1 * (params['loc_range'] / 100.0),
                            max_add_orders=int(params['add_order_cnt'])
                        )
                    
                    # [수정] 최대 매수 가능 수량도 '아침 예수금' 기준으로 제한
                    max_buyable = int(start_cash / (today_close * (1 + params['fee_rate']))) 
                    real_qty = min(final_qty, max_buyable)
                    
                    if real_qty > 0:
                        buy_amt = today_close * real_qty * (1 + params['fee_rate'])
                        cash -= buy_amt # 실제 돈은 줄어듭니다.
                        # start_cash는 줄이지 않습니다 (하루 한 번 진입 규칙이 있다면)
                        
                        holdings.append([today_close, 0, real_qty, phase, new_tier, dates[i]])
                        trade_log.append({
                            'Date': dates[i], 'Type': 'Buy', 'Tier': new_tier, 'Phase': phase, 
                            'Ref_Date': row['Log_Ref_Date'].strftime('%Y-%m-%d') if pd.notnull(row['Log_Ref_Date']) else '-',
                            'QQQ_Fri': row['Log_QQQ_Fri'], 'MA_Calc': row['Log_MA_Fri'], 'Disp': disp,
                            'Start_P': row['Log_Start_Price'], 'Price': today_close, 'Qty': real_qty, 
                            'Profit': 0, 'Reason': 'LOC'
                        })
        
        # 3. [위치 이동] 투자금(Seed Equity) 갱신
        if daily_net_profit_sum != 0:
            rate = params['profit_rate'] if daily_net_profit_sum > 0 else params['loss_rate']
            seed_equity += daily_net_profit_sum * rate
        
        current_eq = cash + sum([h[2]*today_close for h in holdings])
        daily_equity.append(current_eq)
        daily_dates.append(dates[i])
        daily_log.append({
            'Date': dates[i], 'Equity': round(current_eq, 2), 
            'Cash': round(cash, 2), 'SeedEquity': round(seed_equity, 2), 
            'Holdings': len(holdings)
        })

    # 여기 IndentationError가 났던 부분입니다. for문이 들여쓰기 되면 여기도 정상 작동합니다.
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
	    'CurrentHoldings': holdings,
        'LastData': df.iloc[-1]
    }

# --- [UI 구성] ---
# ... (위쪽의 backtest_engine_web 함수까지는 그대로 두세요) ...

# --- [UI 구성] ---
st.title("📊 쪼꼬야옹의 듀얼 전략 연구소")

# 전역 설정 (시트 주소와 수수료는 공통으로 유지 - 필요시 이것도 분리 가능)
with st.sidebar:
    st.header("⚙️ 기본 데이터 연동")
    sheet_url = st.text_input("🔗 구글 시트 주소", value=DEFAULT_SHEET_URL)
    st.caption("※ 시트에 'Date', 'SOXL', 'QQQ' 데이터가 있어야 합니다.")
    
    st.markdown("---")
    st.header("⚔️ 전략별 상세 설정")
    
    # 탭으로 안정형/공격형 완벽 분리
    tab_s, tab_a = st.tabs(["🛡️ 안정형", "🔥 공격형"])

    # === [함수] 파라미터 입력 위젯 생성기 (자본, 기간 포함) ===
    def render_strategy_inputs(suffix, key_prefix):
        st.subheader(f"📊 {key_prefix} 기본 설정")
        
        # [수정] 자본과 기간을 여기로 이동 (독립 설정)
        balance = st.number_input(f"초기 자본 ($)", value=10000, key=f"bal_{suffix}")
        
        # 날짜 설정
        today = datetime.date.today()
        c_d1, c_d2 = st.columns(2)
        start_date = c_d1.date_input("시작일", value=datetime.date(2010, 1, 1), max_value=today, key=f"sd_{suffix}")
        end_date = c_d2.date_input("종료일", value=today, max_value=today, key=f"ed_{suffix}")
        
        st.markdown("---")
        st.write("⚙️ **파라미터 설정**")
        
        # 수수료는 편의상 공통값(0.07)을 기본으로 하되 수정 가능하게
        fee = st.number_input("수수료 (%)", value=0.07, step=0.01, format="%.2f", key=f"fee_{suffix}")
        
        profit_rate = st.slider("이익 복리율 (%)", 0, 100, 70, key=f"pr_{suffix}")
        loss_rate = st.slider("손실 복리율 (%)", 0, 100, 50, key=f"lr_{suffix}")
        
        c_loc1, c_loc2 = st.columns(2)
        add_order_cnt = c_loc1.number_input("분할 횟수", value=4, min_value=1, key=f"add_{suffix}") 
        loc_range = c_loc2.number_input("LOC 범위 (-%)", value=20.0, min_value=0.0, key=f"rng_{suffix}")
        ma_win = st.number_input("이평선 (MA)", 50, 300, 200, key=f"ma_{suffix}")

        st.markdown("##### 📉 바닥 (Bottom)")
        c1, c2 = st.columns(2)
        bt_cond = c1.number_input("기준 이격", 0.8, 1.0, 0.90, step=0.01, key=f"bc_{suffix}")
        bt_buy = c2.number_input("매수점%", -30.0, 30.0, 15.0, step=0.1, key=f"bb_{suffix}")
        bt_prof = c1.number_input("익절%", 0.0, 100.0, 2.5, step=0.1, key=f"bp_{suffix}")
        bt_time = c2.number_input("존버일", 1, 100, 10, key=f"bt_{suffix}")

        st.markdown("##### ➖ 중간 (Middle)")
        c3, c4 = st.columns(2)
        md_buy = c3.number_input("매수점%", -30.0, 30.0, -0.01, step=0.1, key=f"mb_{suffix}")
        md_prof = c4.number_input("익절%", 0.0, 100.0, 2.8, step=0.1, key=f"mp_{suffix}")
        md_time = c3.number_input("존버일", 1, 100, 15, key=f"mt_{suffix}")

        st.markdown("##### 📈 천장 (Ceiling)")
        c5, c6 = st.columns(2)
        cl_cond = c5.number_input("기준 이격", 1.0, 1.5, 1.10, step=0.01, key=f"cc_{suffix}")
        cl_buy = c6.number_input("매수점%", -30.0, 30.0, -0.1, step=0.1, key=f"cb_{suffix}")
        cl_prof = c5.number_input("익절%", 0.0, 100.0, 1.5, step=0.1, key=f"cp_{suffix}")
        cl_time = c6.number_input("존버일", 1, 100, 40, key=f"ct_{suffix}")
        
        st.markdown("---")
        st.write("⚖️ **티어별 비중**")
        default_w = pd.DataFrame({
            'Tier': [f'Tier {i}' for i in range(1, 11)],
            'Bottom': [10.0] * 10, 'Middle': [10.0] * 10, 'Ceiling': [10.0] * 10
        }).set_index('Tier')
        
        edited_w = st.data_editor(
            default_w,
            key=f"w_{suffix}",
            column_config={
                "Bottom": st.column_config.NumberColumn("바닥%", format="%.1f%%"),
                "Middle": st.column_config.NumberColumn("중간%", format="%.1f%%"),
                "Ceiling": st.column_config.NumberColumn("천장%", format="%.1f%%"),
            }, use_container_width=True
        )
        return {
            'start_date': start_date, 'end_date': end_date, # [핵심] 독립된 날짜 반환
            'initial_balance': balance,                     # [핵심] 독립된 자본 반환
            'fee_rate': fee/100,
            'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
            'loc_range': loc_range, 'add_order_cnt': add_order_cnt,
            'force_round': True, 'ma_window': ma_win, 
            'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
            'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
            'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
            'tier_weights': edited_w,
            'label': key_prefix
        }

    # 1. 안정형 설정 (Suffix: s)
    with tab_s:
        params_s = render_strategy_inputs('s', '🛡️ 안정형')

    # 2. 공격형 설정 (Suffix: a)
    with tab_a:
        params_a = render_strategy_inputs('a', '🔥 공격형')


if sheet_url:
    df = load_data_from_gsheet(sheet_url)
    
    if df is not None:
        tab_dash, tab_bt = st.tabs(["📢 듀얼 대시보드", "🚀 성과 비교"])

        # ==========================================
        # 탭 1: 듀얼 대시보드 (오늘의 주문)
        # ==========================================
        with tab_dash:
            st.header(f"📢 오늘의 투자 브리핑 ({df.index[-1].strftime('%Y-%m-%d')})")
            
            col_stable, col_agg = st.columns(2)
            
            # --- 대시보드 출력용 함수 ---
            def render_dashboard(col, p_params, strategy_name):
                with col:
                    st.subheader(f"{strategy_name}")
                    
                    # [중요] 각 전략의 start_date/balance로 백테스트 실행
                    res = backtest_engine_web(df, p_params)
                    if not res:
                        st.error("데이터 부족 (기간 확인)")
                        return

                    last_row = res['LastData']
                    daily_last = res['DailyLog'].iloc[-1]
                    current_cash = daily_last['Cash']
                    seed_equity_basis = daily_last['SeedEquity']
                    current_holdings = res['CurrentHoldings']
                    
                    disp = last_row['Basis_Disp']
                    if disp < p_params['bt_cond']: curr_phase = "📉 바닥"
                    elif disp > p_params['cl_cond']: curr_phase = "📈 천장"
                    else: curr_phase = "➖ 중간"
                    
                    # 요약 지표 (시드 자산 기준 표기)
                    st.metric("시드 자산 (확정)", f"${seed_equity_basis:,.0f}")
                    st.metric("보유 현금", f"${current_cash:,.0f}")
                    st.caption(f"이격도: {disp:.4f} ({curr_phase}) | 초기자본: ${p_params['initial_balance']:,}")
                    st.divider()

                    # 매수 주문 로직
                    n_split = int(p_params['add_order_cnt'])
                    loc_range = p_params['loc_range']
                    next_tier = min(len(current_holdings) + 1, 10)
                    
                    if "바닥" in curr_phase: col_key = "Bottom"; start_rate = p_params['bt_buy']
                    elif "천장" in curr_phase: col_key = "Ceiling"; start_rate = p_params['cl_buy']
                    else: col_key = "Middle"; start_rate = p_params['md_buy']
                    
                    try: target_weight = p_params['tier_weights'].loc[f'Tier {next_tier}', col_key]
                    except: target_weight = 10.0
                    
                    # 1회 시드 계산 (확정 자산 기준)
                    one_time_seed = seed_equity_basis * (target_weight / 100.0)
                    
                    base_price = last_row['SOXL']
                    loc_price = excel_round_down(base_price * (1 + start_rate/100.0), 2)

                    # Smart LOC 내부 함수
                    def get_smart_orders(seed, start_p, range_pct, split_cnt):
                        orders = []
                        if start_p <= 0: return orders
                        base_qty = int(seed / start_p)
                        orders.append({'price': start_p, 'qty': base_qty, 'type': 'MAIN'})
                        if split_cnt <= 0: return orders
                        multiplier = (1 + range_pct) if range_pct <= 0 else (1 - range_pct)
                        bot_p = excel_round_down(start_p * multiplier, 2)
                        if bot_p <= 0: return orders
                        qty_at_bot = seed / bot_p
                        qty_at_top = seed / start_p
                        fix_qty = int((qty_at_bot - qty_at_top) / split_cnt)
                        if fix_qty < 0: fix_qty = 0
                        for i in range(1, split_cnt + 1):
                            target_cum_qty = base_qty + (i * fix_qty)
                            if target_cum_qty > 0:
                                next_p = excel_round_down(seed / target_cum_qty, 2)
                                if next_p > 0 and next_p < start_p:
                                    orders.append({'price': next_p, 'qty': fix_qty, 'type': 'ADD'})
                        return orders

                    # [A] 신규 진입 출력
                    if len(current_holdings) < 10:
                        st.info(f"🆕 **신규 진입 (Tier {next_tier})**")
                        # [핵심] 1회 시드 vs 보유 현금 중 작은 값 사용 (수수료 제외)
                        real_bet = min(one_time_seed, current_cash)
                        net_bet = real_bet / (1 + p_params['fee_rate'])
                        
                        orders = get_smart_orders(net_bet, loc_price, -1*(loc_range/100.0), n_split)
                        
                        rem_cash = current_cash
                        total_est = 0
                        for o in orders:
                            cost = o['price']*o['qty']
                            total_est += cost
                            if rem_cash >= cost:
                                rem_cash -= cost
                                icon = "⭐" if o['type'] == 'MAIN' else "💧"
                                st.write(f"{icon} **${o['price']}** × {o['qty']}개")
                            else:
                                st.caption(f"현금부족 (${o['price']})")
                        st.caption(f"(예상 투입: ${total_est:,.0f})")
                    else:
                        st.warning("슬롯 꽉 참")
                    
                    # [B] 보유 종목 출력
                    if current_holdings:
                        with st.expander(f"보유 종목 ({len(current_holdings)}개) & 추가매수"):
                            for h in current_holdings:
                                buy_p, days, qty, mode, tier, _ = h
                                st.markdown(f"**T{tier}** (${buy_p}) - {days}일차")
                                
                                # 물타기 계산
                                real_bet_add = min(one_time_seed, current_cash)
                                net_bet_add = real_bet_add / (1 + p_params['fee_rate'])
                                orders = get_smart_orders(net_bet_add, loc_price, -1*(loc_range/100.0), n_split)
                                rem_cash = current_cash
                                has_order = False
                                for o in orders:
                                    cost = o['price']*o['qty']
                                    icon = "💧" if o['price'] < buy_p else "🔥"
                                    if rem_cash >= cost:
                                        st.write(f"{icon} ${o['price']} × {o['qty']}개")
                                        has_order = True
                                if not has_order: st.caption("주문 불가")
                                st.divider()

            # 왼쪽: 안정형, 오른쪽: 공격형 렌더링
            render_dashboard(col_stable, params_s, "🛡️ 안정형 전략")
            render_dashboard(col_agg, params_a, "🔥 공격형 전략")


        # ==========================================
        # 탭 2: 성과 비교 (백테스트)
        # ==========================================
        with tab_bt:
            st.info("💡 각 전략의 설정된 기간과 자본금으로 시뮬레이션을 실행합니다.")
            if st.button("🚀 두 전략 비교 실행", type='primary'):
                with st.spinner("듀얼 백테스트 진행 중..."):
                    res_s = backtest_engine_web(df, params_s)
                    res_a = backtest_engine_web(df, params_a)
                
                if res_s and res_a:
                    # 1. 지표 비교 테이블
                    comp_data = {
                        '구분': ['기간', '초기 자본', '최종 자산', '수익률', 'CAGR', 'MDD', '승률'],
                        '🛡️ 안정형': [
                            f"{params_s['start_date']}~", f"${params_s['initial_balance']:,}",
                            f"${res_s['Final']:,.0f}", f"{res_s['Return']:.2f}%", 
                            f"{res_s['CAGR']:.2f}%", f"{res_s['MDD']:.2f}%", f"{res_s['WinRate']}%"
                        ],
                        '🔥 공격형': [
                            f"{params_a['start_date']}~", f"${params_a['initial_balance']:,}",
                            f"${res_a['Final']:,.0f}", f"{res_a['Return']:.2f}%", 
                            f"{res_a['CAGR']:.2f}%", f"{res_a['MDD']:.2f}%", f"{res_a['WinRate']}%"
                        ]
                    }
                    st.table(pd.DataFrame(comp_data).set_index('구분'))
                    
                    # 2. 그래프 겹쳐 그리기 (기간이 달라도 날짜축 기준으로 자동 매핑됨)
                    st.subheader("📈 자산 성장 곡선 비교")
                    chart_df = pd.DataFrame({
                        'Stable': res_s['Series'],
                        'Aggressive': res_a['Series']
                    })
                    st.line_chart(chart_df)
                    
                    # 3. 상세 로그 다운로드
                    c1, c2 = st.columns(2)
                    c1.download_button("📥 안정형 로그", res_s['TradeLog'].to_csv().encode('utf-8-sig'), "stable_log.csv")
                    c2.download_button("📥 공격형 로그", res_a['TradeLog'].to_csv().encode('utf-8-sig'), "agg_log.csv")

else:
    st.warning("👈 구글 시트 URL을 입력해주세요.")
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
                'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
				'tier_weights': edited_weights
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
                # [추가] 시드 계산용 '확정 자산' 가져오기
                seed_equity_basis = res['DailyLog'].iloc[-1]['SeedEquity']

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
                    st.subheader("🛒 오늘의 매수 주문 (Smart LOC)")
                    
                    # 0. 파라미터 및 시드 설정
                    n_split = int(dash_params['add_order_cnt'])
                    loc_range = dash_params['loc_range']
                    if n_split < 1: n_split = 1
                    
                    # [수정] 1회 시드 계산 (비중표 적용)
                    next_tier = len(current_holdings) + 1
                    if next_tier > 10: next_tier = 10
                    
                    # 현재 모드(바닥/중간/천장)에 맞는 컬럼 이름 찾기
                    if "바닥" in curr_phase: col_name = "Bottom"
                    elif "천장" in curr_phase: col_name = "Ceiling"
                    else: col_name = "Middle"
                    
                    # 비중 가져오기
                    try:
                        target_weight = dash_params['tier_weights'].loc[f'Tier {next_tier}', col_name]
                    except:
                        target_weight = 10.0
                        
                    one_time_seed = seed_equity_basis * (target_weight / 100.0)
                    
                    # 1. 구간별 시작 비율
                    if "바닥" in curr_phase: start_rate = dash_params['bt_buy']
                    elif "천장" in curr_phase: start_rate = dash_params['cl_buy']
                    else: start_rate = dash_params['md_buy']
                    
                    base_price = last_row['SOXL']
                    
                    # 메인 LOC 가격 (Start Price)
                    loc_price = excel_round_down(base_price * (1 + start_rate/100.0), 2)
                    
                    st.markdown(f"**📉 기준 종가**: ${base_price} | **구간**: {curr_phase}")
                    st.caption(f"⚙️ 설정: {n_split}단 분할 / LOC 범위 {loc_range}% / 1회시드 ${one_time_seed:,.0f}")
                    st.markdown("---")

                    # ==========================================================
                    # [핵심] 백테스트 엔진과 동일한 'Smart LOC' 계산 로직 함수
                    # ==========================================================
                    def get_smart_orders(seed, start_p, range_pct, split_cnt):
                        orders = []
                        if start_p <= 0: return orders
                        
                        # 1) Base Qty (메인 주문) 계산
                        # 시드를 쪼개지 않고 통째로 계산합니다.
                        base_qty = int(seed / start_p)
                        orders.append({'price': start_p, 'qty': base_qty, 'type': 'MAIN'})
                        
                        if split_cnt <= 0: return orders

                        # 2) Step Qty (추가 주문) 계산
                        # 하단 가격
                        multiplier = (1 + range_pct) if range_pct <= 0 else (1 - range_pct)
                        bot_p = excel_round_down(start_p * multiplier, 2)
                        
                        if bot_p <= 0: return orders
                        
                        # 하단과 상단의 수량 차이를 분할 횟수로 나눔
                        qty_at_bot = seed / bot_p
                        qty_at_top = seed / start_p
                        fix_qty = int((qty_at_bot - qty_at_top) / split_cnt)
                        
                        if fix_qty < 0: fix_qty = 0
                        
                        # 3) 추가 주문 리스트 생성
                        # 가격 결정 논리: seed / (base + i*fix) = Price_i
                        for i in range(1, split_cnt + 1):
                            target_cum_qty = base_qty + (i * fix_qty)
                            if target_cum_qty > 0:
                                next_p = excel_round_down(seed / target_cum_qty, 2)
                                # 가격이 0이거나 메인보다 높으면 스킵
                                if next_p > 0 and next_p < start_p:
                                    orders.append({'price': next_p, 'qty': fix_qty, 'type': 'ADD'})
                        
                        return orders

                    # ==========================================================
                    
                    # --- [A] 신규 진입 (Tier N+1) ---
                    if len(current_holdings) < 10:
                        st.success(f"🆕 **신규 진입 (Tier {len(current_holdings)+1})**")
                        
                        # Smart LOC 주문 생성
                        # 백테스트 로직: buy_range는 음수(예: -0.05)로 입력되어야 함
                        range_val = -1 * (loc_range / 100.0)
                        orders = get_smart_orders(one_time_seed, loc_price, range_val, n_split)
                        
                        remaining_cash = current_cash
                        
                        for i, order in enumerate(orders):
                            p = order['price']
                            q = order['qty']
                            amt = p * q
                            
                            # 현금 체크
                            if remaining_cash >= amt:
                                remaining_cash -= amt
                                if order['type'] == 'MAIN':
                                    st.markdown(f"⭐ **Main**: **${p}** × **{q}주**")
                                else:
                                    st.markdown(f"💧 **Add #{i}**: **${p}** × **{q}주**")
                            else:
                                st.caption(f"#{i} 현금 부족 (${amt:,.0f} 필요)")
                                
                        st.caption(f"(총 예상 투입: ${sum([o['price']*o['qty'] for o in orders]):,.0f})")
                            
                    else:
                        st.info("🚫 슬롯 꽉 참 (신규 진입 없음)")

                    st.markdown("---")

                    # --- [B] 보유 종목 추가 매수 (Smart LOC) ---
                    if current_holdings:
                        st.write(f"🔄 **보유 종목 추가 매수 ({len(current_holdings)}건)**")
                        
                        for h in current_holdings:
                            buy_p, days, qty, mode, tier, buy_dt = h
                            
                            with st.container():
                                st.markdown(f"**Tier {tier}** (평단 ${buy_p})")
                                
                                # 보유 종목도 동일하게 Smart LOC 적용 (시드 재계산)
                                range_val = -1 * (loc_range / 100.0)
                                # 전략에 따라: '평단' 기준이 아닌 '오늘의 LOC 기준가'로 주문 생성
                                # (물타기 시점 판단 로직이 있다면 여기서 필터링 가능)
                                
                                orders = get_smart_orders(one_time_seed, loc_price, range_val, n_split)
                                
                                remaining_cash = current_cash 
                                has_order = False
                                
                                for i, order in enumerate(orders):
                                    p = order['price']
                                    q = order['qty']
                                    
                                    # 물타기/불타기 아이콘
                                    icon = "💧" if p < buy_p else "🔥"
                                    label = "Main" if order['type']=='MAIN' else f"Add #{i}"
                                    
                                    if remaining_cash >= p*q:
                                        # (주의: 실제로는 신규매수와 현금을 공유하므로 로직상 우선순위 필요)
                                        st.write(f"{icon} **{label}**: **${p}** × **{q}주**")
                                        has_order = True
                                    
                                if not has_order:
                                    st.caption("주문 가능 현금 없음")
                                
                                st.divider()
                    else:
                        st.write("보유 종목 없음")

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


