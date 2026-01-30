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

# --- [구글 시트 데이터 로드 함수] ---
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

        # 헤더 위치 찾기
        header_row_idx = -1
        idx_qqq = -1
        idx_soxl = -1
        
        for i, row in enumerate(rows[:20]):
            if "QQQ" in row and "SOXL" in row:
                header_row_idx = i
                idx_qqq = row.index("QQQ")
                idx_soxl = row.index("SOXL")
                break
        
        if header_row_idx == -1:
            st.error("❌ 시트에서 'QQQ'와 'SOXL' 헤더를 찾을 수 없습니다.")
            return None

        # 데이터 추출 함수
        def extract_series(data_rows, col_idx, name):
            start_row = header_row_idx + 2 
            extracted = []
            for r in data_rows[start_row:]:
                if len(r) > col_idx + 1:
                    d = r[col_idx]
                    p = r[col_idx + 1]
                    if d and p:
                        extracted.append([d, p])
            
            df_temp = pd.DataFrame(extracted, columns=['Date', name])
            df_temp['Date'] = df_temp['Date'].astype(str).str.strip()
            df_temp['Date'] = df_temp['Date'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
            df_temp['Date'] = df_temp['Date'].str.replace('.', '-')
            
            def fix_year(date_str):
                try:
                    parts = date_str.split('-')
                    if len(parts) == 3 and len(parts[0]) == 2:
                        return f"20{parts[0]}-{parts[1]}-{parts[2]}"
                    return date_str
                except: return date_str
            
            df_temp['Date'] = df_temp['Date'].apply(fix_year)
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
            
            df_temp[name] = df_temp[name].astype(str).str.replace(',', '').str.replace('$', '')
            df_temp[name] = pd.to_numeric(df_temp[name], errors='coerce')
            
            df_temp.dropna(inplace=True)
            return df_temp

        df_qqq = extract_series(rows, idx_qqq, 'QQQ')
        df_soxl = extract_series(rows, idx_soxl, 'SOXL')

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

# --- [구글 시트로 주문 데이터 전송 함수] ---
def send_orders_to_gsheet(orders_df, sheet_url, worksheet_name="HTS주문"):
    """
    매수/매도 주문 데이터를 구글시트로 전송
    HTS 자동화에서 이 시트를 읽어 주문 실행
    """
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_url(sheet_url)
        
        # 워크시트 찾기 또는 생성
        try:
            worksheet = sheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=worksheet_name, rows=100, cols=10)
        
        # 기존 데이터 클리어
        worksheet.clear()
        
        # 헤더 및 데이터 업데이트
        if not orders_df.empty:
            worksheet.update([orders_df.columns.tolist()] + orders_df.values.tolist())
        
        return True
    except Exception as e:
        st.error(f"구글 시트 전송 실패: {e}")
        return False


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
    
    # 데이터 전처리
    df['QQQ'] = pd.to_numeric(df['QQQ'], errors='coerce')
    ma_win = int(params['ma_window'])
    
    # 이평선 계산
    df['MA_Daily'] = df['QQQ'].rolling(window=ma_win, min_periods=1).mean()
    df['Log_Start_Price'] = df['QQQ'].shift(ma_win - 1)

    # 주간 데이터 처리 (휴장일 대응)
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
    
    # 날짜 필터링
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

    for i in range(len(df)):
        row = df.iloc[i]
        date = row.name
        start_cash = cash # 아침 예수금 기록
        
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
                bet = min(target_seed, start_cash) # 아침 예수금 기준
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
                    
                    max_buyable = int(start_cash / (today_close * (1 + params['fee_rate']))) 
                    real_qty = min(final_qty, max_buyable)
                    
                    if real_qty > 0:
                        buy_amt = today_close * real_qty * (1 + params['fee_rate'])
                        cash -= buy_amt
                        holdings.append([today_close, 0, real_qty, phase, new_tier, dates[i]])
                        trade_log.append({
                            'Date': dates[i], 'Type': 'Buy', 'Tier': new_tier, 'Phase': phase, 
                            'Ref_Date': row['Log_Ref_Date'].strftime('%Y-%m-%d') if pd.notnull(row['Log_Ref_Date']) else '-',
                            'QQQ_Fri': row['Log_QQQ_Fri'], 'MA_Calc': row['Log_MA_Fri'], 'Disp': disp,
                            'Start_P': row['Log_Start_Price'], 'Price': today_close, 'Qty': real_qty, 
                            'Profit': 0, 'Reason': 'LOC'
                        })
        
        # 3. 투자금 갱신
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
st.title("📊 쪼꼬야옹의 듀얼 전략 연구소")

with st.sidebar:
    st.header("⚙️ 기본 데이터 연동")
    sheet_url = st.text_input("🔗 구글 시트 주소", value=DEFAULT_SHEET_URL)
    st.caption("※ 시트에 'Date', 'SOXL', 'QQQ' 데이터가 있어야 합니다.")
    
    st.markdown("---")
    st.header("⚔️ 전략별 상세 설정")
    
    tab_s, tab_a = st.tabs(["🛡️ 안정형", "🔥 공격형"])

    # === [함수] 파라미터 입력 위젯 생성기 ===
    def render_strategy_inputs(suffix, key_prefix):
        st.subheader(f"📊 {key_prefix} 기본 설정")
        
        # [독립 설정] 초기 자본
        balance = st.number_input(f"초기 자본 ($)", value=10000, key=f"bal_{suffix}")
        
        # [독립 설정] 기간
        today = datetime.date.today()
        c_d1, c_d2 = st.columns(2)
        start_date = c_d1.date_input("시작일", value=datetime.date(2010, 1, 1), max_value=today, key=f"sd_{suffix}")
        end_date = c_d2.date_input("종료일", value=today, max_value=today, key=f"ed_{suffix}")
        
        st.markdown("---")
        st.write("⚙️ **파라미터 설정**")
        
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
            'start_date': start_date, 'end_date': end_date,
            'initial_balance': balance,
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
            last_date_str = df.index[-1].strftime('%Y-%m-%d')
            st.header(f"📢 오늘의 투자 브리핑 ({last_date_str})")
            
            col_stable, col_agg = st.columns(2)
            
            # --- 대시보드 출력용 함수 (주문 데이터 반환) ---
            def render_dashboard(col, p_params, strategy_name, stock_name="SOXL"):
                hts_orders = []  # HTS 전송용 주문 리스트
                
                with col:
                    st.subheader(f"{strategy_name}")
                    
                    # [중요] 각 전략의 start_date/balance로 백테스트 실행
                    res = backtest_engine_web(df, p_params)
                    if not res:
                        st.error("데이터 부족 (기간 확인)")
                        return hts_orders

                    last_row = res['LastData']
                    daily_last = res['DailyLog'].iloc[-1]
                    current_cash = daily_last['Cash']
                    seed_equity_basis = daily_last['SeedEquity']
                    current_holdings = res['CurrentHoldings']
                    
                    disp = last_row['Basis_Disp']
                    if disp < p_params['bt_cond']: curr_phase = "📉 바닥"
                    elif disp > p_params['cl_cond']: curr_phase = "📈 천장"
                    else: curr_phase = "➖ 중간"
                    
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
                    
                    one_time_seed = seed_equity_basis * (target_weight / 100.0)
                    
                    base_price = last_row['SOXL']
                    loc_price = excel_round_down(base_price * (1 + start_rate/100.0), 2)

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

                    # [A] 매수 주문 (Buy Orders)
                    st.markdown("#### 🛒 매수 주문")
                    
                    buy_list = []
                    
                    # 1. 신규 진입 계산
                    if len(current_holdings) < 10:
                        real_bet = min(one_time_seed, current_cash)
                        net_bet = real_bet / (1 + p_params['fee_rate'])
                        orders = get_smart_orders(net_bet, loc_price, -1*(loc_range/100.0), n_split)
                        rem_cash = current_cash
                        
                        for i, o in enumerate(orders):
                            cost = o['price'] * o['qty']
                            status = "주문가능"
                            if rem_cash >= cost:
                                rem_cash -= cost
                            else:
                                status = "현금부족"
                            
                            label = "⭐ MAIN" if o['type'] == 'MAIN' else f"💧 ADD #{i}"
                            buy_list.append({
                                "구분": label,
                                "가격 ($)": f"{o['price']}",
                                "수량": f"{o['qty']}",
                                "예상금액 ($)": f"{cost:,.0f}",
                                "상태": status
                            })
                    
                    if buy_list:
                        st.info(f"🆕 **신규 진입 (Tier {next_tier})**")
                        st.dataframe(pd.DataFrame(buy_list), hide_index=True, use_container_width=True)
                    elif len(current_holdings) >= 10:
                        st.warning("🚫 슬롯이 꽉 찼습니다 (추가 매수 불가)")
                    else:
                        st.caption("매수 조건 미달")

                    st.divider()

                    # [B] 매도 주문 (Sell Orders)
                    st.markdown("#### 💰 매도 주문")
                    
                    if not current_holdings:
                        st.caption("보유 중인 종목이 없습니다.")
                    else:
                        sell_list = []
                        for h in current_holdings:
                            # h = [buy_p, days, qty, mode, tier, buy_dt]
                            buy_p, days, qty, mode, tier, buy_dt = h
                            
                            if mode == 'Bottom': 
                                prof_rate = p_params['bt_prof']
                                time_limit = p_params['bt_time']
                            elif mode == 'Ceiling': 
                                prof_rate = p_params['cl_prof']
                                time_limit = p_params['cl_time']
                            else: 
                                prof_rate = p_params['md_prof']
                                time_limit = p_params['md_time']
                            
                            target_sell_p = excel_round_up(buy_p * (1 + prof_rate), 2)
                            curr_return = (last_row['SOXL'] - buy_p) / buy_p * 100
                            current_hold_days = days + 1
                            
                            # 타임컷 로직 적용
                            if current_hold_days >= time_limit:
                                order_type = "🚨 MOC (시장가)"
                                order_price = "Market"
                                note = "TimeCut 발동"
                            else:
                                order_type = "🎯 LOC (지정가)"
                                order_price = f"${target_sell_p}"
                                note = f"{current_hold_days}/{time_limit}일"

                            sell_list.append({
                                "티어": f"T{tier}",
                                "평단가": f"${buy_p}",
                                "수익률": f"{curr_return:.2f}%",
                                "주문타입": order_type,
                                "주문가격": order_price,
                                "비고": note
                            })
                            
                            # HTS 전송용 데이터 수집 (전략 구분 추가)
                            hts_orders.append({
                                "전략": strategy_name,
                                "종목": stock_name,
                                "주문유형": "매도",
                                "주문타입": "MOC" if "MOC" in order_type else "LOC",
                                "가격": target_sell_p if "LOC" in order_type else 0,
                                "수량": qty
                            })
                        
                        # 스타일링 함수 (타임컷 빨간색 강조)
                        def highlight_moc(row):
                            if "MOC" in row['주문타입']:
                                return ['background-color: #ffcccc; color: black'] * len(row)
                            return [''] * len(row)

                        st.dataframe(pd.DataFrame(sell_list).style.apply(highlight_moc, axis=1), hide_index=True, use_container_width=True)
                    
                    # 매수 주문도 HTS 데이터에 추가 (전략 구분 추가)
                    if buy_list:
                        for b in buy_list:
                            if b["상태"] == "주문가능":
                                hts_orders.append({
                                    "전략": strategy_name,
                                    "종목": stock_name,
                                    "주문유형": "매수",
                                    "주문타입": "LOC",
                                    "가격": float(b["가격 ($)"]),
                                    "수량": int(b["수량"])
                                })
                
                return hts_orders

            orders_stable = render_dashboard(col_stable, params_s, "🛡️ 안정형 전략")
            orders_agg = render_dashboard(col_agg, params_a, "🔥 공격형 전략")
            
            # HTS 전송 섹션
            st.divider()
            st.subheader("📤 HTS 자동화 연동")
            
            # 안정형/공격형 분리 표시
            col_hts1, col_hts2 = st.columns(2)
            
            with col_hts1:
                st.markdown("#### 🛡️ 안정형 (탭1)")
                if orders_stable:
                    df_stable = pd.DataFrame(orders_stable)
                    st.dataframe(df_stable, hide_index=True, use_container_width=True)
                else:
                    st.caption("주문 없음")
            
            with col_hts2:
                st.markdown("#### 🔥 공격형 (탭2)")
                if orders_agg:
                    df_agg = pd.DataFrame(orders_agg)
                    st.dataframe(df_agg, hide_index=True, use_container_width=True)
                else:
                    st.caption("주문 없음")
            
            st.divider()
            
            # 전송 옵션
            all_orders = orders_stable + orders_agg
            if all_orders:
                orders_df = pd.DataFrame(all_orders)
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("🚀 지금 전송", type="primary", use_container_width=True):
                        if send_orders_to_gsheet(orders_df, sheet_url, "HTS주문"):
                            st.success("✅ 전송 완료!")
                        else:
                            st.error("❌ 전송 실패 (권한 확인 필요)")
                
                with col_btn2:
                    # 자동 전송 시간 설정
                    auto_time = st.time_input("⏰ 자동 전송 시간", value=datetime.time(22, 30))
                    
                    # 현재 시간과 비교하여 자동 전송
                    now = datetime.datetime.now().time()
                    if 'last_auto_send' not in st.session_state:
                        st.session_state.last_auto_send = None
                    
                    today_str = datetime.date.today().isoformat()
                    
                    # 오늘 이미 전송했는지 확인
                    if st.session_state.last_auto_send == today_str:
                        st.info(f"✅ 오늘 {auto_time} 자동 전송 완료")
                    elif now >= auto_time:
                        # 자동 전송 실행
                        if send_orders_to_gsheet(orders_df, sheet_url, "HTS주문"):
                            st.session_state.last_auto_send = today_str
                            st.success(f"⏰ {auto_time} 자동 전송 완료!")
                    else:
                        st.caption(f"⏳ {auto_time}에 자동 전송 예정")
            else:
                st.caption("전송할 주문 데이터가 없습니다.")


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
    st.warning("👈 왼쪽 사이드바에 구글 시트 주소를 입력하거나, CSV 파일을 업로드해주세요.")
