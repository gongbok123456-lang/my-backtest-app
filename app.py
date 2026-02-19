import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import time
import json
import random
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- [기본 설정 값] ---
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dK11y5aTIhDGfpMduNsuSgTDlDoPo-OF6uE5FIePXVg/edit"
DEFAULT_ORDER_URL = "https://docs.google.com/spreadsheets/d/1PpgexM79XVvr23sVfi_6ZsrfASetVXhqjJQDYuISOnM/edit?gid=117251557#gid=117251557" 

# --- [페이지 설정] ---
# // UI 개선: layout="wide", initial_sidebar_state="expanded"
st.set_page_config(page_title="쪼꼬야옹 백테스트 연구소", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- [UI 개선: 다크 모드 테마 + 액센트 컬러 CSS] ---
st.markdown("""
<style>
/* ===== 글로벌 테마 ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --accent-coral: #FF6B6B;
    --accent-teal: #4ECDC4;
    --bg-card: rgba(30, 30, 46, 0.65);
    --bg-card-hover: rgba(40, 40, 60, 0.8);
    --border-subtle: rgba(255,255,255,0.08);
    --text-muted: #a0a0b8;
}

/* 메인 영역 기본 폰트 */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ===== KPI 메트릭 카드 ===== */
/* // UI 개선: 메트릭을 카드 스타일로 */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    transition: transform 0.15s, box-shadow 0.15s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(78,205,196,0.15);
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.35rem !important;
}

/* ===== 사이드바 스타일 ===== */
/* // UI 개선: 사이드바 시각적 계층 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: var(--accent-teal) !important;
    font-size: 1rem !important;
    border-bottom: 2px solid var(--accent-teal);
    padding-bottom: 6px;
    margin-bottom: 10px;
}

/* ===== 탭 스타일 ===== */
/* // UI 개선: 탭 버튼 스타일 */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-teal), #45b7aa) !important;
    color: #fff !important;
}

/* ===== 매수/매도 주문 컨테이너 ===== */
/* // UI 개선: 매수 주문 – 틸 강조 */
.buy-orders-box {
    border-left: 4px solid var(--accent-teal);
    background: rgba(78,205,196,0.06);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
/* // UI 개선: 매도 주문 – 코랄 강조 */
.sell-orders-box {
    border-left: 4px solid var(--accent-coral);
    background: rgba(255,107,107,0.06);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

/* ===== 프로그레스 바 (시장 상태) ===== */
/* // UI 개선: 프로그레스 바 색상 */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-coral));
}

/* ===== 버튼 ===== */
/* // UI 개선: 기본 버튼 그래디언트 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-coral), #ee5a6f) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(255,107,107,0.35) !important;
}

/* ===== 구분선 ===== */
hr { border-color: var(--border-subtle) !important; }

/* ===== 반응형 ===== */
@media (max-width: 768px) {
    .stColumns { display: flex !important; flex-direction: column !important; }
    .stColumns > div { width: 100% !important; min-width: unset !important; margin-bottom: 0.8rem; }
    .stButton > button { padding: 0.75rem 1rem !important; font-size: 1rem !important; min-height: 44px; }
    .stTextInput input, .stNumberInput input, .stDateInput input,
    .stSelectbox select, .stTextArea textarea {
        font-size: 1rem !important; padding: 0.75rem !important; min-height: 44px !important;
    }
    .stDataFrame, .stTable { overflow-x: auto !important; -webkit-overflow-scrolling: touch !important; }
    div[data-testid="stMetric"] { text-align: center !important; }
    .stTabs [role="tab"] { padding: 0.6rem 0.8rem !important; font-size: 0.85rem !important; }
    h1 { font-size: 1.5rem !important; } h2 { font-size: 1.25rem !important; } h3 { font-size: 1.05rem !important; }
}
@media (min-width: 769px) and (max-width: 1024px) {
    .stColumns > div { min-width: 45% !important; }
}
button, [role="button"], input, select, textarea {
    touch-action: manipulation; -webkit-tap-highlight-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# --- [세션 상태 초기화] ---
if 'opt_results' not in st.session_state: st.session_state.opt_results = pd.DataFrame()
if isinstance(st.session_state.opt_results, list): st.session_state.opt_results = pd.DataFrame(st.session_state.opt_results)
if not st.session_state.opt_results.empty and 'B_Time' not in st.session_state.opt_results.columns:
    st.session_state.opt_results = pd.DataFrame()

if 'trial_count' not in st.session_state: st.session_state.trial_count = 0
if 'last_backtest_result' not in st.session_state: st.session_state.last_backtest_result = None
if 'editor_ver' not in st.session_state: st.session_state.editor_ver = 0

# --- [구글 시트 연동] ---
def get_gspread_client():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"구글 인증 실패: {e}")
        return None

@st.cache_data(ttl=600)
def load_data_from_gsheet(url):
    client = get_gspread_client()
    if not client: return None
    try:
        sheet = client.open_by_url(url)
        worksheet = sheet.get_worksheet(0)
        rows = worksheet.get_all_values()
        if not rows: return None

        header_row_idx = -1
        idx_qqq = -1; idx_soxl = -1
        for i, row in enumerate(rows[:20]):
            if "QQQ" in row and "SOXL" in row:
                header_row_idx = i
                idx_qqq = row.index("QQQ"); idx_soxl = row.index("SOXL")
                break
        
        if header_row_idx == -1: return None

        def extract_series(data_rows, col_idx, name):
            start_row = header_row_idx + 2 
            extracted = []
            for r in data_rows[start_row:]:
                if len(r) > col_idx + 1:
                    d = r[col_idx]; p = r[col_idx + 1]
                    if d and p: extracted.append([d, p])
            df_temp = pd.DataFrame(extracted, columns=['Date', name])
            df_temp['Date'] = df_temp['Date'].astype(str).str.strip().str.replace(r'\(.*?\)', '', regex=True).str.replace('.', '-')
            def fix_year(date_str):
                try:
                    parts = date_str.split('-')
                    if len(parts) == 3 and len(parts[0]) == 2: return f"20{parts[0]}-{parts[1]}-{parts[2]}"
                    return date_str
                except: return date_str
            df_temp['Date'] = df_temp['Date'].apply(fix_year)
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
            df_temp[name] = pd.to_numeric(df_temp[name].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
            df_temp.dropna(inplace=True)
            return df_temp

        df_qqq = extract_series(rows, idx_qqq, 'QQQ')
        df_soxl = extract_series(rows, idx_soxl, 'SOXL')
        df_merged = pd.merge(df_qqq, df_soxl, on='Date', how='left')
        df_merged.set_index('Date', inplace=True)
        df_merged.sort_index(inplace=True)
        return df_merged if not df_merged.empty else None
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def send_orders_to_gsheet(orders_df, sheet_url, worksheet_name="HTS주문"):
    client = get_gspread_client()
    if not client: return False
    try:
        sheet = client.open_by_url(sheet_url)
        try: worksheet = sheet.worksheet(worksheet_name)
        except: worksheet = sheet.add_worksheet(title=worksheet_name, rows=100, cols=10)
        worksheet.clear()
        if not orders_df.empty:
            worksheet.update([orders_df.columns.tolist()] + orders_df.values.tolist())
        return True
    except Exception as e:
        st.error(f"주문 전송 실패: {e}")
        return False

# --- [설정 저장/불러오기] ---
def save_settings_to_gsheet(sheet_url):
    client = get_gspread_client()
    if not client: return
    try:
        sheet = client.open_by_url(sheet_url)
        try: ws = sheet.worksheet("Settings")
        except: ws = sheet.add_worksheet(title="Settings", rows=100, cols=2)
        
        data_to_save = []
        for key in st.session_state:
            if (key.endswith('_s') or key.endswith('_a')) and not key.startswith('w_') and not key.startswith('base_w_') and not key.startswith('current_w_'):
                val = st.session_state[key]
                if isinstance(val, (datetime.date, datetime.datetime)): val = val.strftime('%Y-%m-%d')
                data_to_save.append([key, str(val)])
        
        for suffix in ['s', 'a']:
            current_key = f"current_w_{suffix}"
            if current_key in st.session_state:
                df_val = st.session_state[current_key]
                if isinstance(df_val, pd.DataFrame):
                    val = "DF:" + df_val.to_json()
                    data_to_save.append([f"w_{suffix}", val])

        ws.clear()
        if data_to_save: ws.update(data_to_save)
        st.toast("✅ 설정이 구글 시트에 저장되었습니다!", icon="💾")
    except Exception as e: st.error(f"설정 저장 실패: {e}")

def load_settings_from_gsheet(sheet_url):
    if 'settings_loaded' in st.session_state: return
    client = get_gspread_client()
    if not client: return
    try:
        sheet = client.open_by_url(sheet_url)
        try: ws = sheet.worksheet("Settings")
        except: return
        
        rows = ws.get_all_values()
        df_loaded_flag = False
        for row in rows:
            if len(row) < 2: continue
            key, val_str = row[0], row[1]
            if (key == 'w_s' or key == 'w_a') and val_str.startswith("DF:"):
                try: 
                    suffix = key.split('_')[-1]
                    loaded_df = pd.read_json(val_str[3:])
                    st.session_state[f"base_w_{suffix}"] = loaded_df
                    df_loaded_flag = True
                except: pass
            else:
                try:
                    if key.startswith('sd_'): st.session_state[key] = datetime.datetime.strptime(val_str, '%Y-%m-%d').date()
                    elif not key.startswith('ed_'):
                        if '.' in val_str: st.session_state[key] = float(val_str)
                        else: st.session_state[key] = int(val_str)
                except: st.session_state[key] = val_str
        
        if df_loaded_flag: st.session_state.editor_ver += 1
        st.session_state['settings_loaded'] = True
    except Exception as e: print(f"설정 로드 중 오류: {e}")

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
        if implied_price >= close_price and implied_price >= bot_price: final_qty += base_qty
    for i in range(1, max_add_orders + 1):
        step_qty = fix_qty
        current_cum_qty = base_qty + (i * step_qty)
        if current_cum_qty <= 0: continue
        implied_price = seed_amount / current_cum_qty
        if implied_price >= close_price and implied_price >= bot_price: final_qty += step_qty
    return final_qty

# --- [백테스트 엔진] ---
def backtest_engine_web(df, params):
    df = df.copy()
    df['QQQ'] = pd.to_numeric(df['QQQ'], errors='coerce')
    ma_win = int(params['ma_window'])
    
    # 1. 이동평균 및 이격도 계산
    df['MA_Daily'] = df['QQQ'].rolling(window=ma_win, min_periods=1).mean()
    df['Log_Start_Price'] = df['QQQ'].shift(ma_win - 1)
    
    # 2. RSI 계산
    delta = df['SOXL'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. [NEW] RSI 다이버전스 감지 (상승 다이버전스: 주가 하락 + RSI 상승)
    # 단순화: 최근 10일 최저가가 갱신되었으나, RSI는 이전 저점보다 높을 때
    df['Low_10'] = df['SOXL'].rolling(window=10).min()
    df['RSI_Low_10'] = df['RSI'].rolling(window=10).min()
    # 어제보다 오늘이 더 낮은 신저가인데, RSI는 어제보다 높거나 30 이상 유지될 때 (간단 버전)
    df['Bullish_Div'] = (df['SOXL'] == df['Low_10']) & (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'] < 45)

    # 4. [NEW] 볼린저 밴드 (20일, 2표준편차) - 익절 지연용
    df['BB_MA20'] = df['SOXL'].rolling(window=20).mean()
    df['BB_STD20'] = df['SOXL'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA20'] + (2 * df['BB_STD20'])

    # 5. 주간 데이터 매핑 (기존 로직)
    weekly_resampled = df[['QQQ', 'MA_Daily', 'Log_Start_Price']].resample('W-FRI').last()
    weekly_resampled.columns = ['QQQ_Fri', 'MA_Fri', 'Start_Price_Fri']
    weekly_resampled['Disp_Fri'] = weekly_resampled['QQQ_Fri'] / weekly_resampled['MA_Fri']
    
    daily_expanded = weekly_resampled.resample('D').ffill()
    daily_shifted = daily_expanded.shift(1)
    df_mapped = daily_shifted.reindex(df.index)
    
    df['Basis_Disp'] = df_mapped['Disp_Fri'].fillna(1.0)
    df['Log_Ref_Date'] = daily_shifted['QQQ_Fri'].reindex(df.index).index 
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
    trade_log = []; daily_log = []; daily_equity = []; daily_dates = []
    trade_count = 0; win_count = 0
    MAX_SLOTS = 10; SEC_FEE = 0.0000278

    for i in range(len(df)):
        row = df.iloc[i]
        today_close = row['SOXL']
        if pd.isna(today_close) or today_close <= 0: continue
        if params.get('force_round', True): today_close = round(today_close, 2)
        
        start_cash = cash
        strat_type = params.get('strategy_type', 'MA 이격도')
        current_disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        current_rsi = row['RSI'] if not pd.isna(row['RSI']) else 50.0
        is_div = row['Bullish_Div']

        # [전략 분기]
        if strat_type == 'RSI':
            if current_rsi < params['bt_cond']: phase = 'Bottom'
            elif current_rsi > params['cl_cond']: phase = 'Ceiling'
            else: phase = 'Middle'
            disp_val = current_rsi
        
        elif strat_type == 'RSI 다이버전스':
            # 다이버전스 발생 시 무조건 Bottom(바닥) 모드로 진입하여 과감하게 매수
            if is_div: 
                phase = 'Bottom'
            # 다이버전스가 아니면 RSI 기준을 따름 (평소엔 Middle/Ceiling)
            elif current_rsi > params['cl_cond']: 
                phase = 'Ceiling'
            else: 
                phase = 'Middle'
            disp_val = current_rsi
            
        else: # 기본값: MA 이격도
            if current_disp < params['bt_cond']: phase = 'Bottom'
            elif current_disp > params['cl_cond']: phase = 'Ceiling'
            else: phase = 'Middle'
            disp_val = current_disp

        conf = strategy[phase]
        tiers_sold = set()
        daily_net_profit_sum = 0
        
        for stock in holdings[:]:
            buy_p, days, qty, mode, tier, buy_dt, peak_price = stock
            s_conf = strategy[mode]
            days += 1
            # // NEW: MDD 패치 - peak_price 갱신
            peak_price = max(peak_price, today_close)
            stock[6] = peak_price
            target_p = excel_round_up(buy_p * (1 + s_conf['prof']), 2)
            is_sold = False; reason = ""
            # // NEW: MDD 패치 - 트레일링 스탑 (trailing_pct > 0 일 때만)
            t_pct = params.get('trailing_pct', 0)
            if t_pct > 0 and today_close < peak_price * (1 - t_pct):
                is_sold = True; reason = f"TrailingStop({t_pct*100:.0f}%)"
            elif days >= s_conf['time']: is_sold = True; reason = f"TimeCut({days}d)"
            elif today_close >= target_p: 
                # [NEW] 볼린저 밴드 워크 (익절 지연) 로직
                if params.get('use_bb_walk', False) and today_close > row['BB_Upper']:
                    is_sold = False # 아직 팔지 마! (밴드 상단 돌파 중)
                else:
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
                    'Ref_Date': '-', 'Disp': disp_val, 'Price': today_close, 'Qty': qty, 
                    'Profit': real_profit, 'Reason': reason
                })
            else: stock[1] = days
        
        prev_c = row['Prev_Close'] if not pd.isna(row['Prev_Close']) else today_close
        if pd.isna(prev_c): prev_c = today_close
        target_p = excel_round_down(prev_c * (1 + conf['buy'] / 100), 2)
        
        # 다이버전스 모드일 때는 목표가가 현재가보다 높아도(추격매수) 허용할 수 있음 (여기서는 기본 로직 유지)
        if today_close <= target_p and len(holdings) < MAX_SLOTS:
            curr_tiers = {h[4] for h in holdings}
            unavail = curr_tiers.union(tiers_sold)
            new_tier = 1
            while new_tier in unavail: new_tier += 1
            
            if new_tier <= MAX_SLOTS:
                weight_pct = 10.0
                if 'tier_weights' in params:
                    try: weight_pct = params['tier_weights'].loc[f'Tier {new_tier}', phase]
                    except: weight_pct = 10.0
                
                target_seed = seed_equity * (weight_pct / 100.0)
                bet = min(target_seed, start_cash)
                bet_net_fee = bet / (1 + params['fee_rate'])
                
                if bet >= 10:
                    final_qty = 0
                    if new_tier == MAX_SLOTS: final_qty = int(bet_net_fee / target_p)
                    else: final_qty = calculate_loc_quantity(bet_net_fee, target_p, today_close, -1*(params['loc_range']/100.0), int(params['add_order_cnt']))
                    max_buyable = int(start_cash / (today_close * (1 + params['fee_rate']))) 
                    real_qty = min(final_qty, max_buyable)
                    
                    if real_qty > 0:
                        buy_amt = today_close * real_qty * (1 + params['fee_rate'])
                        cash -= buy_amt
                        holdings.append([today_close, 0, real_qty, phase, new_tier, dates[i], today_close])  # // NEW: peak_price 초기값
                        trade_log.append({
                            'Date': dates[i], 'Type': 'Buy', 'Tier': new_tier, 'Phase': phase, 
                            'Ref_Date': '-', 'Disp': disp_val, 'Price': today_close, 'Qty': real_qty, 
                            'Seed(1회)': round(seed_equity, 0), 'Invest': round(buy_amt, 0),
                            'Profit': 0, 'Reason': 'LOC' if strat_type != 'RSI 다이버전스' or not is_div else 'Divergence'
                        })
        
        if daily_net_profit_sum != 0:
            rate = params['profit_rate'] if daily_net_profit_sum > 0 else params['loss_rate']
            seed_equity += daily_net_profit_sum * rate
        
        current_eq = cash + sum([h[2]*today_close for h in holdings])
        daily_equity.append(current_eq); daily_dates.append(dates[i])
        daily_log.append({'Date': dates[i], 'Equity': round(current_eq, 2), 'Cash': round(cash, 2), 'SeedEquity': round(seed_equity, 2), 'Holdings': len(holdings)})

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
        'CAGR': round(cagr, 2), 'MDD': round(mdd, 2), 'Final': int(final_equity),
        'Return': round(total_ret_pct, 2), 'WinRate': round(win_rate, 2), 'Trades': trade_count,
        'Series': eq_series, 'Yearly': yearly_ret, 'Params': params,
        'TradeLog': pd.DataFrame(trade_log), 'DailyLog': pd.DataFrame(daily_log),
	    'CurrentHoldings': holdings, 'LastData': df.iloc[-1]
    }

# --- [5모드 백테스트 엔진] ---
def backtest_engine_5mode(df, params):
    """
    5모드 체계 백테스트 엔진.
    MA 이격도: PANIC_BOTTOM / BOTTOM / NEUTRAL / BEARISH / CEILING
    RSI/다이버전스: 기존 3모드 유지 (Bottom/Middle/Ceiling)
    출력 형식은 backtest_engine_web과 동일 → UI 호환.
    """
    df = df.copy()
    df['QQQ'] = pd.to_numeric(df['QQQ'], errors='coerce')
    ma_win = int(params['ma_window'])

    # 1. 이동평균 및 이격도
    df['MA_Daily'] = df['QQQ'].rolling(window=ma_win, min_periods=1).mean()
    df['Log_Start_Price'] = df['QQQ'].shift(ma_win - 1)

    # 2. RSI
    delta = df['SOXL'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. RSI 다이버전스
    df['Low_10'] = df['SOXL'].rolling(window=10).min()
    df['RSI_Low_10'] = df['RSI'].rolling(window=10).min()
    df['Bullish_Div'] = (df['SOXL'] == df['Low_10']) & (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'] < 45)

    # 4. 볼린저 밴드
    df['BB_MA20'] = df['SOXL'].rolling(window=20).mean()
    df['BB_STD20'] = df['SOXL'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_MA20'] + (2 * df['BB_STD20'])
    # [5모드] 볼린저 밴드 위치: 표준화된 가격 위치
    df['BB_Pos'] = (df['SOXL'] - df['BB_MA20']) / df['BB_STD20'].replace(0, np.nan)

    # 5. 거래량 비율 (Volume 컬럼 있을 때만)
    has_volume = 'Volume' in df.columns
    if has_volume:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20, min_periods=5).mean()
    else:
        df['Vol_Ratio'] = np.nan

    # 6. 주간 이격도 매핑 (기존 동일)
    weekly_resampled = df[['QQQ', 'MA_Daily', 'Log_Start_Price']].resample('W-FRI').last()
    weekly_resampled.columns = ['QQQ_Fri', 'MA_Fri', 'Start_Price_Fri']
    weekly_resampled['Disp_Fri'] = weekly_resampled['QQQ_Fri'] / weekly_resampled['MA_Fri']
    daily_expanded = weekly_resampled.resample('D').ffill()
    daily_shifted = daily_expanded.shift(1)
    df_mapped = daily_shifted.reindex(df.index)
    df['Basis_Disp'] = df_mapped['Disp_Fri'].fillna(1.0)
    df['Log_Ref_Date'] = daily_shifted['QQQ_Fri'].reindex(df.index).index
    df['Prev_Close'] = df['SOXL'].shift(1)

    # 7. 기간 필터
    start_dt = pd.to_datetime(params['start_date'])
    end_dt = pd.to_datetime(params['end_date'])
    df = df.sort_index()
    df = df[(df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))].copy()
    df = df.dropna(subset=['SOXL'])
    if len(df) == 0: return None

    dates = df.index

    # ===== 모드 설정 ===== - // NEW: 5모드 모드별 UI 커스텀
    # 5모드 파라미터 (MA 이격도용)
    mode_config = {
        'PANIC_BOTTOM': {
            'buy': params.get('panic_buy', 20.0),
            'prof': params.get('panic_prof', 0.04),
            'time': params.get('panic_time', 8),
            'weight_factor': params.get('panic_wf', 1.5)
        },
        'BOTTOM': {'buy': params['bt_buy'], 'prof': params['bt_prof'], 'time': params['bt_time'], 'weight_factor': params.get('bot_wf', 1.2)},
        'NEUTRAL': {'buy': params['md_buy'], 'prof': params['md_prof'], 'time': params['md_time'], 'weight_factor': params.get('neut_wf', 1.0)},
        'BEARISH': {
            'buy': params.get('bear_buy', max(params['md_buy'], -0.5)),
            'prof': params.get('bear_prof', max(params['md_prof'], 0.02)),
            'time': params.get('bear_time', min(params['md_time'], 25)),
            'weight_factor': params.get('bear_wf', 0.7)
        },
        'CEILING': {
            'buy': params.get('ceil_buy', params['cl_buy']),
            'prof': params.get('ceil_prof', params['cl_prof']),
            'time': params.get('ceil_time', params['cl_time']),
            'weight_factor': params.get('ceil_wf', 0.5)
        },
        # 3모드 (RSI/다이버전스용)
        'Bottom':  {'buy': params['bt_buy'], 'prof': params['bt_prof'], 'time': params['bt_time'], 'weight_factor': 1.0},
        'Middle':  {'buy': params['md_buy'], 'prof': params['md_prof'], 'time': params['md_time'], 'weight_factor': 1.0},
        'Ceiling': {'buy': params['cl_buy'], 'prof': params['cl_prof'], 'time': params['cl_time'], 'weight_factor': 1.0},
    }
    # 5모드 → 티어비중 컬럼 매핑
    TIER_COL = {'PANIC_BOTTOM': 'Bottom', 'BOTTOM': 'Bottom', 'NEUTRAL': 'Middle', 'BEARISH': 'Middle', 'CEILING': 'Ceiling',
                'Bottom': 'Bottom', 'Middle': 'Middle', 'Ceiling': 'Ceiling'}

    cash = params['initial_balance']
    seed_equity = cash
    holdings = []
    trade_log = []; daily_log = []; daily_equity = []; daily_dates = []
    trade_count = 0; win_count = 0
    MAX_SLOTS = 10; SEC_FEE = 0.0000278

    for i in range(len(df)):
        row = df.iloc[i]
        today_close = row['SOXL']
        if pd.isna(today_close) or today_close <= 0: continue
        if params.get('force_round', True): today_close = round(today_close, 2)

        start_cash = cash
        strat_type = params.get('strategy_type', 'MA 이격도')
        current_disp = row['Basis_Disp'] if not pd.isna(row['Basis_Disp']) else 1.0
        current_rsi = row['RSI'] if not pd.isna(row['RSI']) else 50.0
        is_div = row['Bullish_Div']
        bb_pos = row['BB_Pos'] if not pd.isna(row['BB_Pos']) else 0.0
        vol_r = row['Vol_Ratio'] if not pd.isna(row['Vol_Ratio']) else 0.0

        # ===== 모드 분류 =====
        if strat_type == 'MA 이격도':
            # 5모드 분류 (우선순위 순서) - // NEW: 5모드 커스텀 params 적용
            vol_ok = (vol_r > 1.8) if has_volume else True
            p_disp = params.get('panic_disp_th', 0.82)
            p_rsi = params.get('panic_rsi_th', 25)
            p_bb = params.get('panic_bb_th', -1.8)
            b_rsi = params.get('bear_rsi_th', 60)
            b_disp = params.get('bear_disp_th', 1.05)
            ceil_rsi = params.get('ceil_rsi_th', 65)
            bot_rsi = params.get('bot_rsi_th', 45)
            if current_disp < p_disp and current_rsi < p_rsi and bb_pos < p_bb and vol_ok:
                phase = 'PANIC_BOTTOM'
            elif current_disp > params['cl_cond'] or current_rsi > ceil_rsi:
                phase = 'CEILING'
            elif current_rsi > b_rsi and current_disp > b_disp and bb_pos > 0:
                phase = 'BEARISH'
            elif current_disp < params['bt_cond'] or current_rsi < bot_rsi:
                phase = 'BOTTOM'
            else:
                phase = 'NEUTRAL'
            disp_val = current_disp
        elif strat_type == 'RSI':
            if current_rsi < params['bt_cond']: phase = 'Bottom'
            elif current_rsi > params['cl_cond']: phase = 'Ceiling'
            else: phase = 'Middle'
            disp_val = current_rsi
        elif strat_type == 'RSI 다이버전스':
            if is_div: phase = 'Bottom'
            elif current_rsi > params['cl_cond']: phase = 'Ceiling'
            else: phase = 'Middle'
            disp_val = current_rsi
        else:
            if current_disp < params['bt_cond']: phase = 'Bottom'
            elif current_disp > params['cl_cond']: phase = 'Ceiling'
            else: phase = 'Middle'
            disp_val = current_disp

        conf = mode_config[phase]
        tiers_sold = set()
        daily_net_profit_sum = 0

        # ===== 매도 로직 (기존 동일) =====
        for stock in holdings[:]:
            buy_p, days, qty, mode, tier, buy_dt, peak_price = stock
            s_conf = mode_config[mode]
            days += 1
            # // NEW: MDD 패치 - peak_price 갱신
            peak_price = max(peak_price, today_close)
            stock[6] = peak_price
            target_p = excel_round_up(buy_p * (1 + s_conf['prof']), 2)
            is_sold = False; reason = ""
            # // NEW: MDD 패치 - 트레일링 스탑
            t_pct = params.get('trailing_pct', 0)
            if t_pct > 0 and today_close < peak_price * (1 - t_pct):
                is_sold = True; reason = f"TrailingStop({t_pct*100:.0f}%)"
            elif days >= s_conf['time']: is_sold = True; reason = f"TimeCut({days}d)"
            elif today_close >= target_p:
                if params.get('use_bb_walk', False) and today_close > row['BB_Upper']:
                    is_sold = False
                else:
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
                    'Ref_Date': '-', 'Disp': disp_val, 'Price': today_close, 'Qty': qty,
                    'Profit': real_profit, 'Reason': reason
                })
            else: stock[1] = days

        # ===== 매수 로직 =====
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
                tier_col = TIER_COL.get(phase, 'Middle')
                if 'tier_weights' in params:
                    try: weight_pct = params['tier_weights'].loc[f'Tier {new_tier}', tier_col]
                    except: weight_pct = 10.0
                # [5모드] weight_factor 적용
                weight_pct = weight_pct * conf['weight_factor']
                target_seed = seed_equity * (weight_pct / 100.0)
                bet = min(target_seed, start_cash)
                bet_net_fee = bet / (1 + params['fee_rate'])
                if bet >= 10:
                    final_qty = 0
                    if new_tier == MAX_SLOTS: final_qty = int(bet_net_fee / target_p)
                    else: final_qty = calculate_loc_quantity(bet_net_fee, target_p, today_close, -1*(params['loc_range']/100.0), int(params['add_order_cnt']))
                    max_buyable = int(start_cash / (today_close * (1 + params['fee_rate'])))
                    real_qty = min(final_qty, max_buyable)
                    if real_qty > 0:
                        buy_amt = today_close * real_qty * (1 + params['fee_rate'])
                        cash -= buy_amt
                        holdings.append([today_close, 0, real_qty, phase, new_tier, dates[i], today_close])  # // NEW: peak_price
                        trade_log.append({
                            'Date': dates[i], 'Type': 'Buy', 'Tier': new_tier, 'Phase': phase,
                            'Ref_Date': '-', 'Disp': disp_val, 'Price': today_close, 'Qty': real_qty,
                            'Seed(1회)': round(seed_equity, 0), 'Invest': round(buy_amt, 0),
                            'Profit': 0, 'Reason': f'5M|{phase}'
                        })

        if daily_net_profit_sum != 0:
            rate = params['profit_rate'] if daily_net_profit_sum > 0 else params['loss_rate']
            seed_equity += daily_net_profit_sum * rate

        current_eq = cash + sum([h[2]*today_close for h in holdings])
        daily_equity.append(current_eq); daily_dates.append(dates[i])
        daily_log.append({'Date': dates[i], 'Equity': round(current_eq, 2), 'Cash': round(cash, 2), 'SeedEquity': round(seed_equity, 2), 'Holdings': len(holdings)})

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
        'CAGR': round(cagr, 2), 'MDD': round(mdd, 2), 'Final': int(final_equity),
        'Return': round(total_ret_pct, 2), 'WinRate': round(win_rate, 2), 'Trades': trade_count,
        'Series': eq_series, 'Yearly': yearly_ret, 'Params': params,
        'TradeLog': pd.DataFrame(trade_log), 'DailyLog': pd.DataFrame(daily_log),
        'CurrentHoldings': holdings, 'LastData': df.iloc[-1]
    }

# --- [향상된 분석 함수] ---
def analyze_backtest_results(result):
    """
    백테스트 결과를 심층 분석.
    backtest_engine_web / backtest_engine_5mode 둘 다 호환.
    반환: {'period_stats', 'tier_stats', 'yearly_stats', 'mode_stats', 'advanced_metrics'}
    """
    if not result or result['Trades'] == 0:
        return None

    tlog = result['TradeLog'].copy()
    dlog = result['DailyLog'].copy()
    eq = result['Series']
    sells = tlog[tlog['Type'] == 'Sell'].copy()
    if sells.empty: return None
    sells['Date'] = pd.to_datetime(sells['Date'])

    # ===== 1. 구간별 통계 (1년/3년/5년) =====
    end_date = sells['Date'].max()
    period_rows = []
    for label, years in [('최근 1년', 1), ('최근 3년', 3), ('최근 5년', 5), ('전체', None)]:
        if years:
            cutoff = end_date - pd.DateOffset(years=years)
            subset = sells[sells['Date'] >= cutoff]
        else:
            subset = sells
        if subset.empty:
            period_rows.append({'구간': label, '거래수': 0, '승률(%)': 0, '평균수익': 0, '손익비': 0})
            continue
        wins = subset[subset['Profit'] > 0]
        losses = subset[subset['Profit'] < 0]
        avg_win = wins['Profit'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['Profit'].mean()) if len(losses) > 0 else 1
        period_rows.append({
            '구간': label,
            '거래수': len(subset),
            '승률(%)': round(len(wins) / len(subset) * 100, 1),
            '평균수익': round(subset['Profit'].mean(), 2),
            '손익비': round(avg_win / avg_loss, 2) if avg_loss > 0 else float('inf')
        })
    period_stats = pd.DataFrame(period_rows)

    # ===== 2. 티어별 성과 =====
    tier_rows = []
    for t in range(1, 11):
        ts = sells[sells['Tier'] == t]
        if ts.empty:
            tier_rows.append({'티어': f'T{t}', '거래수': 0, '승률(%)': 0, '누적수익': 0, '평균수익': 0})
            continue
        tw = ts[ts['Profit'] > 0]
        tier_rows.append({
            '티어': f'T{t}',
            '거래수': len(ts),
            '승률(%)': round(len(tw) / len(ts) * 100, 1),
            '누적수익': round(ts['Profit'].sum(), 2),
            '평균수익': round(ts['Profit'].mean(), 2)
        })
    tier_stats = pd.DataFrame(tier_rows)

    # ===== 3. 연도별 상세 =====
    yearly_rows = []
    sells['Year'] = sells['Date'].dt.year
    for year in sorted(sells['Year'].unique()):
        ys = sells[sells['Year'] == year]
        yw = ys[ys['Profit'] > 0]
        # 월별 수익률 계산
        eq_year = eq[eq.index.year == year]
        if len(eq_year) > 1:
            monthly = eq_year.resample('ME').last().pct_change().dropna() * 100
            max_month = monthly.max() if len(monthly) > 0 else 0
            min_month = monthly.min() if len(monthly) > 0 else 0
        else:
            max_month = min_month = 0
        yearly_rows.append({
            '연도': year,
            '거래수': len(ys),
            '승률(%)': round(len(yw) / len(ys) * 100, 1) if len(ys) > 0 else 0,
            '누적수익': round(ys['Profit'].sum(), 2),
            '최대월(%)': round(max_month, 1),
            '최소월(%)': round(min_month, 1)
        })
    yearly_stats = pd.DataFrame(yearly_rows)

    # ===== 4. 모드별 분석 =====
    mode_rows = []
    for phase in sells['Phase'].unique():
        ms = sells[sells['Phase'] == phase]
        mw = ms[ms['Profit'] > 0]
        # 평균 보유일 추정 (Reason에서 TimeCut 파싱 또는 기본값)
        days_list = []
        for r in ms['Reason']:
            if 'TimeCut' in str(r):
                try: days_list.append(int(str(r).split('(')[1].split('d')[0]))
                except: pass
        avg_days = np.mean(days_list) if days_list else 0
        mode_rows.append({
            '모드': phase,
            '진입수': len(ms),
            '승률(%)': round(len(mw) / len(ms) * 100, 1),
            '평균보유일': round(avg_days, 1),
            '누적기여': round(ms['Profit'].sum(), 2),
            '평균수익': round(ms['Profit'].mean(), 2)
        })
    mode_stats = pd.DataFrame(mode_rows)
    if not mode_stats.empty:
        mode_stats = mode_stats.sort_values('누적기여', ascending=False)

    # ===== 5. 고급 메트릭 =====
    daily_ret = eq.pct_change().dropna()
    ann_factor = np.sqrt(252)
    sharpe = (daily_ret.mean() / daily_ret.std() * ann_factor) if daily_ret.std() > 0 else 0
    down_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / down_ret.std() * ann_factor) if len(down_ret) > 0 and down_ret.std() > 0 else 0

    # 연속 손익
    profits = sells['Profit'].values
    max_consec_win = max_consec_loss = curr_win = curr_loss = 0
    for p in profits:
        if p > 0:
            curr_win += 1; curr_loss = 0; max_consec_win = max(max_consec_win, curr_win)
        elif p < 0:
            curr_loss += 1; curr_win = 0; max_consec_loss = max(max_consec_loss, curr_loss)
        else:
            curr_win = 0; curr_loss = 0

    advanced = {
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        '최대연속수익': max_consec_win,
        '최대연속손실': max_consec_loss,
        '총거래수': len(sells),
        '평균수익': round(sells['Profit'].mean(), 2),
        '거래빈도(일/건)': round(len(eq) / max(len(sells), 1), 1),
    }

    return {
        'period_stats': period_stats,
        'tier_stats': tier_stats,
        'yearly_stats': yearly_stats,
        'mode_stats': mode_stats,
        'advanced_metrics': advanced
    }

# --- [전략 비교 함수] ---
def compare_5mode(df, params):
    """기존 3모드(trailing=0) vs 5모드+패치 비교 실행."""
    # 기존: 트레일링 없는 순수 3모드
    params_base = params.copy()
    params_base['trailing_pct'] = 0
    res_orig = backtest_engine_web(df, params_base)
    # 패치: 트레일링 + 5모드
    res_5m = backtest_engine_5mode(df, params)
    if not res_orig or not res_5m: return None
    # Sharpe 계산 헬퍼
    def _sharpe(series):
        try:
            daily_ret = series.pct_change().dropna()
            if daily_ret.std() == 0: return 0
            return round((daily_ret.mean() / daily_ret.std()) * (252**0.5), 2)
        except: return 0
    comp = pd.DataFrame({
        '지표': ['최종자산', 'CAGR (%)', 'MDD (%)', '승률 (%)', '거래수', '수익률 (%)', 'Sharpe'],
        '🔵 기존 3모드': [f"${res_orig['Final']:,}", res_orig['CAGR'], res_orig['MDD'], res_orig['WinRate'], res_orig['Trades'], res_orig['Return'], _sharpe(res_orig['Series'])],
        '🟢 패치 (5모드+트레일링)': [f"${res_5m['Final']:,}", res_5m['CAGR'], res_5m['MDD'], res_5m['WinRate'], res_5m['Trades'], res_5m['Return'], _sharpe(res_5m['Series'])],
    })
    return {'original': res_orig, 'fivemode': res_5m, 'comparison': comp}

# --- [UI 구성] ---
# // UI 개선: 타이틀 + 서브타이틀 구조
st.markdown("## 📊 쪼꼬야옹의 듀얼 전략 연구소 <sup style='color:#4ECDC4;font-size:0.5em;'>v2.1 BB</sup>", unsafe_allow_html=True)

with st.sidebar:
    # // UI 개선: 사이드바를 Expander로 섹션 분리
    st.markdown("### 🏠 Control Panel")
    
    with st.expander("📡 데이터 연동", expanded=False):
        sheet_url = st.text_input("주가 데이터 시트 (읽기)", value=DEFAULT_SHEET_URL, label_visibility="collapsed", placeholder="주가 데이터 구글시트 URL")
        st.caption("📊 주가 데이터 시트")
        st.markdown("")
        order_sheet_url = st.text_input("주문 전송 시트 (쓰기)", value=DEFAULT_ORDER_URL, placeholder="주문 전송 구글시트 URL", label_visibility="collapsed")
        st.caption("📤 HTS 주문 전송 시트")
    if order_sheet_url: load_settings_from_gsheet(order_sheet_url)
    
    st.markdown("")
    # // UI 개선: 전략 설정 탭 – 명확한 시각적 분리
    st.markdown("## ⚔️ 전략 설정")
    tab_s, tab_a = st.tabs(["🛡️ 안정형", "🔥 공격형"])

    def render_strategy_inputs(suffix, key_prefix):
        # // UI 개선: 기본 설정을 깔끔하게 정리
        st.markdown(f"**{key_prefix}**")
        k_bal = f"bal_{suffix}"
        balance = st.number_input(f"💰 초기 자본 ($)", value=st.session_state.get(k_bal, 10000), key=k_bal)
        today = datetime.date.today()
        c_d1, c_d2 = st.columns(2)
        k_sd = f"sd_{suffix}"; k_ed = f"ed_{suffix}"
        start_date = c_d1.date_input("시작일", value=st.session_state.get(k_sd, datetime.date(2010, 1, 1)), max_value=today, key=k_sd)
        end_date = c_d2.date_input("종료일", value=today, max_value=today, key=k_ed)
        
        # // UI 개선: 전략 기준을 시각적으로 분리
        st.markdown("")
        k_type = f"st_type_{suffix}"
        # [NEW] RSI 다이버전스 추가
        strategy_type = st.radio("📊 매매 기준 지표", ["MA 이격도", "RSI", "RSI 다이버전스"], index=0, horizontal=True, key=k_type)

        # [NEW] 볼린저 밴드 익절 지연 체크박스
        k_bb_walk = f"bb_walk_{suffix}"
        use_bb_walk = st.checkbox("🌭 볼린저 밴드 익절 지연 (Band Walk)", value=st.session_state.get(k_bb_walk, False), key=k_bb_walk, help="목표 수익률에 도달해도 주가가 볼린저 밴드 상단 위에 있으면 매도를 보류합니다.")

        # // UI 개선: 고급 파라미터를 Expander로 숨김
        with st.expander("⚙️ 수수료 & 복리 설정", expanded=False):
            k_fee = f"fee_{suffix}"
            fee = st.number_input("수수료 (%)", value=st.session_state.get(k_fee, 0.07), step=0.01, format="%.2f", key=k_fee)
            k_pr = f"pr_{suffix}"; k_lr = f"lr_{suffix}"
            profit_rate = st.slider("이익 복리율 (%)", 0, 100, st.session_state.get(k_pr, 70), key=k_pr)
            loss_rate = st.slider("손실 복리율 (%)", 0, 100, st.session_state.get(k_lr, 50), key=k_lr)
            
            c_loc1, c_loc2 = st.columns(2)
            k_add = f"add_{suffix}"; k_rng = f"rng_{suffix}"
            add_order_cnt = c_loc1.number_input("분할 횟수", value=st.session_state.get(k_add, 4), min_value=1, key=k_add)
            loc_range = c_loc2.number_input("LOC 범위 (-%)", value=st.session_state.get(k_rng, 20.0), min_value=0.0, key=k_rng)
            k_ma = f"ma_{suffix}"
            ma_win = st.number_input("이평선 (MA)", 50, 300, st.session_state.get(k_ma, 200), key=k_ma)

        if strategy_type.startswith('RSI'):
            lbl_bt = "RSI 기준 (이하)"; def_bt = 30.0; step_val = 1.0; lbl_cl = "RSI 기준 (이상)"; def_cl = 70.0
        else:
            lbl_bt = "이격도 기준 (이하)"; def_bt = 0.90; step_val = 0.01; lbl_cl = "이격도 기준 (이상)"; def_cl = 1.10

        # // UI 개선: 바닥/중간/천장을 Expander로 분리
        with st.expander("📉 바닥 (Bottom)", expanded=True):
            c1, c2 = st.columns(2)
            k_bc=f"bc_{suffix}"; k_bb=f"bb_{suffix}"; k_bp=f"bp_{suffix}"; k_bt=f"bt_{suffix}"
            bt_cond = c1.number_input(lbl_bt, 0.0, 100.0, st.session_state.get(k_bc, def_bt), step=step_val, key=k_bc)
            bt_buy = c2.number_input("매수점%", -30.0, 30.0, st.session_state.get(k_bb, 15.0), step=0.1, key=k_bb)
            bt_prof = c1.number_input("익절%", 0.0, 100.0, st.session_state.get(k_bp, 2.5), step=0.1, key=k_bp)
            bt_time = c2.number_input("존버일", 1, 100, st.session_state.get(k_bt, 10), key=k_bt)

        with st.expander("➖ 중간 (Middle)", expanded=False):
            c3, c4 = st.columns(2)
            k_mb=f"mb_{suffix}"; k_mp=f"mp_{suffix}"; k_mt=f"mt_{suffix}"
            md_buy = c3.number_input("매수점%", -30.0, 30.0, st.session_state.get(k_mb, -0.01), step=0.1, key=k_mb)
            md_prof = c4.number_input("익절%", 0.0, 100.0, st.session_state.get(k_mp, 2.8), step=0.1, key=k_mp)
            md_time = c3.number_input("존버일", 1, 100, st.session_state.get(k_mt, 15), key=k_mt)

        with st.expander("📈 천장 (Ceiling)", expanded=False):
            c5, c6 = st.columns(2)
            k_cc=f"cc_{suffix}"; k_cb=f"cb_{suffix}"; k_cp=f"cp_{suffix}"; k_ct=f"ct_{suffix}"
            cl_cond = c5.number_input(lbl_cl, 0.0, 100.0, st.session_state.get(k_cc, def_cl), step=step_val, key=k_cc)
            cl_buy = c6.number_input("매수점%", -30.0, 30.0, st.session_state.get(k_cb, -0.1), step=0.1, key=k_cb)
            cl_prof = c5.number_input("익절%", 0.0, 100.0, st.session_state.get(k_cp, 1.5), step=0.1, key=k_cp)
            cl_time = c6.number_input("존버일", 1, 100, st.session_state.get(k_ct, 40), key=k_ct)
        
        with st.expander("⚖️ 티어별 비중", expanded=False):
            base_key = f"base_w_{suffix}"
            if base_key in st.session_state: initial_data = st.session_state[base_key]
            else:
                default_data = {'Tier': [f'Tier {i}' for i in range(1, 11)], 'Bottom': [10.0]*10, 'Middle': [10.0]*10, 'Ceiling': [10.0]*10}
                initial_data = pd.DataFrame(default_data).set_index('Tier')
                st.session_state[base_key] = initial_data

            current_ver = st.session_state.editor_ver
            unique_key = f"w_{suffix}_v{current_ver}"
            edited_w = st.data_editor(initial_data, key=unique_key, column_config={"Bottom": st.column_config.NumberColumn("바닥%", format="%.1f%%"), "Middle": st.column_config.NumberColumn("중간%", format="%.1f%%"), "Ceiling": st.column_config.NumberColumn("천장%", format="%.1f%%")}, use_container_width=True)
            st.session_state[f"current_w_{suffix}"] = edited_w

        return {
            'strategy_type': strategy_type, 'use_bb_walk': use_bb_walk,
            'start_date': start_date, 'end_date': end_date,
            'initial_balance': balance, 'fee_rate': fee/100,
            'profit_rate': profit_rate/100.0, 'loss_rate': loss_rate/100.0,
            'loc_range': loc_range, 'add_order_cnt': add_order_cnt,
            'force_round': True, 'ma_window': ma_win, 
            'bt_cond': bt_cond, 'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
            'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
            'cl_cond': cl_cond, 'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
            'tier_weights': edited_w, 'label': key_prefix
        }

    with tab_s: params_s = render_strategy_inputs('s', '🛡️ 안정형')
    with tab_a: params_a = render_strategy_inputs('a', '🔥 공격형')
    
    st.markdown("---")
    if st.button("💾 현재 설정 저장하기", type="primary", use_container_width=True):
        if order_sheet_url: save_settings_to_gsheet(order_sheet_url)
        else: st.error("주문 전송 시트 URL을 먼저 입력해주세요.")

if sheet_url:
    df = load_data_from_gsheet(sheet_url)
    if df is not None:
        tab_dash, tab_lab, tab_mc, tab_opt = st.tabs(["📢 실전 대시보드", "🧪 백테스트 연구소", "🎲 몬테카를로 최적화", "🚀 Optuna 최적화"])

        # --- [탭 1: 실전 대시보드] ---
        with tab_dash:
            last_date_str = df.index[-1].strftime('%Y-%m-%d')
            st.header(f"📢 오늘의 투자 브리핑 ({last_date_str})")
            col_stable, col_agg = st.columns([1, 1])
            
            def render_dashboard(col, p_params, strategy_name, stock_name="SOXL"):
                hts_orders = []
                with col:
                    # // UI 개선: 전략명을 컨테이너 헤더로
                    with st.container(border=True):
                        st.markdown(f"### {strategy_name}")
                        st.caption(f"전략: {p_params['strategy_type']}")
                    res = backtest_engine_web(df, p_params)
                    if not res: st.error("데이터 부족"); return hts_orders

                    last_row = res['LastData']
                    daily_last = res['DailyLog'].iloc[-1]
                    
                    if p_params['strategy_type'] == 'RSI':
                        curr_val = last_row['RSI']; val_fmt = f"{curr_val:.2f}"; label_metric = "현재 RSI"
                    elif p_params['strategy_type'] == 'RSI 다이버전스':
                        curr_val = last_row['RSI']; val_fmt = f"{curr_val:.2f}"; label_metric = "현재 RSI (Div 감지)"
                    else:
                        curr_val = last_row['Basis_Disp']; val_fmt = f"{curr_val:.4f}"; label_metric = "현재 이격도"

                    if curr_val < p_params['bt_cond']: curr_phase = "📉 바닥"
                    elif curr_val > p_params['cl_cond']: curr_phase = "📈 천장"
                    else: curr_phase = "➖ 중간"
                    
                    # // UI 개선: KPI 메트릭 카드를 나란히 배치
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("💰 시드 자산", f"${daily_last['SeedEquity']:,.0f}")
                    kpi2.metric("🏦 보유 현금", f"${daily_last['Cash']:,.0f}")
                    kpi3.metric("📦 보유 슬롯", f"{len(res['CurrentHoldings'])}/10")
                    
                    # // UI 개선: 시장 상태를 프로그레스 바로 시각화
                    st.markdown(f"**{label_metric}: {val_fmt}** — {curr_phase}")
                    if p_params['strategy_type'].startswith('RSI'):
                        progress_val = min(max(curr_val / 100.0, 0.0), 1.0)
                    else:
                        progress_val = min(max((curr_val - 0.7) / 0.6, 0.0), 1.0)
                    st.progress(progress_val)
                    st.markdown("")

                    n_split = int(p_params['add_order_cnt'])
                    loc_range = p_params['loc_range']
                    next_tier = min(len(res['CurrentHoldings']) + 1, 10)
                    
                    if "바닥" in curr_phase: col_key = "Bottom"; start_rate = p_params['bt_buy']
                    elif "천장" in curr_phase: col_key = "Ceiling"; start_rate = p_params['cl_buy']
                    else: col_key = "Middle"; start_rate = p_params['md_buy']
                    
                    try: target_weight = p_params['tier_weights'].loc[f'Tier {next_tier}', col_key]
                    except: target_weight = 10.0
                    
                    one_time_seed = daily_last['SeedEquity'] * (target_weight / 100.0)
                    loc_price = excel_round_down(last_row['SOXL'] * (1 + start_rate/100.0), 2)

                    def get_smart_orders(seed, start_p, range_pct, split_cnt):
                        orders = []
                        if start_p <= 0: return orders
                        base_qty = int(seed / start_p)
                        orders.append({'price': start_p, 'qty': base_qty, 'type': 'MAIN'})
                        if split_cnt <= 0: return orders
                        multiplier = (1 + range_pct) if range_pct <= 0 else (1 - range_pct)
                        bot_p = excel_round_down(start_p * multiplier, 2)
                        fix_qty = max(0, int((seed/bot_p - seed/start_p)/split_cnt)) if bot_p > 0 else 0
                        for i in range(1, split_cnt + 1):
                            target_cum_qty = base_qty + (i * fix_qty)
                            next_p = excel_round_down(seed / target_cum_qty, 2)
                            if next_p > 0 and next_p < start_p: orders.append({'price': next_p, 'qty': fix_qty, 'type': 'ADD'})
                        return orders

                    # // UI 개선: 매수 주문을 틸 액센트 컨테이너로
                    st.markdown('<div class="buy-orders-box">', unsafe_allow_html=True)
                    st.markdown("#### 🛒 매수 주문")
                    buy_list = []
                    if len(res['CurrentHoldings']) < 10:
                        real_bet = min(one_time_seed, daily_last['Cash'])
                        net_bet = real_bet / (1 + p_params['fee_rate'])
                        orders = get_smart_orders(net_bet, loc_price, -1*(loc_range/100.0), n_split)
                        rem_cash = daily_last['Cash']
                        for i, o in enumerate(orders):
                            cost = o['price'] * o['qty']
                            status = "주문가능" if rem_cash >= cost else "현금부족"
                            if rem_cash >= cost: rem_cash -= cost
                            label = "⭐ MAIN" if o['type'] == 'MAIN' else f"💧 ADD #{i}"
                            buy_list.append({"구분": label, "가격": f"${o['price']}", "수량": f"{o['qty']}", "상태": status})
                    
                    if buy_list:
                        st.info(f"🆕 **신규 진입 (Tier {next_tier})**")
                        st.dataframe(pd.DataFrame(buy_list), hide_index=True, use_container_width=True)
                        for b in buy_list:
                            if b["상태"] == "주문가능":
                                hts_orders.append({"전략": strategy_name, "종목": stock_name, "주문유형": "매수", "주문타입": "LOC", "가격": float(b["가격"].replace('$','')), "수량": int(b["수량"])})
                    elif len(res['CurrentHoldings']) >= 10: st.warning("🚫 슬롯 꽉 참")
                    else: st.caption("매수 조건 미달")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # // UI 개선: 매도 주문을 코랄 액센트 컨테이너로
                    st.markdown('<div class="sell-orders-box">', unsafe_allow_html=True)
                    st.markdown("#### 💰 매도 주문")
                    if not res['CurrentHoldings']: st.caption("보유 없음")
                    else:
                        sell_list = []
                        for h in res['CurrentHoldings']:
                            buy_p, days, qty, mode, tier, buy_dt, _peak = h
                            if mode == 'Bottom': prof_rate = p_params['bt_prof']; time_limit = p_params['bt_time']
                            elif mode == 'Ceiling': prof_rate = p_params['cl_prof']; time_limit = p_params['cl_time']
                            else: prof_rate = p_params['md_prof']; time_limit = p_params['md_time']
                            
                            target_sell_p = excel_round_up(buy_p * (1 + prof_rate), 2)
                            curr_return = (last_row['SOXL'] - buy_p) / buy_p * 100
                            current_hold_days = days + 1
                            
                            if current_hold_days >= time_limit:
                                order_type = "🚨 MOC (시장가)"; order_price = "Market"; note = "TimeCut"
                            else:
                                order_type = "🎯 LOC (지정가)"; order_price = f"${target_sell_p}"; note = f"{current_hold_days}/{time_limit}일"

                            sell_list.append({"티어": f"T{tier}", "수량": f"{qty}주", "평단": f"${buy_p}", "수익률": f"{curr_return:.2f}%", "타입": order_type, "가격": order_price, "비고": note})
                            hts_orders.append({"전략": strategy_name, "종목": stock_name, "주문유형": "매도", "주문타입": "MOC" if "MOC" in order_type else "LOC", "가격": target_sell_p if "LOC" in order_type else 0, "수량": qty})
                        
                        def highlight_moc(row): return ['background-color: #ffcccc; color: black'] * len(row) if "MOC" in row['타입'] else [''] * len(row)
                        st.dataframe(pd.DataFrame(sell_list).style.apply(highlight_moc, axis=1), hide_index=True, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                return hts_orders

            orders_stable = render_dashboard(col_stable, params_s, "🛡️ 안정형")
            orders_agg = render_dashboard(col_agg, params_a, "🔥 공격형")
            
            # // UI 개선: HTS 전송 버튼을 강조 컨테이너로
            st.markdown("")
            all_orders = orders_stable + orders_agg
            if all_orders and order_sheet_url:
                hts_col1, hts_col2, hts_col3 = st.columns([1, 2, 1])
                with hts_col2:
                    if st.button("🚀 HTS 주문 전송", type="primary", use_container_width=True):
                        if send_orders_to_gsheet(pd.DataFrame(all_orders), order_sheet_url): st.success("✅ 전송 완료!")
                        else: st.error("❌ 전송 실패")

        # --- [탭 2: 백테스트 연구소] ---
        with tab_lab:
            st.info("🧪 여기서는 사이드바 설정과 무관하게 자유롭게 파라미터를 변경하여 테스트할 수 있습니다.")
            c_lab_in, c_lab_out = st.columns([1, 1])
            
            with c_lab_in:
                st.subheader("🛠️ 실험 조건")
                with st.form("lab_form"):
                    c_l1, c_l2 = st.columns(2)
                    lab_st_type = c_l1.radio("기준", ["MA 이격도", "RSI", "RSI 다이버전스"])
                    l_ma = c_l2.number_input("이평선", value=200)
                    
                    # [NEW] 볼린저 밴드 익절 지연 체크박스 (연구소용)
                    lab_use_bb = st.checkbox("🌭 볼린저 밴드 익절 지연 (Band Walk)", value=False)

                    today = datetime.date.today()
                    l_start = c_l1.date_input("시작", value=datetime.date(2010,1,1))
                    l_end = c_l2.date_input("종료", value=today)
                    
                    c_b1, c_b2, c_b3 = st.columns(3)
                    lab_balance = c_b1.number_input("💰 초기자본($)", value=10000, min_value=100, step=1000)
                    l_add = c_b2.number_input("분할", value=4)
                    l_rng = c_b3.number_input("범위(-%)", value=20.0)
                    
                    st.divider()
                    t_bot, t_mid, t_ceil = st.tabs(["📉 바닥", "➖ 중간", "📈 천장"])
                    with t_bot:
                        l_bc = st.number_input("진입 기준 (이하)", value=30.0 if lab_st_type.startswith('RSI') else 0.90)
                        c_bt1, c_bt2 = st.columns(2)
                        l_bb = c_bt1.number_input("매수(%)", value=15.0)
                        l_bp = c_bt2.number_input("익절(%)", value=5.0)
                        l_bt = st.number_input("존버일", value=10)
                    with t_mid:
                        c_md1, c_md2 = st.columns(2)
                        l_mb = c_md1.number_input("중간 매수(%)", value=-0.01)
                        l_mp = c_md2.number_input("중간 익절(%)", value=2.8)
                        l_mt = st.number_input("중간 존버일", value=15)
                    with t_ceil:
                        l_cc = st.number_input("진입 기준 (이상)", value=70.0 if lab_st_type.startswith('RSI') else 1.10)
                        c_cl1, c_cl2 = st.columns(2)
                        l_cb = c_cl1.number_input("천장 매수(%)", value=-0.1)
                        l_cp = c_cl2.number_input("천장 익절(%)", value=1.5)
                        l_ct = st.number_input("천장 존버일", value=40)

                    with st.expander("⚖️ 티어별 비중 설정"):
                        lab_default_w = pd.DataFrame({'Tier': [f'Tier {i}' for i in range(1, 11)], 'Bottom': [10.0]*10, 'Middle': [10.0]*10, 'Ceiling': [10.0]*10}).set_index('Tier')
                        lab_weights = st.data_editor(lab_default_w, key="lab_w_editor", use_container_width=True)

                    with st.expander("🪡 MDD 패치 옵션", expanded=False):
                        lab_use_5mode = st.checkbox("✅ 5모드 엔진 사용 (MA 이격도 전용)", value=False)
                        lab_compare_5m = st.checkbox("⚔️ 기존 3모드 vs 패치 비교", value=True)
                        lab_trailing = st.number_input("🛑 트레일링 스탑 (%)", 0.0, 100.0, 20.0, step=1.0) / 100

                    # // NEW: 5모드 모드별 UI 커스텀
                    with st.expander("🆕 5모드 상세 설정", expanded=False):
                        st.caption("ℹ️ 5모드 ON 시에만 적용. BOTTOM/NEUTRAL은 위의 바닥/중간 설정 재사용.")
                        st.markdown("**🟥 PANIC_BOTTOM** (급락 바닥 급매수)")
                        pc1, pc2, pc3 = st.columns(3)
                        lab_panic_buy = pc1.number_input("매수점%", -25.0, 0.0, -20.0, step=1.0, key="panic_buy")
                        lab_panic_prof = pc2.number_input("익절%", 3.0, 6.0, 4.0, step=0.1, key="panic_prof")
                        lab_panic_time = pc3.number_input("존버일", 5, 12, 8, step=1, key="panic_time")
                        st.markdown("**🟡 BEARISH** (하락 전환 구간)")
                        bc1, bc2, bc3 = st.columns(3)
                        lab_bear_buy = bc1.number_input("매수점%", -1.0, 0.0, max(l_mb, -0.5), step=0.1, key="bear_buy")
                        lab_bear_prof = bc2.number_input("익절%", 1.5, 3.0, max(l_mp, 2.0), step=0.1, key="bear_prof")
                        lab_bear_time = bc3.number_input("존버일", 20, 35, max(20, min(l_mt, 25)), step=1, key="bear_time")
                        st.markdown("**🟠 CEILING** (천장 구간)")
                        cc1, cc2, cc3 = st.columns(3)
                        lab_ceil_buy = cc1.number_input("매수점%", -0.5, 0.0, l_cb, step=0.1, key="ceil_buy")
                        lab_ceil_prof = cc2.number_input("익절%", 1.0, 2.5, l_cp, step=0.1, key="ceil_prof")
                        lab_ceil_time = cc3.number_input("존버일", 30, 50, l_ct, step=1, key="ceil_time")
                        st.markdown("---")
                        st.markdown("**⚖️ Weight Factor** (고정)")
                        wfc1, wfc2, wfc3, wfc4, wfc5 = st.columns(5)
                        wfc1.metric("PANIC", "1.5x")
                        wfc2.metric("BOTTOM", "1.2x")
                        wfc3.metric("NEUTRAL", "1.0x")
                        wfc4.metric("BEARISH", "0.7x")
                        wfc5.metric("CEILING", "0.5x")
                        st.markdown("---")
                        st.markdown("**🎯 모드 분류 Threshold**")
                        fm_c1, fm_c2 = st.columns(2)
                        lab_panic_disp = fm_c1.number_input("🟥 PANIC 이격도 (<)", value=0.82, step=0.01, format="%.2f")
                        lab_panic_rsi = fm_c2.number_input("🟥 PANIC RSI (<)", value=25.0, step=1.0)
                        lab_panic_bb = fm_c1.number_input("🟥 PANIC BB_Pos (<)", value=-1.8, step=0.1)
                        lab_bear_rsi = fm_c2.number_input("🟡 BEARISH RSI (>)", value=60.0, step=1.0)
                        lab_bear_disp = fm_c1.number_input("🟡 BEARISH 이격도 (>)", value=1.05, step=0.01, format="%.2f")
                        lab_ceil_rsi = fm_c2.number_input("🟠 CEILING RSI (>)", value=65.0, step=1.0)
                        lab_bot_rsi = fm_c1.number_input("🔵 BOTTOM RSI (<)", value=45.0, step=1.0)

                    lab_run = st.form_submit_button("🚀 백테스트 실행", type="primary", use_container_width=True)

            with c_lab_out:
                if lab_run:
                    lab_params = params_s.copy()
                    lab_params.update({
                        'initial_balance': lab_balance,
                        'strategy_type': lab_st_type, 'ma_window': l_ma, 'use_bb_walk': lab_use_bb,
                        'start_date': l_start, 'end_date': l_end,
                        'add_order_cnt': l_add, 'loc_range': l_rng,
                        'bt_cond': l_bc, 'bt_buy': l_bb, 'bt_prof': l_bp/100, 'bt_time': l_bt,
                        'md_buy': l_mb, 'md_prof': l_mp/100, 'md_time': l_mt,
                        'cl_cond': l_cc, 'cl_buy': l_cb, 'cl_prof': l_cp/100, 'cl_time': l_ct,
                        'tier_weights': lab_weights,
                        'trailing_pct': lab_trailing, 'use_5mode': lab_use_5mode,
                        # // NEW: 5모드 모드별 커스텀 (매수/익절/존버일)
                        'panic_buy': lab_panic_buy, 'panic_prof': lab_panic_prof/100, 'panic_time': lab_panic_time,
                        'bear_buy': lab_bear_buy, 'bear_prof': lab_bear_prof/100, 'bear_time': lab_bear_time,
                        'ceil_buy': lab_ceil_buy, 'ceil_prof': lab_ceil_prof/100, 'ceil_time': lab_ceil_time,
                        # // NEW: 5모드 분류 threshold
                        'panic_disp_th': lab_panic_disp, 'panic_rsi_th': lab_panic_rsi, 'panic_bb_th': lab_panic_bb,
                        'bear_rsi_th': lab_bear_rsi, 'bear_disp_th': lab_bear_disp,
                        'ceil_rsi_th': lab_ceil_rsi, 'bot_rsi_th': lab_bot_rsi
                    })

                    if lab_use_5mode and lab_compare_5m:
                        # --- 3모드 vs 5모드 비교 ---
                        comp = compare_5mode(df, lab_params)
                        if comp:
                            st.markdown("### ⚔️ 기존 3모드 vs 패치 비교")
                            st.dataframe(comp['comparison'], hide_index=True, use_container_width=True)

                            st.subheader("📈 자산 추이 비교")
                            chart_df = pd.DataFrame({
                                '🔵 기존 3모드': comp['original']['Series'],
                                '🟢 패치': comp['fivemode']['Series']
                            })
                            st.line_chart(chart_df, color=["#6C7B95", "#4ECDC4"])

                            tab_3m, tab_5m = st.tabs(["🔵 기존 3모드 매매기록", "🟢 패치 매매기록"])
                            with tab_3m:
                                st.dataframe(comp['original']['TradeLog'], use_container_width=True, height=300)
                            with tab_5m:
                                st.dataframe(comp['fivemode']['TradeLog'], use_container_width=True, height=300)

                            # 향상된 분석
                            st.markdown("---")
                            st.subheader("📊 5모드 심층 분석")
                            stats = analyze_backtest_results(comp['fivemode'])
                            if stats:
                                t_p, t_t, t_y, t_m, t_a = st.tabs(["구간별", "티어별", "연도별", "모드별", "고급 메트릭"])
                                with t_p: st.dataframe(stats['period_stats'], hide_index=True, use_container_width=True)
                                with t_t: st.dataframe(stats['tier_stats'], hide_index=True, use_container_width=True)
                                with t_y: st.dataframe(stats['yearly_stats'], hide_index=True, use_container_width=True)
                                with t_m: st.dataframe(stats['mode_stats'], hide_index=True, use_container_width=True)
                                with t_a:
                                    adv = stats['advanced_metrics']
                                    ac1, ac2, ac3, ac4 = st.columns(4)
                                    ac1.metric("Sharpe", adv['Sharpe'])
                                    ac2.metric("Sortino", adv['Sortino'])
                                    ac3.metric("최대연속수익", adv['최대연속수익'])
                                    ac4.metric("최대연속손실", adv['최대연속손실'])
                                    ac1.metric("총거래수", adv['총거래수'])
                                    ac2.metric("평균수익", f"${adv['평균수익']:,.2f}")
                                    ac3.metric("거래빈도", f"{adv['거래빈도(일/건)']}일/건")
                        else:
                            st.error("비교 실행 실패 — 데이터를 확인해주세요.")
                    else:
                        # --- 단일 실행 모드 ---
                        engine_fn = backtest_engine_5mode if lab_use_5mode else backtest_engine_web
                        res_lab = engine_fn(df, lab_params)
                        if res_lab:
                            with st.container(border=True):
                                m1, m2, m3, m4, m5 = st.columns(5)
                                m1.metric("최종 자산", f"${res_lab['Final']:,.0f}")
                                m2.metric("수익률", f"{res_lab['Return']:.2f}%")
                                m3.metric("CAGR", f"{res_lab['CAGR']:.2f}%")
                                m4.metric("MDD", f"{res_lab['MDD']:.2f}%")
                                m5.metric("승률", f"{res_lab['WinRate']}%")
                            st.subheader("📈 자산 추이")
                            st.line_chart(res_lab['Series'], color="#4ECDC4")
                            st.subheader("📜 매매 기록")
                            st.dataframe(res_lab['TradeLog'], use_container_width=True, height=400)

                            # 향상된 분석
                            st.markdown("---")
                            st.subheader("📊 심층 분석")
                            stats = analyze_backtest_results(res_lab)
                            if stats:
                                t_p, t_t, t_y, t_m, t_a = st.tabs(["구간별", "티어별", "연도별", "모드별", "고급 메트릭"])
                                with t_p: st.dataframe(stats['period_stats'], hide_index=True, use_container_width=True)
                                with t_t: st.dataframe(stats['tier_stats'], hide_index=True, use_container_width=True)
                                with t_y: st.dataframe(stats['yearly_stats'], hide_index=True, use_container_width=True)
                                with t_m: st.dataframe(stats['mode_stats'], hide_index=True, use_container_width=True)
                                with t_a:
                                    adv = stats['advanced_metrics']
                                    ac1, ac2, ac3, ac4 = st.columns(4)
                                    ac1.metric("Sharpe", adv['Sharpe'])
                                    ac2.metric("Sortino", adv['Sortino'])
                                    ac3.metric("최대연속수익", adv['최대연속수익'])
                                    ac4.metric("최대연속손실", adv['최대연속손실'])
                                    ac1.metric("총거래수", adv['총거래수'])
                                    ac2.metric("평균수익", f"${adv['평균수익']:,.2f}")
                                    ac3.metric("거래빈도", f"{adv['거래빈도(일/건)']}일/건")

        # === 공통 헬퍼: 점수 계산 ===
        def _calc_score(res):
            """CAGR - 2*|MDD| + 0.5*Sharpe"""
            if not res: return float('-inf'), 0, 0, 0
            cagr = res.get('CAGR', 0); mdd = res.get('MDD', 0); wr = res.get('WinRate', 0)
            try:
                daily_ret = res['Series'].pct_change().dropna()
                sharpe = (daily_ret.mean() / daily_ret.std()) * (252**0.5) if daily_ret.std() > 0 else 0
            except: sharpe = 0
            score = cagr - 2 * abs(mdd) + 0.5 * sharpe
            return round(score, 2), round(cagr, 2), round(mdd, 2), round(sharpe, 2)

        def _build_mc_params(base_params, mc_type, mc_use_bb, mc_trailing, start, end,
                             rnd_bc, rnd_cc, rnd_bb, rnd_bp, rnd_bt,
                             rnd_mb, rnd_mp, rnd_mt, rnd_cb, rnd_cp, rnd_ct, extra_5m=None):
            p = base_params.copy()
            p.update({
                'strategy_type': mc_type, 'use_bb_walk': mc_use_bb,
                'start_date': start, 'end_date': end, 'trailing_pct': mc_trailing,
                'bt_cond': rnd_bc, 'cl_cond': rnd_cc,
                'bt_buy': rnd_bb, 'bt_prof': rnd_bp/100, 'bt_time': rnd_bt,
                'md_buy': rnd_mb, 'md_prof': rnd_mp/100, 'md_time': rnd_mt,
                'cl_buy': rnd_cb, 'cl_prof': rnd_cp/100, 'cl_time': rnd_ct
            })
            if extra_5m: p.update(extra_5m)
            return p

        # --- [탭 3: 몬테카를로 최적화] ---
        with tab_mc:
            st.subheader("🎲 몬테카를로 시뮬레이션 (전역 최적화)")
            st.caption("Score = CAGR − 2×|MDD| + 0.5×Sharpe | OOS 검증으로 과최적화 방지")

            c_mc1, c_mc2 = st.columns([1, 1])
            with c_mc1:
                with st.form("mc_form"):
                    mc_trials = st.number_input("1회 시도 횟수", 10, 5000, 100)
                    mc_type = st.radio("전략 타입", ["MA 이격도", "RSI", "RSI 다이버전스"], horizontal=True)
                    mc_use_bb = st.checkbox("🌭 BB Walk", value=False)
                    mc_use_5mode = st.checkbox("🆕 5모드 엔진 (MA 이격도 전용)", value=False)
                    mc_trailing = st.number_input("🛑 트레일링 (%)", 0.0, 100.0, 0.0, step=5.0) / 100

                    st.markdown("#### 📅 기간")
                    cd1, cd2 = st.columns(2)
                    mc_start = cd1.date_input("시작일", value=datetime.date(2010,1,1), key="mc_s")
                    mc_end = cd2.date_input("종료일", value=datetime.date.today(), key="mc_e")

                    # // NEW: OOS 과최적화 방지 설정
                    st.markdown("#### 🛡️ 과최적화 방지")
                    mc_use_oos = st.checkbox("📊 OOS (Out-of-Sample) 검증", value=True)
                    mc_oos_split = st.date_input("IS→OOS 분기일", value=datetime.date(2022,1,1), key="mc_oos_dt",
                                                  help="이 날짜까지 IS(학습), 이후 OOS(검증)")
                    mc_use_wfo = st.checkbox("🔄 Walk-Forward 분석", value=False)
                    cwf1, cwf2 = st.columns(2)
                    mc_wfo_win = cwf1.number_input("WFO 윈도우(년)", 2, 8, 3, key="mc_wfw")
                    mc_wfo_step = cwf2.number_input("WFO 스텝(년)", 1, 3, 1, key="mc_wfs")

                    st.markdown("#### 🎯 3모드 범위")
                    with st.expander("진입/탈출 기준", expanded=True):
                        c1, c2 = st.columns(2)
                        if mc_type.startswith('RSI'):
                            r_bc_min = c1.number_input("바닥(Min)", value=25.0); r_bc_max = c2.number_input("바닥(Max)", value=35.0)
                            r_cc_min = c1.number_input("천장(Min)", value=70.0); r_cc_max = c2.number_input("천장(Max)", value=80.0)
                        else:
                            r_bc_min = c1.number_input("바닥(Min)", value=0.85); r_bc_max = c2.number_input("바닥(Max)", value=0.95)
                            r_cc_min = c1.number_input("천장(Min)", value=1.05); r_cc_max = c2.number_input("천장(Max)", value=1.15)
                    with st.expander("바닥/중간/천장", expanded=False):
                        c1, c2 = st.columns(2)
                        r_bb_min=c1.number_input("B매수%(Min)",value=10.0);r_bb_max=c2.number_input("B매수%(Max)",value=20.0)
                        r_bp_min=c1.number_input("B익절%(Min)",value=3.0);r_bp_max=c2.number_input("B익절%(Max)",value=10.0)
                        r_bt_min=c1.number_input("B존버(Min)",value=10,step=1);r_bt_max=c2.number_input("B존버(Max)",value=40,step=1)
                        r_mb_min=c1.number_input("M매수%(Min)",value=-2.0);r_mb_max=c2.number_input("M매수%(Max)",value=2.0)
                        r_mp_min=c1.number_input("M익절%(Min)",value=2.0);r_mp_max=c2.number_input("M익절%(Max)",value=8.0)
                        r_mt_min=c1.number_input("M존버(Min)",value=10,step=1);r_mt_max=c2.number_input("M존버(Max)",value=30,step=1)
                        r_cb_min=c1.number_input("C매수%(Min)",value=-10.0);r_cb_max=c2.number_input("C매수%(Max)",value=-5.0)
                        r_cp_min=c1.number_input("C익절%(Min)",value=1.0);r_cp_max=c2.number_input("C익절%(Max)",value=5.0)
                        r_ct_min=c1.number_input("C존버(Min)",value=20,step=1);r_ct_max=c2.number_input("C존버(Max)",value=60,step=1)
                    with st.expander("5모드 범위 (MA 전용)", expanded=False):
                        c1, c2 = st.columns(2)
                        r_pb_min=c1.number_input("P매수%(Min)",value=-10.0);r_pb_max=c2.number_input("P매수%(Max)",value=30.0)
                        r_pp_min=c1.number_input("P익절%(Min)",value=2.0);r_pp_max=c2.number_input("P익절%(Max)",value=6.0)
                        r_pt_min=c1.number_input("P존버(Min)",value=5,step=1);r_pt_max=c2.number_input("P존버(Max)",value=15,step=1)
                        r_brb_min=c1.number_input("BR매수%(Min)",value=-5.0);r_brb_max=c2.number_input("BR매수%(Max)",value=5.0)
                        r_brp_min=c1.number_input("BR익절%(Min)",value=1.0);r_brp_max=c2.number_input("BR익절%(Max)",value=4.0)
                        r_brt_min=c1.number_input("BR존버(Min)",value=15,step=1);r_brt_max=c2.number_input("BR존버(Max)",value=35,step=1)
                        r_pd_min=c1.number_input("PDisp(Min)",value=0.75);r_pd_max=c2.number_input("PDisp(Max)",value=0.90)
                        r_pr_min=c1.number_input("PRSI(Min)",value=20.0);r_pr_max=c2.number_input("PRSI(Max)",value=35.0)
                        r_brd_min=c1.number_input("BRDisp(Min)",value=1.02);r_brd_max=c2.number_input("BRDisp(Max)",value=1.12)
                        r_brr_min=c1.number_input("BRRSI(Min)",value=55.0);r_brr_max=c2.number_input("BRRSI(Max)",value=70.0)

                    mc_run = st.form_submit_button("🎲 시뮬레이션 시작")

                if st.button("🗑️ 기록 초기화"):
                    st.session_state.opt_results = pd.DataFrame()
                    st.success("초기화됨")

            with c_mc2:
                if mc_run:
                    new_results = []
                    bar = st.progress(0)
                    is_5mode = mc_use_5mode and mc_type == 'MA 이격도'

                    for i in range(mc_trials):
                        rnd_bc=random.uniform(r_bc_min,r_bc_max);rnd_cc=random.uniform(r_cc_min,r_cc_max)
                        rnd_bb=random.uniform(r_bb_min,r_bb_max);rnd_bp=random.uniform(r_bp_min,r_bp_max);rnd_bt=random.randint(int(r_bt_min),int(r_bt_max))
                        rnd_mb=random.uniform(r_mb_min,r_mb_max);rnd_mp=random.uniform(r_mp_min,r_mp_max);rnd_mt=random.randint(int(r_mt_min),int(r_mt_max))
                        rnd_cb=random.uniform(r_cb_min,r_cb_max);rnd_cp=random.uniform(r_cp_min,r_cp_max);rnd_ct=random.randint(int(r_ct_min),int(r_ct_max))

                        extra_5m = None
                        row_extra = {}
                        if is_5mode:
                            rnd_pb=random.uniform(r_pb_min,r_pb_max);rnd_pp=random.uniform(r_pp_min,r_pp_max);rnd_pt=random.randint(int(r_pt_min),int(r_pt_max))
                            rnd_brb=random.uniform(r_brb_min,r_brb_max);rnd_brp=random.uniform(r_brp_min,r_brp_max);rnd_brt=random.randint(int(r_brt_min),int(r_brt_max))
                            rnd_pd=random.uniform(r_pd_min,r_pd_max);rnd_pr=random.uniform(r_pr_min,r_pr_max)
                            rnd_brd=random.uniform(r_brd_min,r_brd_max);rnd_brr=random.uniform(r_brr_min,r_brr_max)
                            extra_5m = {
                                'panic_buy':rnd_pb,'panic_prof':rnd_pp/100,'panic_time':rnd_pt,
                                'bear_buy':rnd_brb,'bear_prof':rnd_brp/100,'bear_time':rnd_brt,
                                'panic_disp_th':rnd_pd,'panic_rsi_th':rnd_pr,'bear_disp_th':rnd_brd,'bear_rsi_th':rnd_brr,
                            }
                            row_extra = {'P_Buy':round(rnd_pb,1),'P_Prof':round(rnd_pp,1),'P_Time':rnd_pt,
                                         'BR_Buy':round(rnd_brb,1),'BR_Prof':round(rnd_brp,1),'BR_Time':rnd_brt}

                        engine_fn = backtest_engine_5mode if is_5mode else backtest_engine_web

                        # IS 기간
                        is_end = mc_oos_split if mc_use_oos else mc_end
                        mc_params = _build_mc_params(params_s, mc_type, mc_use_bb, mc_trailing,
                                                     mc_start, is_end, rnd_bc, rnd_cc,
                                                     rnd_bb, rnd_bp, rnd_bt, rnd_mb, rnd_mp, rnd_mt,
                                                     rnd_cb, rnd_cp, rnd_ct, extra_5m)
                        try: res_is = engine_fn(df, mc_params)
                        except: res_is = None
                        is_score, is_cagr, is_mdd, is_sharpe = _calc_score(res_is)
                        if is_score <= float('-inf'): bar.progress((i+1)/mc_trials); continue

                        # OOS 기간
                        oos_score = 0; oos_cagr = 0; oos_mdd = 0
                        if mc_use_oos:
                            mc_params_oos = mc_params.copy()
                            mc_params_oos['start_date'] = mc_oos_split
                            mc_params_oos['end_date'] = mc_end
                            try: res_oos = engine_fn(df, mc_params_oos)
                            except: res_oos = None
                            oos_score, oos_cagr, oos_mdd, _ = _calc_score(res_oos)

                        # WFO (Walk-Forward)
                        wfo_score = 0
                        if mc_use_wfo:
                            wfo_scores = []
                            yr_start = mc_start.year
                            yr_end = mc_end.year
                            for ws in range(yr_start, yr_end - mc_wfo_win, mc_wfo_step):
                                is_s = datetime.date(ws, 1, 1)
                                is_e = datetime.date(ws + mc_wfo_win, 1, 1)
                                oos_s = is_e
                                oos_e = datetime.date(min(ws + mc_wfo_win + mc_wfo_step, yr_end), 12, 31)
                                if oos_s >= oos_e: continue
                                p_wfo = mc_params.copy()
                                p_wfo['start_date'] = oos_s; p_wfo['end_date'] = oos_e
                                try: r_wfo = engine_fn(df, p_wfo)
                                except: r_wfo = None
                                s, _, _, _ = _calc_score(r_wfo)
                                if s > float('-inf'): wfo_scores.append(s)
                            wfo_score = round(np.mean(wfo_scores), 2) if wfo_scores else 0

                        # 안정성 점수
                        oos_ratio = oos_cagr / is_cagr if is_cagr != 0 and mc_use_oos else 1.0
                        stability = round(min(max(oos_ratio, 0), 2.0), 2)

                        row = {
                            'Type': mc_type + (' 5M' if is_5mode else ''),
                            'B_Ref':round(rnd_bc,2),'C_Ref':round(rnd_cc,2),
                            'B_Buy':round(rnd_bb,1),'B_Prof':round(rnd_bp,1),'B_Time':rnd_bt,
                            'M_Buy':round(rnd_mb,1),'M_Prof':round(rnd_mp,1),'M_Time':rnd_mt,
                            'C_Buy':round(rnd_cb,1),'C_Prof':round(rnd_cp,1),'C_Time':rnd_ct,
                            **row_extra,
                            'IS_CAGR':is_cagr,'IS_MDD':is_mdd,'IS_Score':is_score,
                            'OOS_CAGR':oos_cagr,'OOS_MDD':oos_mdd,'OOS_Score':oos_score,
                            'WFO':wfo_score,'Stab':stability,
                            'Score': round(is_score * 0.4 + oos_score * 0.4 + wfo_score * 0.2, 2) if mc_use_oos else is_score
                        }
                        new_results.append(row)
                        bar.progress((i+1)/mc_trials)

                    if new_results:
                        new_df = pd.DataFrame(new_results)
                        if not st.session_state.opt_results.empty:
                            if set(new_df.columns) != set(st.session_state.opt_results.columns):
                                st.session_state.opt_results = new_df
                            else:
                                st.session_state.opt_results = pd.concat([st.session_state.opt_results, new_df], ignore_index=True)
                        else:
                            st.session_state.opt_results = new_df
                        st.session_state.opt_results = st.session_state.opt_results.drop_duplicates().sort_values('Score', ascending=False)

                if isinstance(st.session_state.opt_results, pd.DataFrame) and not st.session_state.opt_results.empty:
                    st.write("🏆 **TOP 10** — Score = 0.4×IS + 0.4×OOS + 0.2×WFO")
                    show_cols = [c for c in ['Type','B_Ref','C_Ref','IS_CAGR','IS_MDD','OOS_CAGR','OOS_MDD','WFO','Stab','Score'] if c in st.session_state.opt_results.columns]
                    st.dataframe(st.session_state.opt_results[show_cols].head(10), use_container_width=True)

                    best = st.session_state.opt_results.iloc[0]
                    with st.expander("🌟 BEST 상세", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        c1.info(f"**📉 바닥**\n- 기준: {best.get('B_Ref','')}\n- 매수: {best['B_Buy']}%\n- 익절: {best['B_Prof']}%\n- 존버: {best['B_Time']}일")
                        c2.warning(f"**➖ 중간**\n- 매수: {best['M_Buy']}%\n- 익절: {best['M_Prof']}%\n- 존버: {best['M_Time']}일")
                        c3.error(f"**📈 천장**\n- 기준: {best.get('C_Ref','')}\n- 매수: {best['C_Buy']}%\n- 익절: {best['C_Prof']}%\n- 존버: {best['C_Time']}일")
                        if 'P_Buy' in best.index:
                            c1.success(f"**🟥 PANIC**\n- 매수:{best['P_Buy']}%\n- 익절:{best['P_Prof']}%\n- 존버:{best['P_Time']}일")
                            c2.success(f"**🟡 BEAR**\n- 매수:{best['BR_Buy']}%\n- 익절:{best['BR_Prof']}%\n- 존버:{best['BR_Time']}일")
                        st.success(f"📊 IS: {best['IS_CAGR']}%/{best['IS_MDD']}% | OOS: {best['OOS_CAGR']}%/{best['OOS_MDD']}% | 안정성: {best['Stab']} | Score: {best['Score']}")

                    # 상위10 통계
                    top10 = st.session_state.opt_results.head(10)
                    st.markdown("#### 📈 상위10 파라미터 평균 ± std")
                    num_cols = [c for c in top10.select_dtypes(include=[np.number]).columns if c not in ['Score','IS_Score','OOS_Score','WFO','Stab']]
                    if num_cols:
                        stats_df = pd.DataFrame({'Mean': top10[num_cols].mean().round(2), 'Std': top10[num_cols].std().round(2)})
                        stats_df['CV%'] = (stats_df['Std'] / stats_df['Mean'].abs().replace(0, 1) * 100).round(1)
                        st.dataframe(stats_df, use_container_width=True)

                    # 산점도
                    fig, ax = plt.subplots(figsize=(8, 5))
                    data = st.session_state.opt_results
                    sc = ax.scatter(data['IS_MDD'], data['IS_CAGR'], c=data['Score'], cmap='viridis', alpha=0.5, label='IS')
                    if 'OOS_MDD' in data.columns and data['OOS_CAGR'].abs().sum() > 0:
                        ax.scatter(data['OOS_MDD'], data['OOS_CAGR'], marker='x', c='red', alpha=0.3, label='OOS', s=20)
                    ax.set_xlabel('MDD (%)'); ax.set_ylabel('CAGR (%)')
                    ax.set_title('IS vs OOS (Score color)'); ax.legend()
                    plt.colorbar(sc, label='Score')
                    st.pyplot(fig, use_container_width=True)

        # --- [탭 4: Optuna 최적화] ---
        with tab_opt:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                _HAS_OPTUNA = True
            except ImportError:
                _HAS_OPTUNA = False

            if not _HAS_OPTUNA:
                st.error("⚠️ Optuna 미설치. `pip install optuna` 실행 후 새로고침.")
                st.code("pip install optuna", language="bash")
            else:
                st.subheader("🚀 Optuna 베이지안 최적화")
                st.caption("TPE + OOS 검증 + Cross-Validation으로 과최적화 방지")

                if 'optuna_results' not in st.session_state: st.session_state.optuna_results = None
                if 'optuna_study' not in st.session_state: st.session_state.optuna_study = None

                c_opt1, c_opt2 = st.columns([1, 1])
                with c_opt1:
                    with st.form("optuna_form"):
                        opt_trials = st.number_input("시도 횟수", 50, 1000, 200, step=50, key="opt_n")
                        opt_type = st.radio("전략 타입", ["MA 이격도", "RSI", "RSI 다이버전스"], horizontal=True, key="opt_type")
                        opt_use_bb = st.checkbox("🌭 BB Walk", value=False, key="opt_bb")
                        opt_use_5mode = st.checkbox("🆕 5모드 (MA 이격도 전용)", value=False, key="opt_5mode")
                        opt_trailing = st.number_input("🛑 트레일링 (%)", 0.0, 100.0, 0.0, step=5.0, key="opt_trail") / 100

                        st.markdown("#### 📅 기간")
                        oc1, oc2 = st.columns(2)
                        opt_start = oc1.date_input("시작일", value=datetime.date(2010,1,1), key="opt_start")
                        opt_end = oc2.date_input("종료일", value=datetime.date.today(), key="opt_end")

                        st.markdown("#### 🛡️ 과최적화 방지")
                        opt_use_oos = st.checkbox("📊 OOS 검증", value=True, key="opt_oos")
                        opt_oos_split = st.date_input("IS→OOS 분기일", value=datetime.date(2022,1,1), key="opt_oos_dt")
                        opt_use_cv = st.checkbox("🔀 시계열 Cross-Validation (5-fold)", value=False, key="opt_cv")
                        opt_repeat = st.number_input("🔁 반복 횟수 (안정성)", 1, 5, 1, key="opt_rep")

                        st.markdown("#### 🎯 목표")
                        opt_objective = st.selectbox("최적화 목표", ["CAGR−2|MDD|+0.5Sharpe (균형)", "CAGR 최대화", "MDD 최소화", "Sharpe 최대화"], key="opt_obj")

                        opt_run = st.form_submit_button("🚀 최적화 시작", type="primary")

                with c_opt2:
                    if opt_run:
                        is_5mode_opt = opt_use_5mode and opt_type == 'MA 이격도'
                        status_box = st.empty()
                        bar = st.progress(0)
                        trial_counter = {'n': 0}

                        def optuna_objective(trial):
                            if opt_type.startswith('RSI'):
                                bt_cond = trial.suggest_float('bt_cond', 25, 35)
                                cl_cond = trial.suggest_float('cl_cond', 70, 80)
                            else:
                                bt_cond = trial.suggest_float('bt_cond', 0.85, 0.95)
                                cl_cond = trial.suggest_float('cl_cond', 1.05, 1.15)
                            bt_buy = trial.suggest_float('bt_buy', 5, 25)
                            bt_prof = trial.suggest_float('bt_prof', 2, 12)
                            bt_time = trial.suggest_int('bt_time', 5, 40)
                            md_buy = trial.suggest_float('md_buy', -3, 3)
                            md_prof = trial.suggest_float('md_prof', 1.5, 8)
                            md_time = trial.suggest_int('md_time', 8, 30)
                            cl_buy = trial.suggest_float('cl_buy', -12, -1)
                            cl_prof = trial.suggest_float('cl_prof', 1, 6)
                            cl_time = trial.suggest_int('cl_time', 15, 60)

                            p = params_s.copy()
                            is_end_dt = opt_oos_split if opt_use_oos else opt_end
                            p.update({
                                'strategy_type': opt_type, 'use_bb_walk': opt_use_bb,
                                'start_date': opt_start, 'end_date': is_end_dt,
                                'trailing_pct': opt_trailing,
                                'bt_cond': bt_cond, 'cl_cond': cl_cond,
                                'bt_buy': bt_buy, 'bt_prof': bt_prof/100, 'bt_time': bt_time,
                                'md_buy': md_buy, 'md_prof': md_prof/100, 'md_time': md_time,
                                'cl_buy': cl_buy, 'cl_prof': cl_prof/100, 'cl_time': cl_time,
                            })

                            if is_5mode_opt:
                                p.update({
                                    'panic_buy': trial.suggest_float('panic_buy', -10, 30),
                                    'panic_prof': trial.suggest_float('panic_prof', 2, 6) / 100,
                                    'panic_time': trial.suggest_int('panic_time', 5, 15),
                                    'bear_buy': trial.suggest_float('bear_buy', -5, 5),
                                    'bear_prof': trial.suggest_float('bear_prof', 1, 4) / 100,
                                    'bear_time': trial.suggest_int('bear_time', 15, 35),
                                    'panic_disp_th': trial.suggest_float('panic_disp_th', 0.75, 0.90),
                                    'panic_rsi_th': trial.suggest_float('panic_rsi_th', 20, 35),
                                    'panic_bb_th': trial.suggest_float('panic_bb_th', -2.5, -1.2),
                                    'bear_disp_th': trial.suggest_float('bear_disp_th', 1.02, 1.12),
                                    'bear_rsi_th': trial.suggest_float('bear_rsi_th', 55, 70),
                                    'ceil_rsi_th': trial.suggest_float('ceil_rsi_th', 60, 75),
                                    'bot_rsi_th': trial.suggest_float('bot_rsi_th', 35, 50),
                                })

                            engine_fn = backtest_engine_5mode if is_5mode_opt else backtest_engine_web

                            # IS 백테스트
                            try: res_is = engine_fn(df, p)
                            except: return float('-inf')
                            if not res_is: return float('-inf')
                            is_score, is_cagr, is_mdd, is_sharpe = _calc_score(res_is)

                            # OOS 백테스트
                            oos_score = 0; oos_cagr = 0
                            if opt_use_oos:
                                p_oos = p.copy()
                                p_oos['start_date'] = opt_oos_split; p_oos['end_date'] = opt_end
                                try: res_oos = engine_fn(df, p_oos)
                                except: res_oos = None
                                oos_score, oos_cagr, _, _ = _calc_score(res_oos)

                            # CV (시계열 5-fold)
                            cv_score = 0
                            if opt_use_cv:
                                total_days = (opt_end - opt_start).days
                                fold_size = total_days // 5
                                cv_scores = []
                                for fold in range(5):
                                    val_start = opt_start + datetime.timedelta(days=fold_size * fold)
                                    val_end = val_start + datetime.timedelta(days=fold_size)
                                    if val_end > opt_end: val_end = opt_end
                                    p_cv = p.copy()
                                    p_cv['start_date'] = val_start; p_cv['end_date'] = val_end
                                    try: r_cv = engine_fn(df, p_cv)
                                    except: r_cv = None
                                    s, _, _, _ = _calc_score(r_cv)
                                    if s > float('-inf'): cv_scores.append(s)
                                cv_score = np.mean(cv_scores) if cv_scores else 0

                            trial.set_user_attr('IS_CAGR', is_cagr)
                            trial.set_user_attr('IS_MDD', is_mdd)
                            trial.set_user_attr('IS_Sharpe', is_sharpe)
                            trial.set_user_attr('OOS_CAGR', oos_cagr)
                            trial.set_user_attr('OOS_Score', oos_score)
                            trial.set_user_attr('CV_Score', round(cv_score, 2))
                            trial.set_user_attr('WinRate', res_is.get('WinRate', 0))

                            oos_ratio = oos_cagr / is_cagr if is_cagr != 0 and opt_use_oos else 1.0
                            stability = min(max(oos_ratio, 0), 2.0)
                            trial.set_user_attr('Stability', round(stability, 2))

                            trial_counter['n'] += 1
                            bar.progress(min(trial_counter['n'] / opt_trials, 1.0))
                            try:
                                best_v = trial.study.best_value
                                status_box.caption(f"⏳ {trial_counter['n']}/{opt_trials} — Best: {best_v:.2f}")
                            except ValueError:
                                status_box.caption(f"⏳ {trial_counter['n']}/{opt_trials}")

                            # 최종 점수
                            if opt_objective.startswith("CAGR−"):
                                final = is_score
                                if opt_use_oos: final = is_score * 0.4 + oos_score * 0.4 + cv_score * 0.2 if opt_use_cv else is_score * 0.5 + oos_score * 0.5
                                return final
                            elif opt_objective == "CAGR 최대화":
                                return is_cagr if not opt_use_oos else (is_cagr * 0.5 + oos_cagr * 0.5)
                            elif opt_objective == "MDD 최소화":
                                return -abs(is_mdd)
                            else:
                                return is_sharpe

                        # 반복 실행
                        all_studies = []
                        for rep in range(opt_repeat):
                            if opt_repeat > 1: status_box.caption(f"🔁 반복 {rep+1}/{opt_repeat}")
                            trial_counter['n'] = 0
                            study = optuna.create_study(direction='maximize', study_name=f'soxl_opt_r{rep}')
                            study.optimize(optuna_objective, n_trials=opt_trials, show_progress_bar=False)
                            all_studies.append(study)
                        st.session_state.optuna_study = all_studies[-1]

                        # 결과 수집 (전체 반복 통합)
                        rows = []
                        for study in all_studies:
                            for t in study.trials:
                                if t.state == optuna.trial.TrialState.COMPLETE:
                                    row = {**t.params}
                                    for k in ['IS_CAGR','IS_MDD','IS_Sharpe','OOS_CAGR','OOS_Score','CV_Score','WinRate','Stability']:
                                        row[k] = t.user_attrs.get(k, 0)
                                    row['Score'] = round(t.value, 2)
                                    rows.append(row)
                        if rows:
                            st.session_state.optuna_results = pd.DataFrame(rows).sort_values('Score', ascending=False)
                        status_box.success(f"✅ 완료! {len(rows)}개 유효 시도 (반복 {opt_repeat}회)")

                    # 결과 표시
                    if st.session_state.optuna_results is not None and not st.session_state.optuna_results.empty:
                        results_df = st.session_state.optuna_results

                        st.markdown("### 🏆 Optuna TOP 10")
                        priority_cols = ['Score','IS_CAGR','IS_MDD','IS_Sharpe','OOS_CAGR','OOS_Score','CV_Score','Stability','WinRate',
                                         'bt_cond','cl_cond','bt_buy','bt_prof','bt_time','md_buy','md_prof','md_time','cl_buy','cl_prof','cl_time']
                        if 'panic_buy' in results_df.columns:
                            priority_cols += ['panic_buy','panic_prof','panic_time','bear_buy','bear_prof','bear_time']
                        avail_cols = [c for c in priority_cols if c in results_df.columns]
                        st.dataframe(results_df[avail_cols].head(10), use_container_width=True)

                        best_row = results_df.iloc[0]
                        with st.container(border=True):
                            m1, m2, m3, m4, m5, m6 = st.columns(6)
                            m1.metric("🏅 Score", f"{best_row.get('Score',0):.2f}")
                            m2.metric("IS CAGR", f"{best_row.get('IS_CAGR', best_row.get('CAGR',0)):.1f}%")
                            m3.metric("IS MDD", f"{best_row.get('IS_MDD', best_row.get('MDD',0)):.1f}%")
                            m4.metric("OOS CAGR", f"{best_row.get('OOS_CAGR',0):.1f}%")
                            m5.metric("안정성", f"{best_row.get('Stability',0):.2f}")
                            m6.metric("Sharpe", f"{best_row.get('IS_Sharpe',0):.2f}")

                        # 상위10 평균 ± std
                        top10_opt = results_df.head(10)
                        num_c = [c for c in top10_opt.select_dtypes(include=[np.number]).columns if c not in ['Score','IS_CAGR','IS_MDD','IS_Sharpe','OOS_CAGR','OOS_Score','CV_Score','Stability','WinRate']]
                        if num_c:
                            st.markdown("#### 📊 상위10 파라미터 통계")
                            pstats = pd.DataFrame({'Mean': top10_opt[num_c].mean().round(3), 'Std': top10_opt[num_c].std().round(3)})
                            pstats['CV%'] = (pstats['Std'] / pstats['Mean'].abs().replace(0, 1) * 100).round(1)
                            st.dataframe(pstats, use_container_width=True)
                            low_cv = (pstats['CV%'] < 30).sum()
                            st.caption(f"✅ CV% < 30인 파라미터: {low_cv}/{len(num_c)} (높을수록 안정적)")

                        # 파라미터 중요도
                        if st.session_state.optuna_study:
                            try:
                                importance = optuna.importance.get_param_importances(st.session_state.optuna_study)
                                if importance:
                                    imp_df = pd.DataFrame({'파라미터': list(importance.keys()), '중요도': list(importance.values())}).sort_values('중요도', ascending=True)
                                    st.markdown("### 📊 파라미터 중요도 (fANOVA)")
                                    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(imp_df)*0.35)))
                                    ax2.barh(imp_df['파라미터'], imp_df['중요도'], color='#4ECDC4')
                                    ax2.set_xlabel('Importance')
                                    plt.tight_layout()
                                    st.pyplot(fig2, use_container_width=True)
                            except: pass

                        # IS vs OOS 산점도
                        st.markdown("### 🎯 IS vs OOS 분포")
                        fig3, ax3 = plt.subplots(figsize=(8, 5))
                        sc3 = ax3.scatter(results_df['IS_MDD'], results_df['IS_CAGR'], c=results_df['Score'], cmap='viridis', alpha=0.5, label='IS')
                        if 'OOS_CAGR' in results_df.columns and results_df['OOS_CAGR'].abs().sum() > 0:
                            ax3.scatter(results_df.get('IS_MDD', results_df['IS_MDD']), results_df['OOS_CAGR'], marker='x', c='red', alpha=0.3, s=20, label='OOS')
                        ax3.scatter([best_row['IS_MDD']], [best_row['IS_CAGR']], c='gold', s=200, marker='*', zorder=5, label='Best')
                        ax3.set_xlabel('MDD (%)'); ax3.set_ylabel('CAGR (%)')
                        ax3.set_title('Optuna: IS vs OOS Risk-Return'); ax3.legend()
                        plt.colorbar(sc3, label='Score')
                        st.pyplot(fig3, use_container_width=True)


else:
    st.markdown("")
    st.info("👈 **사이드바**의 📡 데이터 연동에서 구글 시트 주소를 입력해주세요.")
