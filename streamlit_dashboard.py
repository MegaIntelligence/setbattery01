# streamlit_dashboard.py
import streamlit as st
from generate_and_analyze import (
    NUM_PACKS, NUM_CELLS, CELL_COL_NAMES,
    INITIAL_DESIGN_LIFE, INITIAL_CAPACITY, INITIAL_EFFICIENCY,
    INITIAL_SOC, INITIAL_V_STATIC, INITIAL_V_DYNAMIC, INITIAL_IR, INITIAL_TEMP,
    generate_pack_data, parse_data, calculate_std_dev, 
    calculate_health_scores, get_status
)
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

# --- Data Generation Function (Cached for Streamlit) ---
@st.cache_data # Cache the generated data within the Streamlit app session
def generate_all_pack_data_st():
    """Generates and returns a DataFrame for all packs for Streamlit."""
    # Use st.spinner for feedback during generation if it takes time
    with st.spinner("🔄 Generating simulation data... (This happens only once per session or when code changes)"):
        all_data_rows = []
        header = ['Pack_ID', 'Data_Type', 'Parameter', 'Unit', 'Value_Initial', 'Value_Present'] + CELL_COL_NAMES
        all_data_rows.append(header)
        for i in range(1, NUM_PACKS + 1):
            all_data_rows.extend(generate_pack_data(f'Pack_{i:02d}'))
        # Convert to DataFrame
        df = pd.DataFrame(all_data_rows[1:], columns=all_data_rows[0])
        
        # Convert 'NA' strings to np.nan for numeric columns
        numeric_cols = ['Value_Initial', 'Value_Present'] + CELL_COL_NAMES
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace('NA', np.nan).infer_objects(copy=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
    return df

# --- Analysis Functions (Adapted from Colab/Script) ---

def parse_data_st(df_single_pack):
    """Parses the DataFrame FOR A SINGLE PACK to extract relevant values for Streamlit."""
    if df_single_pack.empty:
        st.error("❌ Empty data frame provided")
        return None
        
    try:
        pack_id = df_single_pack['Pack_ID'].iloc[0] if 'Pack_ID' in df_single_pack.columns else "Unknown"
        data = parse_data(df_single_pack)
        if data is None:
            st.error(f"❌ Error parsing data for Pack {pack_id}")
            return None
            
        # Validate required keys exist
        required_keys = ['residual_life', 'present_cap', 'present_eff']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            st.error(f"❌ Missing required data fields: {', '.join(missing_keys)}")
            return None
            
        # Additional Streamlit-specific validation
        expected_len = NUM_CELLS
        incomplete_metrics = []
        for key, lst in data.items():
            if '_cells' in key:
                if not isinstance(lst, list):
                    st.error(f"❌ Invalid data type for {key} - expected list")
                    return None
                if len(lst) != expected_len:
                    incomplete_metrics.append(f"{key} (found {len(lst)})")
                
        if incomplete_metrics:
            st.warning(f"⚠️ Warning (Pack {pack_id}): Incomplete cell data for: {', '.join(incomplete_metrics)}. Expected {expected_len} cells.")
            # Continue processing but with potentially incomplete data
            
        return data
        
    except Exception as e:
        st.error(f"❌ Error parsing data for Pack {pack_id}: {str(e)}")
        return None

# --- Visualization Functions (for Streamlit) ---

def create_radar_chart_st(scores, pack_id):
    categories = ['電芯平衡度', '電流一致性', '操作區間', 'SOC一致性', '轉換效率', '電壓一致性']
    values = [
        scores.get('Score_V_Static_Consistency', 0), scores.get('Score_IR_Consistency', 0),
        scores.get('Score_Capacity', 0), scores.get('Score_SoC_Consistency', 0),
        scores.get('Score_Efficiency', 0), scores.get('Score_V_Dynamic_Consistency', 0)
    ]
    values = [round(v, 1) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar( r=[100] * len(categories), theta=categories, mode='lines', line_color='lightgreen', line_width=8, fill='toself', fillcolor='rgba(144, 238, 144, 0.1)', hoverinfo='skip', name='Excellent'))
    fig.add_trace(go.Scatterpolar( r=[75] * len(categories), theta=categories, mode='lines', line_color='yellow', line_width=1, hoverinfo='skip', name='Caution'))
    fig.add_trace(go.Scatterpolar( r=values, theta=categories, fill='toself', name='目前狀態', line_color='red', text=[f"{v:.1f}" for v in values], hoverinfo='text+theta'))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[0, 25, 50, 75, 100], showline=False, gridcolor='rgba(255,255,255,0.3)'),
            angularaxis=dict(tickfont=dict(size=14, color='white'), linecolor='grey', gridcolor='rgba(255,255,255,0.3)')
        ),
        showlegend=False,
        title=dict(text=f"電池組健康狀態 ({pack_id})", font=dict(size=18, color='white')),
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(size=12, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.8)',
        # template="plotly_dark" # Alternative
    )
    # Add labels near points (optional, can be cluttered)
    # angles_rad = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    # for i, val in enumerate(values):
    #      fig.add_annotation(
    #          # Polar coordinates for annotation positioning need careful handling
    #          # This is a simple attempt, might need adjustment
    #          ax=0, ay=0, # Anchor point
    #          x=np.cos(angles_rad[i]) * val * 1.1, # Position based on value
    #          y=np.sin(angles_rad[i]) * val * 1.1,
    #          text=f"<b>{val:.1f}</b>",
    #          showarrow=False,
    #          font=dict(size=11, color="white")
    #      )

    return fig


def display_status_box_streamlit(grade, status_text, color, icon):
    """Displays a status box using Streamlit markdown."""
    border_color = color if color != "orange" else "#FFA500" # Hex for orange
    st.markdown(f"""
    <div style="
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
        background-color: #262730; /* Match Streamlit dark theme */
        color: white;
    ">
        <span style="font-size: 2em; margin-right: 15px; vertical-align: middle;">{icon}</span>
        <span style="font-size: 1.1em; font-weight: bold; vertical-align: middle;">Grade: {grade:.1f}</span><br>
        <span style="color: {border_color}; font-weight: bold; font-size: 1.1em; margin-left: 50px;">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Streamlit App Main Logic ---

st.set_page_config(layout="wide", page_title="電池健康狀態分析儀")

st.title("🔋 電池組健康狀態分析儀")
st.markdown("依序顯示模擬電池組數據與分析結果。")

# --- Load or Generate Data ---
# This uses the cached function specifically for Streamlit
import os
import sys

def get_base_path():
    """Get the base path for data files (works for both script and executable)"""
    if getattr(sys, 'frozen', False):
        # Running as executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

# Ensure data files are in the same directory as executable
data_dir = get_base_path()
os.chdir(data_dir)

df_all_packs = generate_all_pack_data_st()
pack_ids = df_all_packs['Pack_ID'].unique()

# --- Initialize Session State ---
if 'current_pack_index' not in st.session_state:
    st.session_state.current_pack_index = 0

# --- Ensure index is valid ---
# This check might be needed if the number of packs changes, though unlikely here
if st.session_state.current_pack_index >= len(pack_ids):
    st.session_state.current_pack_index = 0 # Reset

# --- Get Current Pack Data ---
current_pack_id = pack_ids[st.session_state.current_pack_index]
df_current_pack = df_all_packs[df_all_packs['Pack_ID'] == current_pack_id].copy()

# --- UI Layout ---
with st.container():
    # Navigation and pack info at top
    st.header(f"📊 電池組數據: {current_pack_id}")
    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        if st.button("⬅️ 上一筆", use_container_width=True):
            st.session_state.current_pack_index = (st.session_state.current_pack_index - 1) % len(pack_ids)
            st.rerun()
    with nav_col2:
        if st.button("➡️ 下一筆", use_container_width=True):
            st.session_state.current_pack_index = (st.session_state.current_pack_index + 1) % len(pack_ids)
            st.rerun()

# Main content in tabs
tab1, tab2 = st.tabs(["📈 主要指標", "🔍 詳細分析"])

with tab1:
    # Key metrics in a grid
    st.subheader("電池健康狀態概覽")
    parsed_data_display = parse_data_st(df_current_pack)
    if parsed_data_display:
        # Calculate health scores first
        scores, raw_metrics = calculate_health_scores(parsed_data_display)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("剩餘壽命", f"{parsed_data_display.get('residual_life', 'N/A'):.1f} 年", 
                   help="電池組剩餘使用年限")
        col2.metric("可用容量", f"{parsed_data_display.get('present_cap', 'N/A'):.1f} Ah",
                   help="當前可用容量")
        col3.metric("能量效率", f"{parsed_data_display.get('present_eff', 'N/A'):.1f} %",
                   help="能量轉換效率")
        
        # Radar chart visualization
        st.plotly_chart(create_radar_chart_st(scores, current_pack_id), use_container_width=True)
        
        # Overall status
        overall_grade = scores.get('Overall_Grade', 0)
        status_text, icon = get_status(overall_grade)
        color = "green" if overall_grade >= 90 else "orange" if overall_grade >= 75 else "red"
        display_status_box_streamlit(overall_grade, status_text, color, icon)

with tab2:
    # Detailed analysis
    st.subheader("詳細電池數據")
    with st.expander("📋 完整數據表格"):
        st.dataframe(df_current_pack, use_container_width=True, height=300)
    
    st.subheader("電芯一致性分析")
    cols = st.columns(2)
    with cols[0]:
        st.metric("SoC 標準差", f"{calculate_std_dev(parsed_data_display.get('soc_present_cells',[])):.3f} %")
        st.metric("靜態電壓標準差", f"{calculate_std_dev(parsed_data_display.get('v_static_present_cells',[])):.4f} V")
    with cols[1]:
        st.metric("動態電壓標準差", f"{calculate_std_dev(parsed_data_display.get('v_dynamic_present_cells',[])):.4f} V")
        st.metric("內阻標準差", f"{calculate_std_dev(parsed_data_display.get('ir_present_cells',[])):.3f} mΩ")

st.divider()
st.caption(f"目前顯示第 {st.session_state.current_pack_index + 1} 筆資料，共 {len(pack_ids)} 筆。")
