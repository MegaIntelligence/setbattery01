import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import csv
import time
import os
import webbrowser

# --- Configuration & Constants (from Cells 3 & 5) ---
NUM_PACKS = 10
NUM_CELLS = 10
CELL_COL_NAMES = [f'Cell_{i}' for i in range(1, NUM_CELLS + 1)]
OUTPUT_FILENAME = 'multi_battery_sim_data.csv' # Define CSV filename

# --- Initial State (Consistent across all simulated new packs) ---
INITIAL_DESIGN_LIFE = 5.0
INITIAL_CAPACITY = 100.0
INITIAL_EFFICIENCY = 95.0
INITIAL_SOC = 80.0
INITIAL_V_STATIC = 3.90
INITIAL_V_DYNAMIC = 3.75
INITIAL_IR = 2.0
INITIAL_TEMP = 25.0

# --- Data Generation Function (from Cell 3) ---
def generate_pack_data(pack_id_str):
    """Generates simulated data rows for a single pack."""
    pack_rows = []
    pack_id_num = int(pack_id_str.split('_')[1]) # Extract the numeric part

    # Simulate degradation
    degradation_factor = (pack_id_num / NUM_PACKS) * 0.8 + np.random.uniform(-0.1, 0.1)
    degradation_factor = max(0, min(1, degradation_factor))

    # Simulate present pack values
    present_residual_life = INITIAL_DESIGN_LIFE * (1 - degradation_factor * np.random.uniform(0.8, 1.2))
    present_capacity = INITIAL_CAPACITY * (1 - degradation_factor * np.random.uniform(0.1, 0.4))
    present_efficiency = INITIAL_EFFICIENCY * (1 - degradation_factor * np.random.uniform(0.05, 0.15))

    # Clamp values
    present_residual_life = max(0.5, present_residual_life)
    present_capacity = max(INITIAL_CAPACITY * 0.5, present_capacity)
    present_efficiency = max(70.0, present_efficiency)

    # Simulate present cell means and std devs
    mean_soc_present = INITIAL_SOC * (1 - degradation_factor * 0.1)
    std_soc_present = degradation_factor * 1.5 + np.random.uniform(0.1, 0.5)
    mean_v_static_present = INITIAL_V_STATIC * (1 - degradation_factor * 0.03)
    std_v_static_present = degradation_factor * 0.02 + np.random.uniform(0.005, 0.01)
    mean_v_dynamic_present = INITIAL_V_DYNAMIC * (1 - degradation_factor * 0.05)
    std_v_dynamic_present = degradation_factor * 0.03 + np.random.uniform(0.01, 0.02)
    mean_ir_present = INITIAL_IR * (1 + degradation_factor * 0.8)
    std_ir_present = degradation_factor * 0.5 + np.random.uniform(0.1, 0.3)
    mean_temp_present = INITIAL_TEMP + degradation_factor * 8
    std_temp_present = degradation_factor * 2.0 + np.random.uniform(0.2, 1.0)

    # Generate cell values
    soc_present_cells = np.clip(np.random.normal(mean_soc_present, std_soc_present, NUM_CELLS), 0, 100).round(1)
    v_static_present_cells = np.clip(np.random.normal(mean_v_static_present, std_v_static_present, NUM_CELLS), 3.0, 4.2).round(2)
    v_dynamic_present_cells = np.clip(np.random.normal(mean_v_dynamic_present, std_v_dynamic_present, NUM_CELLS), 2.8, 4.0).round(2)
    ir_present_cells = np.clip(np.random.normal(mean_ir_present, std_ir_present, NUM_CELLS), 1.0, 10.0).round(1)
    temp_present_cells = np.clip(np.random.normal(mean_temp_present, std_temp_present, NUM_CELLS), 10, 60).round(1)

    # Create data rows
    na_cells = ['NA'] * NUM_CELLS
    pack_rows.append([pack_id_str, 'Pack_Info', 'Designed_Calendar_Life', 'Years', INITIAL_DESIGN_LIFE, 'NA'] + na_cells)
    pack_rows.append([pack_id_str, 'Pack_Info', 'Residual_Calendar_Life', 'Years', INITIAL_DESIGN_LIFE, f"{present_residual_life:.1f}"] + na_cells)
    pack_rows.append([pack_id_str, 'Pack_Info', 'Available_Capacity', 'Ah', INITIAL_CAPACITY, f"{present_capacity:.1f}"] + na_cells)
    pack_rows.append([pack_id_str, 'Pack_Info', 'Energy_Efficiency', 'Percent', INITIAL_EFFICIENCY, f"{present_efficiency:.1f}"] + na_cells)
    pack_rows.append([pack_id_str, 'Cell_Data', 'SoC_Initial', 'Percent', 'NA', 'NA'] + [INITIAL_SOC] * NUM_CELLS)
    pack_rows.append([pack_id_str, 'Cell_Data', 'V_Static_Initial', 'V', 'NA', 'NA'] + [INITIAL_V_STATIC] * NUM_CELLS)
    pack_rows.append([pack_id_str, 'Cell_Data', 'V_Dynamic_Initial', 'V', 'NA', 'NA'] + [INITIAL_V_DYNAMIC] * NUM_CELLS)
    pack_rows.append([pack_id_str, 'Cell_Data', 'IR_Initial', 'mOhm', 'NA', 'NA'] + [INITIAL_IR] * NUM_CELLS)
    pack_rows.append([pack_id_str, 'Cell_Data', 'Temp_Initial', 'C', 'NA', 'NA'] + [INITIAL_TEMP] * NUM_CELLS)
    pack_rows.append([pack_id_str, 'Cell_Data', 'SoC_Present', 'Percent', 'NA', 'NA'] + list(soc_present_cells))
    pack_rows.append([pack_id_str, 'Cell_Data', 'V_Static_Present', 'V', 'NA', 'NA'] + list(v_static_present_cells))
    pack_rows.append([pack_id_str, 'Cell_Data', 'V_Dynamic_Present', 'V', 'NA', 'NA'] + list(v_dynamic_present_cells))
    pack_rows.append([pack_id_str, 'Cell_Data', 'IR_Present', 'mOhm', 'NA', 'NA'] + list(ir_present_cells))
    pack_rows.append([pack_id_str, 'Cell_Data', 'Temp_Present', 'C', 'NA', 'NA'] + list(temp_present_cells))
    return pack_rows

# --- Analysis Functions (from Cell 5) ---

def parse_data(df_single_pack):
    """Parses the DataFrame FOR A SINGLE PACK to extract relevant values."""
    data = {}
    pack_id = "Unknown" # Default
    try:
        pack_id = df_single_pack['Pack_ID'].iloc[0]
        pack_info = df_single_pack[df_single_pack['Data_Type'] == 'Pack_Info'].set_index('Parameter')
        data['design_life'] = float(pack_info.loc['Designed_Calendar_Life', 'Value_Initial'])
        data['residual_life'] = float(pack_info.loc['Residual_Calendar_Life', 'Value_Present'])
        data['initial_cap'] = float(pack_info.loc['Available_Capacity', 'Value_Initial'])
        data['present_cap'] = float(pack_info.loc['Available_Capacity', 'Value_Present'])
        data['initial_eff'] = float(pack_info.loc['Energy_Efficiency', 'Value_Initial'])
        data['present_eff'] = float(pack_info.loc['Energy_Efficiency', 'Value_Present'])

        cell_data_raw = df_single_pack[df_single_pack['Data_Type'] == 'Cell_Data'].set_index('Parameter')
        cell_data_numeric = cell_data_raw[CELL_COL_NAMES].apply(pd.to_numeric, errors='coerce')

        data['soc_initial_cells'] = cell_data_numeric.loc['SoC_Initial'].dropna().tolist()
        data['v_static_initial_cells'] = cell_data_numeric.loc['V_Static_Initial'].dropna().tolist()
        data['v_dynamic_initial_cells'] = cell_data_numeric.loc['V_Dynamic_Initial'].dropna().tolist()
        data['ir_initial_cells'] = cell_data_numeric.loc['IR_Initial'].dropna().tolist()
        data['temp_initial_cells'] = cell_data_numeric.loc['Temp_Initial'].dropna().tolist()

        data['soc_present_cells'] = cell_data_numeric.loc['SoC_Present'].dropna().tolist()
        data['v_static_present_cells'] = cell_data_numeric.loc['V_Static_Present'].dropna().tolist()
        data['v_dynamic_present_cells'] = cell_data_numeric.loc['V_Dynamic_Present'].dropna().tolist()
        data['ir_present_cells'] = cell_data_numeric.loc['IR_Present'].dropna().tolist()
        data['temp_present_cells'] = cell_data_numeric.loc['Temp_Present'].dropna().tolist()

        # Basic validation in terminal
        valid_lengths = all(len(lst) == NUM_CELLS for key, lst in data.items() if '_cells' in key and isinstance(lst, list) and lst)
        if not valid_lengths:
             print(f"⚠️ Warning (Pack {pack_id}): Some cell data might be missing or incomplete.")

    except KeyError as e:
        print(f"❌ Error parsing CSV (Pack {pack_id}): Missing parameter - {e}.")
        return None
    except ValueError as e:
        print(f"❌ Error parsing CSV (Pack {pack_id}): Conversion error - {e}.")
        return None
    except Exception as e:
        print(f"❌ Unexpected parsing error (Pack {pack_id}): {e}")
        return None
    return data

def calculate_std_dev(values):
    """Calculates the sample standard deviation."""
    if not isinstance(values, list) or len(values) < 2: return 0.0
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if len(numeric_values) < 2: return 0.0
    return np.std(numeric_values, ddof=1)

def calculate_health_scores(data):
    """Calculates health scores (0-100, higher is better) based on parsed data."""
    scores = {}
    raw_metrics = {}
    if not data: return scores, raw_metrics

    if data.get('initial_cap', 0) > 0: scores['Score_Capacity'] = max(0.0, min(100.0, (data.get('present_cap', 0) / data['initial_cap']) * 100))
    else: scores['Score_Capacity'] = 0.0
    raw_metrics['Capacity_SOH'] = scores.get('Score_Capacity', 0)
    scores['Score_Efficiency'] = max(0.0, min(100.0, data.get('present_eff', 0)))
    raw_metrics['Present_Efficiency'] = data.get('present_eff', 0)

    soc_std_present = calculate_std_dev(data.get('soc_present_cells', [])); raw_metrics['SoC_StdDev_Present'] = soc_std_present; scores['Score_SoC_Consistency'] = max(0.0, 100.0 - 20 * soc_std_present)
    v_static_std_present = calculate_std_dev(data.get('v_static_present_cells', [])); raw_metrics['V_Static_StdDev_Present'] = v_static_std_present; scores['Score_V_Static_Consistency'] = max(0.0, 100.0 - 1000 * v_static_std_present)
    v_dynamic_std_present = calculate_std_dev(data.get('v_dynamic_present_cells', [])); raw_metrics['V_Dynamic_StdDev_Present'] = v_dynamic_std_present; scores['Score_V_Dynamic_Consistency'] = max(0.0, 100.0 - 1000 * v_dynamic_std_present)
    ir_std_present = calculate_std_dev(data.get('ir_present_cells', [])); raw_metrics['IR_StdDev_Present'] = ir_std_present; scores['Score_IR_Consistency'] = max(0.0, 100.0 - 50 * ir_std_present)
    temp_std_present = calculate_std_dev(data.get('temp_present_cells', [])); raw_metrics['Temp_StdDev_Present'] = temp_std_present; scores['Score_Temp_Consistency'] = max(0.0, 100.0 - 10 * temp_std_present)

    radar_scores_keys = ['Score_Capacity', 'Score_SoC_Consistency', 'Score_Efficiency', 'Score_V_Dynamic_Consistency', 'Score_V_Static_Consistency', 'Score_IR_Consistency']
    radar_scores = [scores.get(key, 0) for key in radar_scores_keys]; scores['Overall_Grade'] = min(radar_scores) if radar_scores else 0.0
    return scores, raw_metrics

def get_status(grade):
    """Determines the status text and color based on the grade."""
    # Using text only for terminal output
    if grade >= 90: return "良好", "✅"
    elif grade >= 75: return "須注意", "⚠️"
    else: return "須檢修", "❌"

# --- Visualization Function (Modified for Saving) ---

def create_and_save_radar_chart(scores, pack_id):
    """Creates a Plotly radar chart and saves it as an HTML file."""
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

    status_text, icon = get_status(scores.get('Overall_Grade', 0))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[0, 25, 50, 75, 100], showline=False, gridcolor='rgba(128,128,128,0.5)'),
            angularaxis=dict(tickfont_size=14, linecolor='grey', gridcolor='rgba(128,128,128,0.5)')
        ),
        showlegend=False,
        title=dict(text=f"電池組健康狀態 ({pack_id}) {icon} {status_text}", font=dict(size=18)),
        margin=dict(l=60, r=60, t=80, b=40),
        font=dict(size=12),
        # Using default theme which works well for HTML files
        # template="plotly_dark" # Can uncomment if preferred
    )

    # --- Save the figure ---
    chart_filename = f"radar_chart_{pack_id}.html"
    try:
        fig.write_html(chart_filename)
        print(f"📈 Radar chart saved as: {chart_filename}")
        # Optionally save as image (requires kaleido):
        # image_filename = f"radar_chart_{pack_id}.png"
        # fig.write_image(image_filename)
        # print(f"📈 Radar chart image saved as: {image_filename}")
    except Exception as e:
        print(f"❌ Error saving chart for {pack_id}: {e}")
        print("   Ensure 'kaleido' is installed if saving images (`pip install kaleido`).")



# --- Main Execution Logic ---
if __name__ == "__main__":
    # == Part 1: Generate Data (like Cell 3) ==
    print("--- Starting Data Generation ---")
    all_data_rows = []
    header = ['Pack_ID', 'Data_Type', 'Parameter', 'Unit', 'Value_Initial', 'Value_Present'] + CELL_COL_NAMES
    all_data_rows.append(header)

    generation_start_time = time.time()
    for i in range(1, NUM_PACKS + 1):
        pack_id_str = f'Pack_{i:02d}'
        all_data_rows.extend(generate_pack_data(pack_id_str))
        print(f"   Generated data for {pack_id_str}")

    # Save to CSV
    try:
        with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_data_rows)
        generation_end_time = time.time()
        print(f"\n✅ Generated simulation data for {NUM_PACKS} packs in '{OUTPUT_FILENAME}'")
        print(f"   Generation took {generation_end_time - generation_start_time:.2f} seconds.")
        # Optional: Display head of generated file in terminal
        try:
            df_generated = pd.read_csv(OUTPUT_FILENAME)
            print("\n--- Preview of Generated CSV ---")
            print(df_generated.head(15).to_string()) # Use to_string for better terminal view
        except Exception as e:
            print(f"\n⚠️ Could not read or display preview of generated CSV: {e}")

    except Exception as e:
        print(f"\n❌ Error saving data to CSV '{OUTPUT_FILENAME}': {e}")
        exit() # Stop if data generation failed

    # == Part 2: Analyze Data (like Cell 5) ==
    print("\n--- Starting Data Analysis ---")

    # Check if the file exists before attempting to read
    if not os.path.exists(OUTPUT_FILENAME):
        print(f"❌ Error: File '{OUTPUT_FILENAME}' not found.")
        print("   Cannot proceed with analysis.")
    else:
        analysis_start_time = time.time()
        try:
            df_all_packs = pd.read_csv(OUTPUT_FILENAME)
            print(f"✅ File '{OUTPUT_FILENAME}' read successfully for analysis.")

            if 'Pack_ID' not in df_all_packs.columns:
                 raise ValueError("CSV missing the required 'Pack_ID' column.")
            pack_ids = df_all_packs['Pack_ID'].unique()
            print(f"Found {len(pack_ids)} Pack IDs: {list(pack_ids)}")

            # Loop through each Pack ID
            for pack_id in pack_ids:
                print(f"\n--- Analyzing Pack ID: {pack_id} ---")
                df_current_pack = df_all_packs[df_all_packs['Pack_ID'] == pack_id].copy()
                if df_current_pack.empty:
                    print(f"🤷 No data found for Pack ID: {pack_id}. Skipping.")
                    continue

                parsed_data = parse_data(df_current_pack)
                if parsed_data:
                    scores, raw_metrics = calculate_health_scores(parsed_data)

                    # Display Results in Terminal
                    print("\n📊 Detailed Metrics:")
                    print(f"- Capacity SOH (%):              {raw_metrics.get('Capacity_SOH', 0):.1f}")
                    print(f"- Present Efficiency (%):        {raw_metrics.get('Present_Efficiency', 0):.1f}")
                    print(f"- SoC Std Dev Present (%):       {raw_metrics.get('SoC_StdDev_Present', 0):.3f}")
                    print(f"- V Static Std Dev Present (V):  {raw_metrics.get('V_Static_StdDev_Present', 0):.4f}")
                    print(f"- V Dynamic Std Dev Present (V): {raw_metrics.get('V_Dynamic_StdDev_Present', 0):.4f}")
                    print(f"- IR Std Dev Present (mOhm):     {raw_metrics.get('IR_StdDev_Present', 0):.3f}")
                    print(f"- Temp Std Dev Present (C):      {raw_metrics.get('Temp_StdDev_Present', 0):.2f}")

                    # Create and Save Radar Chart
                    create_and_save_radar_chart(scores, pack_id)

                    # Print Overall Grade
                    print("\n⭐ Overall Grade:")
                    overall_grade = scores.get('Overall_Grade', 0)
                    status_text, icon = get_status(overall_grade)
                    print(f"   Grade: {overall_grade:.1f} ({icon} {status_text})")
                    # time.sleep(0.1) # Optional small delay if output is too fast

                else:
                    print(f"\n❌ Could not calculate scores for Pack ID {pack_id} due to parsing errors.")

            analysis_end_time = time.time()
            print(f"\n--- Finished processing {len(pack_ids)} packs in {analysis_end_time - analysis_start_time:.2f} seconds ---")

        except pd.errors.EmptyDataError:
            print(f"❌ Error: File '{OUTPUT_FILENAME}' is empty.")
        except UnicodeDecodeError:
             print(f"❌ Error: Could not decode file '{OUTPUT_FILENAME}'. Ensure it is UTF-8 encoded.")
        except ValueError as e:
             print(f"❌ Value Error during analysis: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred during analysis: {e}")

    print("\n--- Script execution finished. ---")
