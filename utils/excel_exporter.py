import pandas as pd
import xlsxwriter

def generate_hex_colors(names):
    palette = [
        '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', 
        '#911EB4', '#46FBEB', '#F032E6', '#BCF60C', '#FABEBE', 
        '#008080', '#E6BEFF', '#9A6324', '#FFFAC8', '#800000'
    ]
    color_map = {}
    for i, name in enumerate(names):
        color_map[name] = palette[i % len(palette)]
    return color_map

def export_solution_to_excel(solution, input_data, filename="Line_Schedule.xlsx"):
    # 1. Khởi tạo dữ liệu cơ bản
    dates = sorted(list(input_data.set['setT']))
    all_styles = sorted(list(input_data.set['setS']))
    lines = sorted(list(input_data.set['setL']))
    
    # Map tiêu đề ngày
    if 'real_dates' in input_data.set and len(input_data.set['real_dates']) == len(dates):
        date_headers = [d.strftime("%b %d") for d in input_data.set['real_dates']]
    else:
        date_headers = [f"T{t}" for t in dates]
    
    col_names = [f"T{t}" for t in dates]
    style_colors = generate_hex_colors(all_styles)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- SHEET 1: LINE-SCHEDULE (TỔNG HỢP) ---
        rows_main = []
        for line in lines:
            for r_type in ['Style', 'Qty', 'Eff', 'Exp', 'MaxEff']:
                row_data = {'Line': line, 'Type': r_type}
                for t in dates:
                    col_name = f"T{t}"
                    if r_type == 'Style':
                        val = solution['assignment'].get((line, t), "")
                    elif r_type == 'Qty':
                        style = solution['assignment'].get((line, t))
                        val = solution['production'].get((line, style, t), 0) if style else 0
                    elif r_type == 'Eff':
                        val = solution['efficiency'].get((line, t), 0)
                    elif r_type == 'Exp':
                        val = solution['experience'].get((line, t), 0)
                    else: # MaxEff (Tạm lấy bằng Eff hoặc giá trị tối đa tùy logic)
                        val = solution['efficiency'].get((line, t), 0)
                    row_data[col_name] = val
                rows_main.append(row_data)

        df_main = pd.DataFrame(rows_main)
        df_main.to_excel(writer, sheet_name='Line-Schedule', index=False)
        
        # Format Sheet chính (như code cũ của bạn)
        ws_main = writer.sheets['Line-Schedule']
        header_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D3D3D3', 'border': 1})
        center_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
        num_fmt = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '#,##0'})
        pct_fmt = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '0%'})

        for col_num, value in enumerate(df_main.columns.values):
            ws_main.write(0, col_num, value, header_fmt)

        # Merge Line cells và tô màu Style
        style_formats = {s: workbook.add_format({'bg_color': c, 'font_color': 'white', 'bold': 1, 'border': 1, 'align': 'center'}) 
                         for s, c in style_colors.items()}

        for i in range(0, len(df_main), 5):
            ws_main.merge_range(i+1, 0, i+5, 0, df_main.iloc[i]['Line'], center_fmt)
            for t_idx, t in enumerate(dates):
                s_val = df_main.iloc[i, t_idx + 2]
                fmt = style_formats.get(s_val, center_fmt) if s_val else center_fmt
                ws_main.write(i+1, t_idx + 2, s_val, fmt) # Style
                ws_main.write(i+2, t_idx + 2, df_main.iloc[i+1, t_idx + 2], num_fmt) # Qty
                ws_main.write(i+3, t_idx + 2, df_main.iloc[i+2, t_idx + 2], pct_fmt) # Eff

        # --- SHEET TIẾP THEO: MỖI STYLE MỘT SHEET ---
        for style in all_styles:
            style_rows = []
            
            # Khởi tạo giá trị tồn kho/backlog ban đầu từ input_data
            inv_fab = input_data.param.get('paramI0fabric', {}).get(style, 0)
            inv_fg = input_data.param.get('paramI0product', {}).get(style, 0)
            backlog = input_data.param.get('paramB0', {}).get(style, 0)

            # Dictionary lưu trữ data cho từng hàng
            data_map = {m: {} for m in [
                'Demand', 'Fabric Receiving', 'Beg. Inv Fabric', 'Producing', 
                'End. Inv Fabric', 'Beg. Inv FG', 'Shipping', 'End. Inv FG', 'Backlog'
            ]}

            for t in dates:
                # 1. Lấy thông số cơ bản
                demand_t = input_data.param.get('paramD', {}).get((style, t), 0)
                fab_recv_t = input_data.param.get('paramF', {}).get((style, t), 0)
                prod_t = sum(solution['production'].get((l, style, t), 0) for l in lines)
                
                # 2. Tính toán cân bằng (Inventory Balance)
                # Fabric
                data_map['Beg. Inv Fabric'][t] = inv_fab
                inv_fab = inv_fab + fab_recv_t - prod_t
                data_map['End. Inv Fabric'][t] = inv_fab
                
                # FG & Shipping & Backlog
                data_map['Beg. Inv FG'][t] = inv_fg
                total_available = inv_fg + prod_t
                total_needed = demand_t + backlog
                
                ship_t = min(total_available, total_needed)
                inv_fg = total_available - ship_t
                backlog = total_needed - ship_t
                
                data_map['Demand'][t] = demand_t
                data_map['Fabric Receiving'][t] = fab_recv_t
                data_map['Producing'][t] = prod_t
                data_map['Shipping'][t] = ship_t
                data_map['End. Inv FG'][t] = inv_fg
                data_map['Backlog'][t] = backlog

            # Chuyển data_map thành danh sách hàng cho DataFrame
            for metric, vals in data_map.items():
                row = {'Metric': metric}
                for t in dates:
                    row[f"T{t}"] = vals[t]
                style_rows.append(row)

            df_style = pd.DataFrame(style_rows)
            # Tên sheet giới hạn 31 ký tự
            sheet_name = f"S_{str(style)[:28]}"
            df_style.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Format sheet Style
            ws_s = writer.sheets[sheet_name]
            style_header_fmt = workbook.add_format({'bold': True, 'bg_color': style_colors.get(style, '#D7E4BC'), 
                                                   'font_color': 'white', 'border': 1})
            
            for col_num, value in enumerate(df_style.columns.values):
                ws_s.write(0, col_num, value, style_header_fmt)
            ws_s.set_column(0, 0, 22)
            ws_s.set_column(1, len(dates), 10, num_fmt)

    print(f"Hoàn tất xuất file: {filename}")
