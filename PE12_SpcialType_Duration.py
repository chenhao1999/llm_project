import pandas as pd

file_path = 'PE12.xlsx'

# 读取Excel文件的工作表
with pd.ExcelFile(file_path) as xls:
    sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
def analyze_cdrs(sheets):
 
    error_records = {}

    # 遍历每一个工作表
    for sheet_name, df in sheets.items():
        # 遍历DataFrame中的每一行
        for index, row in df.iterrows():
                       if '错误分支条件' in df.columns and row['错误分支条件'] is not None:
                condition = row['错误分支条件']
                if "目计费事件.特殊通话类型 match ['8163','8169']and 目计费事件.通话时长<60" in condition:
                    if condition not in error_records:
                        error_records[condition] = []
                    error_records[condition].append(row)
    
    return error_records


error_records = analyze_cdrs(sheets)


for condition, records in error_records.items():
    print(f"Error Condition: {condition}")
    for record in records:
        print(f"SPECIALTYPE: {record['SPECIALTYPE']}, DURATION: {record['DURATION']}")
    print("\n---\n")