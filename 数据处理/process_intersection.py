'''
Author: chenhao1999 1027783090@qq.com
Date: 2024-07-29 21:16:34
LastEditors: chenhao1999 1027783090@qq.com
LastEditTime: 2024-07-29 21:18:37
FilePath: \llm_project\数据处理\process_intersection.py
Description: 给两个文件取交集
'''
import pandas as pd

# 载入包含原始表格的 Excel 文件
file_path = r'数据处理\raw.xlsx'
xls = pd.ExcelFile(file_path)

# 读取 Excel 文件中的表格以获取列名和数据
df1 = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
df2 = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
df1.dropna(axis=1, how='all', inplace=True)
df2.dropna(axis=1, how='all', inplace=True)
# 过滤掉空列名和以 'Unnamed:' 开头的列名
columns1 = [col for col in df1.columns if col and not col.startswith('Unnamed:')]
columns2 = [col for col in df2.columns if col and not col.startswith('Unnamed:')]

# 获取两个表格列名的交集
all_columns = list(set(columns1)&(set(columns2)))

# 创建一个包含合并后列名的空 DataFrame
combined_df = pd.DataFrame(columns=all_columns)

# 需要处理的表格列表（在此示例中，假设处理相同的 df1 和 df2 表格）
sheets_to_process = [df1, df2]

# 处理每个表格
for df in sheets_to_process:
    # 创建一个与 combined_df 具有相同列的 DataFrame，数据用 NaN 填充
    temp_df = pd.DataFrame(columns=combined_df.columns)
    
    # 遍历当前表格的列
    for col in df.columns:
        if col in combined_df.columns:
            temp_df[col] = df[col]
    
    # 将对齐后的 DataFrame 追加到合并的 DataFrame 中
    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# 删除没有数据的列
combined_df.dropna(axis=1, how='all', inplace=True)

# 将最终合并后的 DataFrame 保存到一个新的 Excel 文件中
output_path = r'数据处理\raw_intersection.xlsx'
combined_df.to_excel(output_path, index=False)

print(f'数据已成功写入 {output_path}')
