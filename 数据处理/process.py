import pandas as pd

# 载入包含原始表格的 Excel 文件
file_path = r'数据处理\raw.xlsx'
xls = pd.ExcelFile(file_path)

# 定义翻译字典
translations = {
    "ACCOUNTID": "用户账号ID",
    "AUTHDURATION": "认证时长",
    "BDS_PROC_STATUS": "处理状态",
    "BILLINGCYCLE": "计费周期",
    "BILLING_TRACK": "计费跟踪",
    "BILL_WRITE_FLAG": "计费写入标志",
    "CALLREFERENCENO": "呼叫参考号",
    "CALLTYPE": "呼叫类型",
    "CDRTYPE": "话单类型",
    "DEVICETYPE": "设备类型",
    "DURATION": "通话时长",
    "ERRORCODE": "错误代码",
    "FREEFORMATDATA": "自由格式数据",
    "GSM_VTVD_FLAG": "GSM VTVD标志",
    "HMANAGE": "家庭管理",
    "HREGION": "家庭区域",
    "INITIATIVEFLAG": "主动标志",
    "INSTORETIME": "入库时间",
    "LOCALFEE": "本地费用",
    "LOCALFEE_S": "本地费用_分",
    "OFFSET_IN_SOURCE_FILE": "在源文件中的偏移量",
    "OLD_CALLTYPE": "旧呼叫类型",
    "OLD_ROAMTYPE": "旧漫游类型",
    "ORG_HREGION": "原家庭区域",
    "OTHERCELLID": "其他小区ID",
    "OTHERHREGION": "其他家庭区域",
    "OTHERLAC": "其他位置区码",
    "OTHERMANAGE": "其他管理",
    "OTHERNETTYPE": "其他网络类型",
    "OTHERVREGION": "其他归属地",
    "PACKAGE_INFO": "套餐信息",
    "PARTIALFLAG": "部分标志",
    "PART_FLAG": "部分标志",
    "PROCESSTIME": "处理时间",
    "PRODUCT_CODE": "产品代码",
    "RESERVED1": "预留1",
    "RESERVED2": "预留2",
    "RESERVED3": "预留3",
    "RESERVED4": "预留4",
    "RESERVED5": "预留5",
    "RESERVED6": "预留6",
    "RESERVED7": "预留7",
    "RESERVED8": "预留8",
    "RESERVED9": "预留9",
    "RFPOWERCAPABILITY": "射频功率能力",
    "RNCODE": "路由代码",
    "ROAMFEE": "漫游费用",
    "ROAMFEE_S": "漫游费用_分",
    "ROAMTYPE": "漫游类型",
    "RURALADDFEE": "农村附加费",
    "RURALADDFEE_S": "农村附加费_分",
    "SERVERMANAGE": "服务器管理",
    "SERVER_INFO": "服务器信息",
    "SERVICEBASIC": "基础服务",
    "SERVICECODE": "服务代码",
    "SERVICETYPE": "服务类型",
    "SPECIALTYPE": "特殊类型",
    "SPLITFLAG": "分割标志",
    "STARTTIME": "开始时间",
    "SUBSCRIBERID": "用户ID",
    "TARIFFFLAG": "费率标志",
    "TARIFFTRACK": "费率跟踪",
    "TELNUM": "电话号码",
    "THIRDTELNUM": "第三方电话号码",
    "TOLLADDFEE": "长途附加费",
    "TOLLADDFEE_S": "长途附加费_分",
    "TOLLDISCOUNT": "长途折扣",
    "TOLLFEE": "长途费用",
    "TOLLFEE2": "长途费用2",
    "TOLLFEE2_S": "长途费用2_分",
    "TOLLFEE_S": "长途费用_分",
    "TOTAL_FREE": "总免费",
    "TRUNKMANAGE": "中继管理",
    "VIEDOFLAG": "视频标志",
    "VREGION": "区域",
    "错误分支条件": "错误分支条件",
    "OPERATOR": "操作员",
    "PROCTIME": "处理时间",
    "NOTE": "备注",
    "UPDATETYPE": "更新类型"
}

# 创建翻译字典的反向字典
reverse_translations = {v: k for k, v in translations.items()}

# 定义翻译列名的函数
def translate_columns(df, translation_dict):
    df.columns = [translation_dict.get(col, col) for col in df.columns]
    return df

# 定义处理Excel文件的函数
def process_excel(file_path, output_path, direction="en_to_cn"):
    xls = pd.ExcelFile(file_path)
    sheets = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if direction == "en_to_cn":
            df = translate_columns(df, translations)
        elif direction == "cn_to_en":
            df = translate_columns(df, reverse_translations)
        sheets[sheet_name] = df

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# 示例用法
input_file_path = r'数据处理\raw.xlsx'
output_file_path_cn = r'数据处理\processed_cn.xlsx'
output_file_path_en = r'数据处理\processed_en.xlsx'

# 英文转中文
process_excel(input_file_path, output_file_path_cn, direction="en_to_cn")

# 中文转英文
process_excel(input_file_path, output_file_path_en, direction="cn_to_en")

print(f'数据已成功写入 {output_file_path_cn} 和 {output_file_path_en}')
