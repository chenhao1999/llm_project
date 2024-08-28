import pandas as pd
import joblib  # 用于加载模型

model_filename = './best_decision_tree_model.pkl'
file_path = './PE12.xlsx'
data = pd.read_excel(file_path)

features = ['VREGION', 'HREGION', 'OTHERTELNUM_right3',
            'OTHERVREGION', 'OTHERHREGION',  'LAC', 'CELLID',
            'TOTAL_FREE', 'FREEFORMATDATA', 'CALLREFERENCENO']

X = data[features]

# 对类别型特征进行编码
X = pd.get_dummies(X, columns=['VREGION', 'HREGION', 'OTHERTELNUM_right3',
                               'OTHERVREGION', 'OTHERHREGION', 'LAC', 'CELLID',
                               'TOTAL_FREE', 'FREEFORMATDATA', 'CALLREFERENCENO'
                             ])


# 加载保存的模型
loaded_model = joblib.load(model_filename)

# 对新的X数据进行预测
y_pred_new = loaded_model.predict(X)

# 输出预测结果
print("Predicted labels for the new data:")
print(y_pred_new)
