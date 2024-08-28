# 设计文档
### 总体框架
1. 拿到异常话单统计表后，对表格数据进行处理  
    1. 根据表格首先人工分析，找出可能有用的或是可能对话单造成影响的信息类型作为特征。
    2. 使用数据清洗以及特征工程进行进一步的处理，并划分训练集和测试集。
2. 对处理完的特征，通过调研，使用过采样+网格搜索+留一验证作为机器学习模型，同时选择Decision Tree，Random Forest，Gradient Boosting，CatBoost这4种经典的算法进行实验以及对比。
3. 代入数据集分别进行实验。
4. 对4种算法的结果进行结果可视化，分析对比各自的优势以及最终效果。
5. 将训练好的模型，使用docker封装成容器。
### 数据处理
1. 通过人工进行初步分析，找出可能有用的或是可能对话单造成影响的信息类型作为特征。
2. 数据清洗：
    1. 删除单一特征值。
    2. 处理缺失值。
3. 数据编码：使用类别型特征做编码。
4. 划分数据集：将数据集划分为训练集与测试集。
### 模型选择
#### 我们的算法
##### 基于过采样、网格搜索以及留一验证
1. 过采样：
    - 过采样是一种处理数据不平衡问题的方法。当数据集中某些类别的数据样本非常少时，模型往往会倾向于预测多数类别。为了缓解这个问题，可以使用过采样技术增加少数类样本的数量，从而平衡数据分布。

    - 常用方法：

        1. 随机过采样：简单地复制少数类样本，直到达到与多数类相似的数量。
        2. SMOTE（Synthetic Minority Over-sampling Technique）：通过插值生成新的少数类样本，而不是简单复制。
2. 网格搜索（Grid Search）：
    - 网格搜索是一种超参数优化方法。它通过穷举搜索给定的超参数空间，找到一组最优的参数组合，从而提高模型的性能。
    - 工作流程：

        1. 定义一个超参数空间，包括多个可能的参数值。
        2. 对每一组参数组合，训练模型并评估其性能。
        3. 选择在验证集上表现最好的参数组合。
3. 留一验证（Leave-One-Out Cross-Validation, LOOCV）：
    - 留一验证是一种特殊的交叉验证方法。它将数据集中的每一个样本依次作为验证集，剩余的样本作为训练集来训练模型。这意味着如果数据集中有N个样本，模型将训练N次，每次留下一个样本用于验证。
    - 优点：

        1. 充分利用了所有数据进行训练和验证。
        2. 特别适用于小数据集。
4. 结合使用：
    1. 过采样：首先对训练数据进行过采样，平衡各类别的样本数量。
    2. 网格搜索：在平衡后的数据集上进行网格搜索，寻找最优的超参数组合。
    3. 留一验证：使用留一验证评估每组超参数组合的性能，从而找到最适合的数据和模型的参数。
#### 对比算法
1. Decision Tree：
    1. 工作原理：
        1. 特征选择：决策树从所有特征中选择一个最佳特征来分割数据。这通常是通过一些指标来衡量的，比如信息增益、基尼不纯度或方差减少。在分类任务中，最常用的指标是信息增益和基尼不纯度；而在回归任务中，方差减少是一个常见的选择。
        2. 数据分割：一旦选择了最佳特征，数据就会根据该特征的不同取值进行分割。这个过程会持续进行，直到数据集中的所有样本属于同一类别（对于分类任务）或者达到预定的停止条件（如树的最大深度或分支中样本数目少于某个阈值）。
        3. 叶节点生成：当数据不能再进一步分割时，生成叶节点。叶节点中的值通常代表该分支下的样本所属的类别（在分类任务中）或是预测值（在回归任务中）。
    2. 优点：
        1. 简单易懂：决策树的逻辑类似于人类的决策过程，结构清晰，容易理解和解释。
        2. 处理混合数据类型：决策树可以处理数值型和分类型数据。
        3. 无需特征缩放：决策树不依赖于特征的尺度，因此无需进行特征标准化或归一化。
2. Random Forest：
    1. 工作原理：
        1. 集成学习：随机森林是一种“集成学习”（Ensemble Learning）方法，意思是通过组合多个弱学习器（如决策树）来构建一个强学习器。具体来说，随机森林会生成多个决策树，每个树都是在训练数据的随机子集上构建的。
        2. 随机样本和特征：
            - Bagging（自助采样）：随机森林使用了一种称为Bagging的技术，即从原始训练集中随机抽取样本，允许重复采样，生成多个不同的子集。每个决策树都在不同的子集上进行训练。
            - 随机特征选择：在构建每个节点时，随机森林不是考虑所有特征，而是随机选择一部分特征进行分裂。这种随机性减少了决策树之间的相关性，从而提高了集成模型的整体性能。
        3. 决策树集成：在进行预测时，分类任务中，随机森林会让所有的树投票，选择出现最多的类别作为最终预测结果。对于回归任务，随机森林会对所有树的预测结果求平均值。
    2. 优点：
        1. 高准确性：通过集成多个决策树，随机森林通常比单一决策树具有更高的准确性。
        2. 防止过拟合：由于使用了Bagging和随机特征选择，随机森林减少了模型的方差，从而降低了过拟合的风险。
        3. 鲁棒性：随机森林对数据中的噪声和异常值不敏感，表现稳定。
        4. 特征重要性：随机森林可以自动计算出特征的重要性，帮助我们理解哪些特征对模型的决策最为重要。
3. Gradient Boosting：这是一种强大的集成学习方法，主要用于回归和分类任务。它通过逐步构建多个弱学习器（通常是决策树），并将它们组合起来，形成一个强大的预测模型。
    1. 工作原理：
        1. 初始化模型：首先，使用一个简单的模型（如常数模型）来初始化预测。

        2. 计算残差：在每一步中，计算当前模型的残差，即预测值与真实值之间的差异。这些残差反映了模型的误差。

        3. 拟合新模型：针对残差拟合一个新的弱学习器（如决策树），使得新的学习器能够尽可能减少这些残差。

        4. 更新模型：将新学习器与之前的模型组合起来，更新模型的预测。这个组合通常通过加权求和的方式完成。

        5. 迭代：重复上述步骤，继续拟合新的学习器并更新模型，直到达到预定的迭代次数或模型误差达到可以接受的水平。
    2. 优点：
        1. 高准确性：梯度提升方法通常能够实现很高的预测准确性，尤其是在结构复杂的数据上。
        2. 灵活性：它可以应用于各种类型的数据（分类、回归）和损失函数。
        3. 自动特征选择：在训练过程中，梯度提升会自动选择最重要的特征进行学习。
4. CatBoost：是一种专门为处理类别型特征优化的梯度提升算法。
    1. 工作原理：
        1. Ordered Boosting：为了解决梯度提升中的预测偏差问题，CatBoost使用了一种叫做"ordered boosting"的技术。这个方法将训练数据划分为多个顺序排列的块，每个块只使用前面的块来进行模型训练，避免了信息泄露。

        2. 目标编码（Target Encoding）：对于类别型特征，CatBoost通过一种先进的目标编码技术来转换类别型数据。这个方法不会引入过多的噪声，同时避免了传统编码方式中可能导致的过拟合问题。

        3. 对称树（Symmetric Trees）：CatBoost使用对称树结构，这种树结构在构造时会选择所有分裂节点在相同的深度进行分裂，从而提高了模型的训练速度和预测速度。
    2. 优点：
        1. 易用性：CatBoost可以直接处理包含类别型特征的数据集，不需要额外的编码或转换。

        2. 高性能：在处理大规模数据集时，CatBoost的训练速度非常快，尤其是在GPU上的表现尤为出色。

        3. 鲁棒性：CatBoost具有很强的鲁棒性，减少了模型的调参需求，同时具有很好的泛化能力。

        4. 开箱即用：CatBoost在默认设置下就能提供非常强的表现，使得新手和专家都能轻松使用。

### 模型训练
#### 我们的算法
1. 过采样：使用SMOTE方法进行过采样，平衡数据集。参考代码如下：
    ```from imblearn.over_sampling import SMOTE

    # 初始化SMOTE
    smote = SMOTE(random_state=42)

    # 过采样数据集
    X_resampled, y_resampled = smote.fit_resample(X, y)
    ```

2. 使用网格搜索结合留一验证优化超参数，参考代码如下：
    ```
    from sklearn.model_selection import GridSearchCV, LeaveOneOut
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    #初始化随机森林分类器
    rf_clf = RandomForestClassifier(random_state=42)

    #定义超参数空间
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    #使用留一验证
    loo = LeaveOneOut()

    #初始化网格搜索
    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=loo, scoring='accuracy', n_jobs=-1)

    #在过采样的数据上进行网格搜索
    grid_search.fit(X_resampled, y_resampled)

    #输出最优参数和最高得分
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_}")
    ```
3. 最优模型的训练和评估，参考代码如下:
    ```
    # 使用最优参数训练模型
    best_rf_clf = grid_search.best_estimator_

    # 在原始测试集上进行预测（这里为了简化，将过采样后的数据作为整个数据集）
    y_pred = best_rf_clf.predict(X_resampled)

    # 评估模型准确性
    accuracy = accuracy_score(y_resampled, y_pred)
    print(f"Final model accuracy: {accuracy}")
    ```
#### 对比算法
1. Decision Tree：
    1. 使用scikit-learn库
    2. 初始化决策树分类器：`clf = DecisionTreeClassifier()`
    3. 训练模型：`clf.fit()`
    4. 模型预测与评估：
        ```
        #预测测试集
        y_pred = clf.predict(X_test)
        #评估模型
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy}")
        ```
    5. 可视化：`sklearn.tree.export_text`
2. Random Forest：
    1. 使用RandomForestClassifier创建并训练随机森林模型：
        ```
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        ```
    2. 模型预测与评估:
        ```
        # 预测测试集
        y_pred = rf_clf.predict(X_test)

        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy}")
        ```
3. Gradient Boosting：
    1. 使用GradientBoostingClassifier来创建并训练模型：
        ```
        # 初始化梯度提升分类器
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # 训练模型
        gb_clf.fit(X_train, y_train)
        ```
    2. 模型预测与评估:
        ```
        # 预测测试集
        y_pred = gb_clf.predict(X_test)

        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy}")
        ```
4. CatBoost：
    1. 初始化CatBoost分类器
        ```
        model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, cat_features=categorical_features_indices, verbose=False)
        ```
    2. 训练模型
        ```
        model.fit(X_train, y_train)
        ```
    3. 模型评估与预测
        ```
        # 预测测试集
        y_pred = model.predict(X_test)

        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        ```
### 封装成容器
1. 安装好docker并编写一个Python脚本来加载和预测模型。
2. 构建docker镜像，参考代码如下：
    ```
    docker build -t my-catboost-model .
    ```
3. 运行docker容器，通过挂载的方式，把要预测的PE12数据挂载到data路径下，可以调用模型对结果进行预测。参考代码如下：
    ```
    docker run -rm -v ./myroot my-catboost-model
    ```

