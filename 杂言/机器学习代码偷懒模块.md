# 机器学习可重用模块

> 记录经常使用的代码模块，加快开发的效率

### Stacking模型

```python
def get_out_fold(clf,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    # kf.split 返回划分好的训练集和测试集的索引
    for i, (train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.fit(x_tr,y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
```

### 时间记录

```python
from time import time
import datetime

t0 = time()

# 处理的程序

datetime.datetime.fromtimestamp(time()-t0).strftime("%M:%S:%f")
```

### ROC曲线绘制

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC

# 必须指明少数类的label
FPR, recall, thresholds = roc_curve(y, clf_proba.decision_function(X), pos_label=1)
area = AUC(y, clf_proba.decision_function(X))

plt.figure()
plt.plot(FPR, recall, color="red",
         label = "ROC curve(area = %0.2f)" % area)
plt.plot([0, 1], [0, 1], c="black", linestyle = "--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```

### 绘制3D图像

```python
from mpl_toolkits import mplot3d
def plot_3D(elev=30,azim=30,X=X,y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()
```

### 绘制饼状图

```python
## 绘制饼状图，查看存活情况
train_data.Survived.value_counts().plot.pie(labeldistance = 1.1
                                            , autopct = "%1.2f%%" #显示百分比
                                            , shadow=False
                                            , startangle = 90
                                            , pctdistance = 0.6
                                           )
plt.legend()
plt.show()
```

### 先编码在转为哑变量

```python
# 首先转为数字形式的编码
combined_train_test.Embarked = pd.factorize(combined_train_test.Embarked)[0]
# 使用哑变量编码，消除特征之间的共性(参数prefix表示前缀)原则会以_1出现
embark_dummies_df = pd.get_dummies(combined_train_test.Embarked,prefix=combined_train_test[["Embarked"]].columns[0])
```

### 网格搜索模型

```python
class GridSearch:
    """网格搜索"""
    
    def __init__(self, model):
        self.model = model
    
    def grid_get(self, X, y, param_grid):
        """参数搜索"""
        grid_search = GridSearchCV(self.model
                                   , param_grid =param_grid, scoring="neg_mean_squared_error"　# 此处为回归
                                   ,cv=5
                                  )
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        print(pd.DataFrame(grid_search.cv_results_)[["params", "mean_test_score", "std_test_score"]])
```

### 线性回归模型（均方根误差）

```python
def rmse(model, x, y):
    """定义均方根误差"""
    
    rmse = np.sqrt(-cross_val_score(model, x
                                    , y
                                    , scoring="neg_mean_squared_error"
                                    , cv=5))
    return rmse
```

### 缺失值的显示

- 表格显示

```python
Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([Total, percent], axis=1, keys=["Total", "Percent"])
missing_data
```

- 可视化显示

```python
## 可视化缺失值
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=missing_data[:20].index, y=missing_data.Percent[:20])
plt.xticks(rotation=90)
plt.xlabel("missing_features", fontsize=13)
plt.show()
```

#### 检查数据的偏度

目的：在回归模型中要求输入的数据必须正态分布，所以，可以通过计算数据的偏度来查看特征维度数据存在的左右偏度情况，也即正偏和负偏。

```python
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
## 检查数据的偏度
skew_fea = pd.DataFrame(all_data[numeric_feats].apply(lambda x:skew(x.dropna()                                                       )).sort_values(ascending=False))
skew_fea.rename(columns={0:"Skew"}, inplace=True)
```



