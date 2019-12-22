## PCA 数据降维

### 前言

​		最近，在学习机器学习算法时，看到了PCA（主成分分析）和SVD（奇异值分解）,这是两个基本数据降维的算法，而在降维算法中的“降维”主要是指<font color = red>**降低特征矩阵中特征数量的维度**</font>,直观上理解我们希望数据带有较少的特征，表示较好的效果。本文主要讲解PCA和SVD算法，算法的理解请仔细阅读文章后的参考链接，本文主要侧重应用和理解。:smile:

### PCA（主成分分析）

#### 算法原理

这里直接给出算法的具体步骤，算法的数学原理强烈推荐参考，[PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)。

>设有m条n维数据。
>
>1）将原始数据按列组成n行m列矩阵X
>
>2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值
>
>3）求出协方差矩阵
>
>4）求出协方差矩阵的特征值及对应的特征向量
>
>5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P
>
>6）Y=PX即为降维到k维后的数据

#### 代码演示（案例来自于上述链接）

1. **将原始数据按列组成n行m列矩阵X**

   ```python
   # 引入包
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   from scipy import mat, linalg
   ```

   ```python
   # 创造特征数据
   data = np.array([[-1, -2],
                   [-1, 0],
                   [0, 0],
                   [2, 1],
                   [0, 1]])
   data = data.T　# 为了保持和上述推文中的数据在相同的维度，这里做了数据装置，理解为m行特征和n列样本
   
   # 显示原始数据数据
   plt.scatter(data[0,:], data[1,:])
   plt.title("Initial data")
   plt.savefig("imgs/原始数据分布.png")
   plt.xlabel("feature1")
   plt.ylabel("feature2")
   plt.show()
   ```

   ![](./imgs/原始数据分布.png)

2. **将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值**

   ```python
   # 2. 每一字段减均值
   mean = data.mean(axis = 1)
   for i in range(len(mean)):
       for j in range(data.shape[1]):
           data[i, j] = data[i, j] - mean[i]
   data    
   ```

3. **求出协方差矩阵** [np.cov详解](https://blog.csdn.net/jeffery0207/article/details/83032325)

   ```python
   # 3. 计算协方差矩阵()
   X_cov = np.cov(data,ddof=0) 
   X_cov
   ```

   输出结果：

   ```
   array([[1.2, 0.8],
          [0.8, 1.2]])
   ```

4. **求出协方差矩阵的特征值及对应的特征向量**

   ```python
   # 4.求解特征向量和特征值
   d, u = linalg.eig(X_cov)
   print("特征值:", d)
   print("特征向量", u)
   
   ```

   输出结果：

   ```
   特征值: [2. +0.j 0.4+0.j]
   特征向量 [[ 0.70710678 -0.70710678]
    [ 0.70710678  0.70710678]]
   ```

5. **将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P**

   ```python
   # 4.排列特征值 从大到小
   ind = np.argsort(d)
   ind = ind[::-1]
   
   u = u[:, ind]
   u
   ```

   输出结果：

   ```
   array([[ 0.70710678, -0.70710678],
          [ 0.70710678,  0.70710678]])
   ```

6. **Y=PX即为降维到k维后的数据**

   ```python
   # 计算新基下的坐标（选择减低到一个维度的数据）
   Y = np.dot(u[1, :].T, data)
   Y
   ```

   输出结果：

   ```
   array([-2.12132034, -0.70710678,  0.        ,  2.12132034,  0.70710678])
   ```

   ![](./imgs/数据降到一个维度.png)

#### Sklearn中 PCA算法

​		本小结主要针对iris数据中四个特征维度的数据，通过使用Sklearn中封装的好的降维算法，通过把数据特征降低到人眼可视化的层次，直观的理解PCA减维后的数据的分布，更好的理解的PCA算法。

1. **加载数据**

   ```python
   import matplotlib.pyplot as plt
   from sklearn.datasets import load_iris
   from sklearn.decomposition import  PCA
   from mpl_toolkits.mplot3d import Axes3D
   import pandas as pd
   
   # 1.加载数据
   iris = load_iris()
   iris.data.shape
   Y = iris.target
   X = iris.data
   df = pd.DataFrame(X, columns=iris.feature_names)
   df
   ```

   **鸢尾花数据**：

   ![](./imgs/iris_data.png)

   2. **降到三维**

      ```python
      pca = PCA(n_components=3)
      pca = pca.fit_transform(X)
      new_X = pca
      
      # 绘制可视化图
      fig1 = plt.figure(figsize=(12, 6))
      ax = Axes3D(fig1)
      
      for i, c in enumerate(["r", "b", "g"]):
          ax.scatter(new_X[Y==i, 0]
                     , new_X[Y==i, 1]
                     , new_X[Y==i, 2]
                     , c = c
                     , label = iris.target_names[i]
                    )
          
      ax.set_zlabel("feature3")
      ax.set_ylabel("feature2")
      ax.set_xlabel("feature1")
      plt.title("The 3dim distribution of data")
      plt.legend()
      plt.savefig("imgs/3dim.png")
      plt.show()
      ```

      ![](./imgs/3dim.png)

   3. **降到二维**

      ```python
      ＃ 实例化
      pca = PCA(n_components=2)
      new_X = pca.fit_transform(X)
      pd.DataFrame(new_X)
      
      # 画个图
      plt.figure(figsize=(12, 6))
      plt.scatter(new_X[Y == 0, 0], new_X[Y==0, 1], label = iris.target_names[0])
      plt.scatter(new_X[Y == 1, 0], new_X[Y==1, 1], label = iris.target_names[1])
      plt.scatter(new_X[Y == 2, 0], new_X[Y==2, 1], label = iris.target_names[2])
      plt.xlabel("feature 1")
      plt.ylabel("feature 2")
      plt.legend()
      plt.title("The 2dim distribution of data")
      plt.savefig("imgs/2dim.png")
      plt.show()
      ```

      ![](./imgs/2dim.png)

   4. **降到一维**

      ```python
      pca = PCA(n_components=1)
      pca = pca.fit_transform(iris.data)
      x_new = pca
      x_new.shape
      Y_data = [0] * 50
      
      for i in [0, 1, 2]:
          plt.scatter(x_new[Y==i, :], Y_data, label = iris.target_names[i])
      
      plt.ylim()
      plt.title("PCA of iris datasets")
      plt.xlabel("feature")
      plt.savefig("imgs/1dim.png")
      plt.legend()
      plt.show()
      ```

      ![](./imgs/1dim.png)

      <font color = red>**注意：**</font>PCA是将已经存在的特征进行压缩，减低维度后的特征不是原本特征矩阵中的任何一个，而是通过某些方式组合起来的新特征，这使得降低维度后的特征不在具有原本数据下的可读性。
### SVD（奇异值分解）

   >​		首先，引用别人博客中的一段对SVD的描述：<font color =red>**奇异值分解是一个有着很明显的物理意义的一种方法，它可以将一个比较复杂的矩阵用更小更简单的几个子矩阵的相乘来表示，这些小矩阵描述的是矩阵的重要的特性。**</font>就像是描述一个人一样，给别人描述说这个人长得浓眉大眼，方脸，络腮胡，而且带个黑框的眼镜，这样寥寥的几个特征，就让别人脑海里面就有一个较为清楚的认识，实际上，人脸上的特征是有着无数种的，之所以能这么描述，是因为人天生就有着非常好的抽取重要特征的能力，让机器学会抽取重要的特征，SVD是也一个重要的方法。在机器学习领域，有相当多的应用与奇异值都可以扯上关系，比如做feature reduction的PCA，做数据压缩（以图像压缩为代表）的算法，还有做搜索引擎语义层次检索的LSI（Latent Semantic Indexing）。
   >
   >​	本小结主要针对SVD的应用作出一些总结，如图像压缩，数据降维。详细概念请拜读最底部参考链接。

#### 图像压缩实例

**原理**：如果对一个图像矩阵A进行奇异值分解，那么直观上可以用一个矩阵分解后的的线性组合来表示，公式如下：

**奇异值分解表达形式**：
$$
A = U * L * V_{T}
$$
**图解**：

![](./imgs/奇异值图形化.png)

**线性表示**：（$u_{i}$表示左奇异值矩阵向量，$v_{i}$表示右奇异值矩阵向量，$\sigma_{i}^{T}$表示奇异值矩阵中的奇异值，且**$\sigma_{i} >= \sigma_{i+}$**）
$$
A=\sigma_{1} u_{1} v_{1}^{\mathrm{T}}+\sigma_{2} u_{2} v_{2}^{\mathrm{T}}+\ldots+\sigma_{r} u_{r} v_{r}^{\mathrm{T}}
$$
**矩阵形式表示如下**：（其中$U$表示原始图片的左奇异矩阵，$L$为奇异值矩阵，$V_t$表示右奇异矩阵）
$$
Image_{dir} = U[:, :n]*L[:n, :,n] * V_T[:n, n]
$$
**代码演示**：

（１） **导入数据（图片来之我喜欢的一个韩剧女主）**:smile:

```python
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('./imgs/11.jpg')
plt.imshow(img)
plt.savefig("./imgs/123.png")
plt.show()
```

输出结果：

![](./imgs/123.png)

（２）**计算奇异值矩阵**

```python
# 装换为二维数据
img_temp = img.reshape(img.shape[0], img.shape[1]*img.shape[2])
U, L, V_T = np.linalg.svd(img_temp)
```

（３）**取前n个奇异值（从大到小）**

```python
# 创建画布
fig, ax = plt.subplots(4
                       ,4
                       , figsize = (24, 12)
                       ,subplot_kw={"yticks": []
                                    , "xticks":[]}
                      )

# 绘制压缩后图片
for i, ax in enumerate((ax.flat)):
    # 选取前i个奇异值
    svd_k = i+1
    img_res = (U[:, 0:svd_k]).dot(np.diag(L[0:svd_k])).dot(V_T[:svd_k, :])
    ax.imshow(img_res.reshape(500, 500, 3).astype(np.uint8))
    ax.set(title = "svd_k = {}".format(svd_k))

plt.savefig("./imgs/SVD.png")
```

**输出结果**：

![](./imgs/SVD.png)

**说明**：从图中可以看出当选取较少的奇异值，例如1, 2时，此时的压缩后的图片较为模糊，也即压缩后不能表示原来的图片，而随着所选奇异值个数的增加，图像越来清晰，当奇异值的个数到达合适的值后，就能达到压缩图片的效果。原则上如果一张图片为500*500，那么它的像素值为2500000，而经过奇异值分解后的图片的像素为(500 + 500 + 1)×20（所选奇异值个数） = 20020。所以，通过奇异值分解极大减少了图片存储量。

（3）**数据分析**

- **奇异值**

  ```python
  plt.figure(figsize=(12, 6))
  plt.plot(range(len(L)), L)
  plt.title("The Singular Value")
  plt.xlabel("The number of singular value")
  plt.savefig("./imgs/The number of singular value")
  plt.show()
  ```

  ![The number of singular value](/home/gavin/Machine/Sklearn Machine/PCA数据降维/imgs/The number of singular value.png)

  **说明：**奇异值下降的速度特别快，在很多情况下前10%甚至1%的奇异值之和就就占全部奇异值之和的99%以上，所以可以用前r个奇异值来近似描述矩阵。

- **累加奇异值**

  ```python
  plt.figure(figsize=(12, 6))
  plt.plot(range(len(L)), np.cumsum(L))
  plt.title("The Cumulative singular value")
  plt.xlabel("The number of singular value")
  plt.savefig("./imgs/The Cumulative singular value")
  plt.show()
  ```

  ![The Cumulative singular value](/home/gavin/Machine/Sklearn Machine/PCA数据降维/imgs/The Cumulative singular value.png)

#### 数据降维实例

SVD可以实现行实例降维和列数特征数据降维，列数据降维公式如下：
$$
A_{m*n} * V_{n*r} = U_{m*r} * L_{r*r}
$$
（１）**Sklearn演示**

```python
"""说明:在slklearn中PCA的降维操作分为两步,(1)奇异值分解（２）PCA选择降为后的维度"""
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载数据
faces = fetch_lfw_people(min_faces_per_person=60)
X = faces.data

# 绘制特征减少后的图
pca = PCA(1).fit(X)

# 右奇异矩阵(可以取得前k行作为降维后的维度), 默认右奇异矩阵的维数为n*n
# 通过SVD奇异值分解实现得到右奇异矩阵
V = pca.components_ # 保存计算后的V_T值
V
```

结果显示:

```pythom
array([[-0.00579705, -0.00595365, -0.00615767, ..., -0.01000114,
        -0.00901094, -0.00813919]], dtype=float32)
```

### 分类模型实战（手写数字识别）

​		本小结主要针对手写数据识别，由于原始的数据每一个样本的特征维度为28*28，这是一个很高的维度，我们希望通过PAC实现特征维度的降维来实现用较少的数据特征表示最佳的分类效果，话不多说，let's do it ！:smirk:

（１）**加载数据**

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

digits = pd.read_csv("data/digit recognizor.csv")
Y = digits.iloc[:, 0]
X = digits.iloc[:, 1:]
```

（２）**绘制可累计解释方差率**（注：如果我们不知道应该把数据降到多少维较好，那么这是一个不错的指标）

- **粗略绘制**

  ```python
  # 绘制累计方差贡献率
  pca_line = PCA().fit(X)
  plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
  plt.ylabel("ratio")
  plt.xlabel("the number of feature vector")
  plt.title("explained_variance_ratio")
  plt.savefig("./imgs/特征选择.png")
  plt.show()
  ```

  ![](./imgs/特征选择.png)

（３）**模型批判**

```python
# 使用降维后的数据进行特征向量的选择和模型的评估
score = []

for i in range(1, 101, 10):
    X_dir = PCA(i).fit_transform(X)
    # 使用随机森林
    once = cross_val_score(RFC(n_estimators=10
                            , random_state=10)
                        , X_dir
                        , Y
                        , cv=5).mean()
    score.append(once)

# 画学习曲线
plt.plot(range(1, 101, 10), score)
plt.ylabel("accuracy")
plt.xlabel("the number of feature vector")
plt.title("The accuracy of model in different feature vectors")
plt.savefig("./imgs/RFC.png")
plt.show()
```

![](./imgs/RFC.png)

**说明**：从图中可以看出降低维度较少时，模型的效果极差，而随着降维特征数量上升会有一个急剧加速的曲线，而后变得平缓，所以，可以在前面达到峰值的左右区间进行进一步的探索。

（３）**进一步细化降维特征区间**

```python
# 细化特征维数，选择更加精确到降维维度
score = []

for i in range(10, 25):
    X_dir = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10
                            , random_state=10)
                        , X_dir
                        , Y
                        , cv=5).mean()
    score.append(once)

# 画学习曲线
plt.plot(range(10, 25), score)
plt.ylabel("accuracy")
plt.xlabel("the number of feature vector")
plt.title("The accuracy of model in different feature vectors")
plt.savefig("./imgs/精确调节.png")
plt.show()
```

![](./imgs/精确调节.png)

  （４）**降维前后模型效果对比**

- **降维前**

  ```python
  rfc = RFC(n_estimators=100, random_state=10)
  score = cross_val_score(rfc, X, Y, cv=5).mean()
  print("Score : {}".format(score))
  ```

  Score : 0.9430240579036873

- **降维后**

  ```python
  rfc = RFC(n_estimators=100, random_state=10)
  score = cross_val_score(rfc, X_dir, Y, cv=5).mean()
  print("Score : {}".format(score))
  ```

  Score : 0.9641429333681589

  **说明**：尽管降低维度后模型的准确率降低了，但是相比与数据本身，我们用较少的特征就达到了客观的效果，不仅如此，降维后，模型的训练时间也极大的减少，这使得效率大大增加。当然，也可能受到模型本身的限制，模型效果不佳，可以继续尝试其他模型。:smiley:

### 总结

​		本文主要介绍PCA数据降维算法，主要总结了自己所学的一些分散知识点，现在汇总到一起，做了一些总结，整理了一些自己的思路，文章有错误的地方和不足的地方，请及时私信我，谢谢！后续，文章会不断完善......:slightly_smiling_face:

### 参考链接

**知乎讨论**：[奇异值的物理意义是什么？](https://www.zhihu.com/question/22237507)

**PCA详细推导**：[PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)

[SVD（奇异值分解）小结](https://www.cnblogs.com/endlesscoding/p/10033527.html)

[如何让奇异值分解(SVD)变得不“奇异”？](https://redstonewill.com/1529/)

[机器学习实战——SVD（奇异值分解）](https://blog.csdn.net/qq_36523839/article/details/82347332)

[奇异值分解（SVD）原理](https://blog.csdn.net/u013108511/article/details/79016939)

[菜菜的机器学习sklearn课堂](https://study.163.com/course/introduction/1006536008.htm)

**图像压缩**：[奇异值的物理意义是什么？强大的矩阵奇异值分解(SVD)及其应用](https://blog.csdn.net/c2a2o2/article/details/70159320)

**有干货：**[奇异值分解(SVD)详解及其应用](https://blog.csdn.net/shenziheng1/article/details/52916278)







































