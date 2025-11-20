import numpy as np

# 创建一个示例矩阵
matrix = np.array([
    [0, 2, 0, 5],
    [1, 0, 3, 0],
    [0, 0, 0, 0],
    [4, 0, 6, 7]
])

# 指定要查询的行（例如第1行，索引从0开始）
row_index = 1

# 获取该行中非零元素的列索引
nonzero_cols = np.nonzero(matrix[row_index])[0][0]

print("非零元素的列索引:", nonzero_cols)