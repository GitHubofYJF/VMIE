import numpy as np

def vector_scramble(data_vector, mu, x0, size,m, n):
    """向量置乱加密（核心函数）"""
    # 转换为64x64矩阵
    matrix = data_vector.reshape(size,size)
    # 生成行置换序列
    row_seq = logistic_map(x0, mu, size, m)
    row_perm = generate_permutation(row_seq)
    
    # 生成列置换序列（继承行序列最终状态）
    col_seq = logistic_map(row_seq[-1], mu, size, n)
    col_perm = generate_permutation(col_seq)
    
    # 执行置乱
    scrambled = matrix[row_perm, :]
    scrambled = scrambled[:, col_perm]
    
    # 转换回向量
    return scrambled.flatten()

def vector_descramble(scrambled_vector, mu, x0,size, m, n):
    """向量置乱解密（核心函数）"""
    # 转换为64x64矩阵
    matrix = scrambled_vector.reshape(size, size)
    
    # 重新生成置换序列（必须与加密时一致）
    row_seq = logistic_map(x0, mu, size, m)
    row_perm = generate_permutation(row_seq)
    col_seq = logistic_map(row_seq[-1], mu, size, n)
    col_perm = generate_permutation(col_seq)
    
    # 计算逆置换
    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)
    
    # 执行逆置换（顺序相反）
    descrambled = matrix[:, inv_col]
    descrambled = descrambled[inv_row, :]
    
    return descrambled.flatten()

# 辅助函数保持不变
def logistic_map(initial, mu, length, discard=0):
    x = initial
    for _ in range(discard):
        x = mu * x * (1 - x)
    seq = []
    for _ in range(length):
        x = mu * x * (1 - x)
        seq.append(x)
    return np.array(seq)

def generate_permutation(seq):
    return np.argsort(seq)
def encryted_64(data_vector, mu, x0, m, n):
    data_vector = data_vector.flatten() 
    z1=data_vector[0:4096]
    z2=data_vector[4096:8192]
    z3=data_vector[8192:12288]
    size=64
    m1=vector_scramble(z1, mu,size, x0, m, n)
    m2 = vector_scramble(z2, mu,size, x0, m, n)
    m3= vector_scramble(z3, mu,size, x0, m, n)
    tempz = np.concatenate((m1, m2, m3))
    return tempz
def decryted_64(data_vector, mu, x0, m, n):
    data_vector = data_vector.flatten() 
    z1=data_vector[0:4096]
    z2=data_vector[4096:8192]
    z3=data_vector[8192:12288]
    size=128
    m1=vector_descramble(z1, mu, x0, size,m, n)
    m2 = vector_descramble(z2, mu, x0,size, m, n)
    m3= vector_descramble(z3, mu, x0,size, m, n)
    tempz = np.concatenate((m1, m2, m3))
    return tempz

def encryted_128(data_vector, mu, x0, m, n):
    data_vector = data_vector.flatten() 
    z1=data_vector[:16384]
    z2=data_vector[16384:32768]
    z3=data_vector[32768:]
    size=128
    m1=vector_scramble(z1, mu, x0,size, m, n)
    m2 = vector_scramble(z2, mu,x0,size, m, n)
    m3= vector_scramble(z3, mu, x0,size, m, n)
    tempz = np.concatenate((m1, m2, m3))
    return tempz
def decryted_128(data_vector, mu, x0, m, n):
    data_vector = data_vector.flatten() 
    z1=data_vector[:16384]
    z2=data_vector[16384:32768]
    z3=data_vector[32768:]
    size=128
    m1=vector_descramble(z1, mu, x0, size,m, n)
    m2 = vector_descramble(z2, mu, x0,size, m, n)
    m3= vector_descramble(z3, mu, x0,size, m, n)
    tempz = np.concatenate((m1, m2, m3))
    return tempz
# 测试用例
if __name__ == "__main__":
    # 生成测试向量（64x64=4096个元素）
    original_vector = np.arange(4096).astype(np.uint8)
    print("Original vector:", original_vector)
    # 可视化测试数据
   
    # 设置加密参数
    mu = 3.9999    # 更强的混沌参数
    x0 = 0.3333    # 初始值
    m = 100        # 增强安全性的预迭代
    n = 200
    
    # 加密过程
    encrypted = vector_scramble(original_vector, mu, x0, m, n)
    print("Encrypted vector:", encrypted)
    
    # 解密过程
    decrypted = vector_descramble(encrypted, mu, 0.444, m, n)
    print("Decrypted vector:", decrypted)
    
    # 验证解密结果
    print("Decryption correct?", np.all(original_vector == decrypted))