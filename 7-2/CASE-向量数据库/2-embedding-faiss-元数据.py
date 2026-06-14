import os
import numpy as np
import faiss
from openai import OpenAI

# Step1. 初始化 API 客户端
try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
except Exception as e:
    print("初始化OpenAI客户端失败，请检查环境变量'DASHSCOPE_API_KEY'是否已设置。")
    print(f"错误信息: {e}")
    exit()

# Step2. 准备示例文本和元数据
# 模拟贵州茅台财务报告中的关键段落，实际应用中这些数据来自 PDF 解析
documents = [
    {
        "id": "doc1",
        "text": "贵州茅台2025年实现营业总收入1738.04亿元，同比增长15.71%；归属于上市公司股东的净利润862.28亿元，同比增长15.38%。其中茅台酒实现营业收入1459.37亿元。",
        "metadata": {"source": "贵州茅台2025年年度报告.pdf", "category": "财务概况", "page": 8}
    },
    {
        "id": "doc2",
        "text": "报告期内，公司毛利率为91.57%，较上年同期下降0.65个百分点，主要系系列酒收入占比提升所致。销售费用率为3.21%，管理费用率为5.86%，整体费用控制良好。",
        "metadata": {"source": "贵州茅台2025年年度报告.pdf", "category": "盈利能力", "page": 15}
    },
    {
        "id": "doc3",
        "text": "贵州茅台拟向全体股东每10股派发现金红利275.48元（含税），合计拟派发现金红利346.03亿元，占2025年度归属于上市公司股东净利润的40.13%。",
        "metadata": {"source": "贵州茅台2025年利润分配方案.pdf", "category": "分红方案", "page": 3}
    },
    {
        "id": "doc4",
        "text": "截至2025年末，公司经营活动产生的现金流量净额为695.41亿元，同比增长8.23%。货币资金余额为1877.53亿元，资产负债率为18.72%，财务状况稳健。",
        "metadata": {"source": "贵州茅台2025年年度报告.pdf", "category": "现金流与偿债", "page": 22}
    }
]

# Step3. 创建元数据存储和向量列表
# 我们使用一个简单的列表来存储元数据。列表的索引将作为FAISS的ID。
# 这种方式简单直接，适用于中小型数据集。
# 对于大型数据集，可以考虑使用字典或数据库（如Redis, SQLite）
metadata_store = []
vectors_list = []
vector_ids = []

print("正在为文档生成向量...")
for i, doc in enumerate(documents):
    try:
        # 调用API生成向量
        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=doc["text"],
            dimensions=1024,
            encoding_format="float"
        )
        
        # 获取向量
        vector = completion.data[0].embedding
        vectors_list.append(vector)
        
        # 存储元数据，并使用列表索引作为唯一ID
        metadata_store.append(doc)
        vector_ids.append(i) # 自定义ID与列表索引一致
        
        print(f"  - 已处理文档 {i+1}/{len(documents)}")

    except Exception as e:
        print(f"处理文档 '{doc['id']}' 时出错: {e}")
        continue

# 将向量列表转换为NumPy数组，FAISS需要这种格式
vectors_np = np.array(vectors_list).astype('float32')
vector_ids_np = np.array(vector_ids)

# Step4. 构建并填充 FAISS 索引
dimension = 1024  # 向量维度
k = 2             # 查找最近的2个邻居

# 创建一个基础的L2距离索引
index_flat_l2 = faiss.IndexFlatL2(dimension)

# 使用IndexIDMap来包装基础索引，能够映射我们自定义的ID
# 这就是关联向量和元数据的关键！
index = faiss.IndexIDMap(index_flat_l2)

# 将向量和它们对应的ID添加到索引中
index.add_with_ids(vectors_np, vector_ids_np)

print(f"\nFAISS 索引已成功创建，共包含 {index.ntotal} 个向量。")


# Step5. 执行搜索并检索元数据
query_text = "贵州茅台2025年的利润和分红情况如何"
print(f"\n正在为查询文本生成向量: '{query_text}'")

try:
    # 为查询文本生成向量
    query_completion = client.embeddings.create(
        model="text-embedding-v4",
        input=query_text,
        dimensions=1024,
        encoding_format="float"
    )
    query_vector = np.array([query_completion.data[0].embedding]).astype('float32')

    # 在FAISS索引中执行搜索
    # search方法返回两个NumPy数组：
    # D: 距离 (distances)
    # I: 索引/ID (indices/IDs)
    distances, retrieved_ids = index.search(query_vector, k)

    # Step6. 展示结果
    print("\n--- 搜索结果 ---")
    # `retrieved_ids[0]` 包含与查询最相似的k个向量的ID
    for i in range(k):
        doc_id = retrieved_ids[0][i]
        
        # 检查ID是否有效
        if doc_id == -1:
            print(f"\n排名 {i+1}: 未找到更多结果。")
            continue

        # 使用ID从我们的元数据存储中检索信息
        retrieved_doc = metadata_store[doc_id]
        
        print(f"\n--- 排名 {i+1} (相似度得分/距离: {distances[0][i]:.4f}) ---")
        print(f"ID: {doc_id}")
        print(f"原始文本: {retrieved_doc['text']}")
        print(f"元数据: {retrieved_doc['metadata']}")

except Exception as e:
    print(f"执行搜索时发生错误: {e}")
