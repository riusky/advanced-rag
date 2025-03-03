# 掌握大语言模型（LLM）与检索增强生成（RAG）

本仓库包含用于企业级 RAG 系统构建课程的配套资源，深入探讨开发过程中遇到的核心问题与解决方案。

课程可在 [edX](https://www.edx.org/learn/computer-science/pragmatic-ai-labs-advanced-rag) 学习。

## 基础 RAG 流程

![Naive RAG](images/Naiive_RAG.png)

## Jupyter 实验手册

1. [基础 RAG](01_simple_rag.ipynb)：本手册详解检索增强生成（RAG）的核心概念与实现。[![在 SageMaker Studio Lab 中打开](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/guyernest/advanced-rag/blob/main/01_simple_rag.ipynb) [![在 Google Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guyernest/advanced-rag/blob/main/01_simple_rag.ipynb)
2. [嵌入模型](02_embedding_model.ipynb)：解析 RAG 系统中嵌入模型的应用场景与局限性。
3. [语义分块](03_semantic_chunking.ipynb)：探讨基于语义的文档分块策略对信息检索的优化。
4. [上下文检索](04_contextual_retrieval.ipynb)：研究表格数据与数值信息的上下文感知检索方案。
5. [反向 Hyde](05_reverse_hyde.ipynb)：通过反向 Hyde 技术解决查询歧义与上下文脱节问题。
6. [混合搜索](06_hybrid_search.ipynb)：集成多语言支持与时效性感知的混合搜索方案。
7. [结果重排](07_reranking.ipynb)：优化检索结果排序提升输出精准度的关键技术。
8. [多模态检索](08_multimodal_pdf.ipynb)：突破文本限制的图像-文本跨模态检索实践。

## RAG 系统典型问题与优化方案

检索环节是 RAG 系统的核心痛点，可能因文档匹配偏差导致错误响应，本质上是机器学习中经典的 **精度-召回率权衡问题**。

![召回-精度权衡](images/Recall_Precision_in_RAG_Diagram.png)

以下现实场景中的文档复杂性因素会显著影响检索效果，我们提供针对性优化策略：

1. **长文档处理**
   - **痛点**：海量文本导致信息提取困难
   - **解决方案**：
     - 智能分块策略
       - 句子/段落分块
       - 重叠式固定分块
       - 统计驱动分块（见：[03_semantic_chunking.ipynb](03_semantic_chunking.ipynb)）
     - 层级式检索（父子块关联）
     - 上下文感知检索（见：[04_contextual_retrieval.ipynb](04_contextual_retrieval.ipynb)）

2. **查询-文档结构失配**
   - **痛点**：用户提问方式与文档组织形式不兼容
   - **解决方案**：
     - 假设文档嵌入（HyDE）
     - 反向 HyDE（见：[05_reverse_hyde.ipynb](05_reverse_hyde.ipynb)）

3. **领域专业术语**
   - **痛点**：通用模型难以理解专业术语
   - **解决方案**：
     - 领域定制化嵌入（见：[02_embedding_model.ipynb](02_embedding_model.ipynb)）
     - 混合检索增强（见：[06_hybrid_search.ipynb](06_hybrid_search.ipynb)）
     - 领域语料微调

4. **复杂格式文档**
   - **痛点**：扫描件等非结构化文档的传统检索失效
   - **解决方案**：
     - 多模态检索（见：[08_multimodal_pdf.ipynb](08_multimodal_pdf.ipynb)）
       - 计算机视觉提取图像信息
       - 多模态数据融合检索

## 企业级 RAG 系统架构

![Advanced RAG](images/Advanced_RAG.png)

## 本地开发环境配置

支持两种依赖管理方式：高性能工具 `uv` 或传统 `pip`。

**推荐使用 [`uv`](https://github.com/astral-sh/uv)**

### mac 环境

```shell
pip install uv  # 安装 UV 工具链
uv venv --python 3.12   # 创建 Python 3.12 虚拟环境（自动识别系统架构）
source .venv/bin/activate   # 激活虚拟环境
uv pip compile requirements.in -o requirements.txt  # 依赖解析
uv pip install -r requirements.txt  # 极速依赖安装
```

### windows 环境

```shell
# 安装 uv
pip install uv

# 创建虚拟环境（假设 Python 3.12 已安装）
uv venv --python 3.12

# 激活虚拟环境
.\.venv\Scripts\activate

# 编译依赖
uv pip compile requirements.in --output-file requirements.txt

# 安装依赖
uv pip install -r requirements.txt
```

 ![Advanced RAG Setup](images/advanced-rag-setup.gif)

 如果虚拟环境找不到 pip 或其他模块：

 ```shell
 curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py  # 获取 pip 包
python3 get-pip.py # 安装 pip 包
pip install ipykernel # 安装 ipykernel 以支持 VSC 中的 Jupyter 笔记本
```

要在 Jupyter lab 环境中执行笔记本，您需要添加以下命令：

```shell
python3 -m pip install jupyterlab # 安装 Jupyter Lab 和 ipykernel 以管理 Jupyter 的内核
python3 -m ipykernel install --user --name=.venv --display-name="Python (.venv)" # 从虚拟环境创建内核
jupyter lab
```

然后从内核列表中选择 Python (.venv)。

**使用传统的 `pip`**

1. 使用您喜欢的方法创建虚拟环境（例如，python -m venv myenv）。

2. 激活虚拟环境。

3. 使用 pip install -r requirements.txt 安装项目依赖。

注意：推荐使用 uv 方法，因为它速度快、易于使用且能有效管理项目依赖。然而，传统的 pip 方法也支持熟悉它的用户。

## 在 Google Colab 中设置

点击链接进入第一个实践实验室： [![Open In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guyernest/advanced-rag/blob/main/01_simple_rag.ipynb) 

在第一个单元格之前添加以下命令：

```shell
!git clone https://github.com/guyernest/advanced-rag.git
%cd advanced-rag
!pip install -q -r requirements.txt
```

在依赖安装结束时，您可能需要重启 Colab 运行时。记得切换回课程文件夹：

```shell
%cd advanced-rag
```

## 在 SageMaker Studio Lab 中设置

点击链接进入第一个实践实验室：[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/guyernest/advanced-rag/blob/main/01_simple_rag.ipynb)

在服务提示时克隆 GitHub 仓库。

在第一个单元格之前添加以下命令：

```
%cd advanced-rag
!pip install -q -r requirements.txt
```