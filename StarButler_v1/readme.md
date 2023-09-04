第一代，计划实现一个拥有长期记忆，并且可以进行连续语音对话能力的语音助手
- [x] 关键词唤醒
- [x] 语音输入
- [x] 语音对话
- [x] 长期记忆

# 运行：

main.py是最先写的代码，main_restruct是在main.py基础上进行重构的。



拥有长期记忆的直接运行：

main_restruct_memory.py

# 使用到的：

Whisper 语音识别

OpenaiEmbeddings

Chroma

# 参考：

OpenAI Embeddings和向量数据库速成课程： https://www.bilibili.com/video/BV1Kk4y1M7BT
OpenAI 官方文档：https://platform.openai.com/docs/api-reference/embeddings

Chroma官网：https://www.trychroma.com/

Chroma官方文档：https://docs.trychroma.com/getting-started



在Chroma中，集合相当于数据表，数据表内含有文档，文档相当于行数据。
将两个文档添加到了"all-my-documents"集合中。
每个文档都有一个唯一的标识符（ids），
以及与之相关的元数据（metadatas），如文档的来源。
documents是实际的文本内容。Chroma会处理文档的分词、嵌入和索引等操作。

主要涉及到的需要学习的就是：新建表集合，添加文档（根据embeddings），查找文档（根据embeddings）



# 存在的问题：

尽管能够提取长期记忆，但是每次对话前查找相关记忆、对话后添加相关记忆的模式，过于单一，容易出现奇奇怪怪的内容

1、记忆插入格式需要优化，有时容易使得内容 被GPT认为是按照记忆格式回答问题。

2、完全依靠embeddings的存储和提取，使针对同样的问句，提取出的数据都一样。而不一样的话题，可能因为同样的询问方式而判定成相关的记忆，使得回答不符合期望。



# 计划优化

1、设计优化：改进成利用api的选择性记忆模式，而不是现在的全盘性记忆

2、针对问题1：尚无想法

3、针对问题2：尚无改进措施，计划引入时间和相关度权重