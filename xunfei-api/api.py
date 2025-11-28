from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# 星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v4.0/chat'
# 星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = '3322e568'
SPARKAI_API_SECRET = 'ZjhjNjU5ZDYyMTIxMjNmOTk0MjU5MjUz'
SPARKAI_API_KEY = 'a7555bcb3e1135275dfb32ec7a83cca0'
# 星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = '4.0Ultra'
tran = [
"与主流BioNER不同，我们将整个文档视为一组示例，从而能够整合更远距离的上下文信息。Unlike mainstream BioNER, we consider the entire document as a set of examples, which"

]

if __name__ == '__main__':
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    for s in tran:
        # 输入内容
        system = "将下面句子的中文翻译成英文，如果已给英文，则在英文的基础上补充，给我返回整个句子。要求：要专业，不啰嗦，学术，\n"
        content = "content: " + s
        messages = [ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=content), ]
        handler = ChunkPrintHandler()
        a = spark.generate([messages], callbacks=[handler])
        a = a.generations[0]
        a = a[0]
        a = a.message
        a = a.content
        print(a)
