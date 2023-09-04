#当前实现语音与朗诵，下面实现 接口处理
#在v1 main2的基础上进行修改，实现连续对话功能
import json
import os
import threading

import pvporcupine
import pyaudio
import struct
import wave
import time
import math

import pygame
import requests
#whisper语音识别预加载
import whisper
from gtts import gTTS

model = whisper.load_model("small")


# 唤醒词模型文件
WAKEUP_MODEL = "你好_windows.ppn"
# 访问密钥
ACCESS_KEY = "xxxxxx"

# 初始化Porcupine
porcupine = pvporcupine.create(keyword_paths=[WAKEUP_MODEL], model_path='./porcupine_params_zh.pv', access_key=ACCESS_KEY)

# 关键词唤醒后的处理函数
def wakeup_detected_callback():
    print("Wakeup word detected!")

    # 初始化PyAudio
    pa = pyaudio.PyAudio()

    # 打开麦克风输入流
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length)

    frames = []  # 用于保存录音数据的列表
    timeout = time.time() + 1.0  # 超时时间，1秒

    # 用于跟踪声音是否连续存在的变量
    sound_detected = True

    print("Listening",end="")
    # 持续录音直到超时或检测不到声音
    while time.time() < timeout:
        print(".",end="")
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        frames.extend(pcm)

        # 获取当前PCM数据的音频振幅大小
        sum_squares = sum(abs(x) ** 2 for x in pcm)
        rms = math.sqrt(sum_squares / len(pcm))

        # 如果振幅大小小于10，输出1，并将声音连续存在的标志置为True
        if rms >= 400:
            timeout = time.time() + 1.0


    # 停止录音
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # 将PCM数据转换为bytes对象
    frames_bytes = b"".join(struct.pack("h", pcm_val) for pcm_val in frames)

    # 保存录音为wav文件
    with wave.open("./my_words.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(porcupine.sample_rate)
        wf.writeframes(frames_bytes)

    print("Recording saved as my_words.wav")
    result = model.transcribe("my_words.wav", language="Chinese")
    result_transcript = result["text"]
    print("User: ",result_transcript)
    return result_transcript

#语音合成与朗诵 固定
#朗诵线程
class TextToSpeechThread(threading.Thread):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.stop_playing = False  # 添加停止播放的标志
    def run(self):
        # 创建 gTTS 对象
        tts = gTTS(self.text, lang='zh')

        # 保存为临时音频文件
        audio_file = "output_audio.mp3"
        tts.save(audio_file)

        # 初始化 pygame
        pygame.init()

        # 初始化音频模块
        pygame.mixer.init()

        # 加载音频文件
        pygame.mixer.music.load(audio_file)

        # 设置音量（可选）
        pygame.mixer.music.set_volume(0.5)  # 0.0 到 1.0

        # 播放音频
        pygame.mixer.music.play()

        # 等待音频播放完毕
        while pygame.mixer.music.get_busy():
            if self.stop_playing:  # 如果标志为True，停止播放
                pygame.mixer.music.stop()
                break

        # 停止 pygame
        pygame.quit()

        # 删除临时音频文件
        os.remove(audio_file)

    def stop_playback(self):
        self.stop_playing = True  # 设置停止播放标志

# 请求openai，获取文本的embedding
def post_get_embedding(inputText):
    url = "https://api.openai.com/v1/embeddings"

    payload = json.dumps({
      "input": inputText,
      "model": "text-embedding-ada-002"
    })
    headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer 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'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # 解析响应的JSON数据
    response_data = json.loads(response.text)
    # 提取嵌入向量
    embeddings = response_data['data'][0]['embedding']
    # 打印嵌入向量
    # print(embeddings)
    return embeddings


# gpt接口调用
def main():
    tts_thread = None

    print("Listening for the wakeup word...")
    # 唤醒词模型文件
    WAKEUP_MODEL = "你好_windows.ppn"
    # 访问密钥
    ACCESS_KEY = "xxxxxx"

    # 初始化Porcupine
    porcupine = pvporcupine.create(keyword_paths=[WAKEUP_MODEL], model_path='./porcupine_params_zh.pv',
                                   access_key=ACCESS_KEY)

    # 初始化PyAudio
    pa = pyaudio.PyAudio()


    # Chroma initialization
    import chromadb
    chromaClient = chromadb.Client()
    chromaClient = chromadb.PersistentClient(path="./chromadb")
    chromaCollection = chromaClient.get_or_create_collection(name="myhelper-collection")
    # 关联度，涉及到长期记忆提取几个事件
    chromaSettingNum=2
    #关联个数
    chromaSettingMinCos=0.2



    # 打开麦克风输入流
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length)

    # 初始化ChatGPT会话
    import openai
    openai.api_key = "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"
    chat_history = []
    # chatHistoryWithMemory=[]



    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm)
        #检测到关键词
        if keyword_index >= 0:
            if tts_thread is not None and tts_thread.is_alive():
                tts_thread.stop_playback()

            text=wakeup_detected_callback()
            chatEachUser = "User:"+text

            chatHistoryWithMemory=chat_history[:]
            chatHistoryWithMemory.append(chatEachUser)



            #chroma 查询
            # chroma_query = str(input("chroma query"))
            query_embeddings = post_get_embedding(chatEachUser)
            chromaQueryResults = chromaCollection.query(
                query_embeddings=[query_embeddings],
                n_results=2,
                # where={"metadata_field": "is_equal_to_this"}, # optional filter
                # where_document={"$contains":"search_string"}  # optional filter
            )
            #进行关联度判定和引入
            # 创建一个新的distances列表，用于存储不大于0.2的元素
            chatQueryMemory = ""

            # 遍历distances列表，查找小于0.2的元素
            for i, distance_list in enumerate(chromaQueryResults["distances"]):
                for j, distance in enumerate(distance_list):
                    if distance < 0.4:
                        # 如果距离小于0.2，将documents和metadatas添加到filtered_data
                        chatQueryMemory += f"{chromaQueryResults['metadatas'][i][j]}"
                        chatQueryMemory += f"{chromaQueryResults['documents'][i][j]};"
            print(chatQueryMemory)
            # chatHistoryWithMemory.insert(0, chatQueryMemory)

            # 将用户输入和ChatGPT的对话历史合并为一个字符串
            chat_text = "\n".join(chatHistoryWithMemory)
            # 调用ChatGPT获取助手回复
            response = openai.ChatCompletion.create(
                stream=True,
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "请你作为我的助手，为我提供各种包含但不限于信息查询等功能"},
                    {"role": "user", "content": "这是我们的历史对话，User是我，Assistant是你，你的回答只需要说你自己的话就行。"+chatQueryMemory},
                    {"role": "user", "content": chat_text}]
            )

            text_whole = ""
            for chunk in response:
                if "choices" in chunk and chunk["choices"]:
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        text_whole = text_whole + content
                        print(content, end="")

            # 加入长期记忆功能





            if text_whole.startswith("Assistant: "):
                text_whole = text_whole[len("Assistant: "):]
            if text_whole.startswith("AI: "):
                text_whole = text_whole[len("AI: "):]
            tts_thread = TextToSpeechThread(text_whole)
            tts_thread.start()


            chatEachAssistant="Assistant: " + text_whole
            chatEach=chatEachUser+" "+chatEachAssistant
            print("chateach: ",chatEach)
            chat_history.append(chatEachUser)
            chat_history.append(chatEachAssistant)
            print("chatHistory: ",chat_history)
            # print("chroma history:", chatHistoryWithMemory)

            # 生成唯一时间戳，作为ids
            import datetime
            # 获取当前日期和时间
            current_datetime = datetime.datetime.now()
            # 将日期和时间格式化为字符串，包括毫秒
            chromaAddIds = str(current_datetime.strftime("%Y%m%d%H%M%S%f"))
            chromaAddMetadatas=str(current_datetime.strftime("%Y年%m月%d日%H时%M分%S秒发生"))

            # add
            chromaAddEmbeddings = post_get_embedding(chatEach)
            chromaCollection.add(
                documents=[chatEach],
                embeddings=[chromaAddEmbeddings],
                metadatas=[{"time":str(chromaAddMetadatas)}],
                ids=[str(chromaAddIds)]
            )
            print("Finish chromaAdd")

            #
if __name__ == '__main__':
    main()