#当前实现语音与朗诵，下面实现 接口处理
#在v1 main2的基础上进行修改，实现连续对话功能
import os
import threading

import pvporcupine
import pyaudio
import struct
import wave
import time
import math

import pygame
#whisper语音识别预加载
import whisper
from gtts import gTTS

model = whisper.load_model("small")


# 唤醒词模型文件
WAKEUP_MODEL = "你好_windows.ppn"
# 访问密钥
ACCESS_KEY = "xxx"

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

    # 持续录音直到超时或检测不到声音
    while time.time() < timeout:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        frames.extend(pcm)

        # 获取当前PCM数据的音频振幅大小
        sum_squares = sum(abs(x) ** 2 for x in pcm)
        rms = math.sqrt(sum_squares / len(pcm))

        # 如果振幅大小小于10，输出1，并将声音连续存在的标志置为True
        if rms >= 400:
            timeout = time.time() + 1.0
            print("Output 1")

    # 停止录音
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # 将PCM数据转换为bytes对象
    frames_bytes = b"".join(struct.pack("h", pcm_val) for pcm_val in frames)

    # 保存录音为wav文件
    with wave.open("./mlp.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(porcupine.sample_rate)
        wf.writeframes(frames_bytes)

    print("Recording saved as mlp.wav")
    # # 调取科大讯飞的语音识别
    # result_transcript = recognize_speech("mlp.wav")
    # print("讯飞识别：",result_transcript)

    result = model.transcribe("mlp.wav", language="Chinese")
    result_transcript = result["text"]
    print(result_transcript)
    return result_transcript

#对话流录制与识别

#用户经历库提取

#知识库提取

#接口能力层提取

#对话gpt
# 助手
def gpt_input(input_text):
    import openai

    openai.api_key = "xxxxxx"

    from datetime import datetime
    # 获取当前日期和时间
    now = datetime.now()
    # 将日期和时间格式化为字符串
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")



    completion_stream=openai.ChatCompletion.create(
        stream=True,
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "user", "content": "下面所有对话都讲中文,你将作为我的私人助手，为我提供各种服务。每次交流如果了解我的指令，需要回复我'好的'或者其他肯定性的回答，并回复我你将要执行的动作。"},
            #       #格式调教
            #       {"role": "user",
            #        "content": "我询问你的可能包含一些实时性的数据或操作，这些必须调用接口来获取数据或执行操作。在正常的交流之外，如果涉及调用接口的回答必须在回复的末尾，使用数组和json格式回答。"},
            #
            #
            #       #api能力介绍层
            #       {"role": "user", "content": '有关天气查询，可以使用 "#[{plugin:"weather",plugin-data:[{place:"北京",datetime:"2023年6月7日"}]}]#" 代表查询2023年6月7日北京的天气，place和datetime为必要参数，返回天气情况。'
            #                                   '当我询问天气相关的问题时，如"明天的天气怎么样?"，你可以回答"好的，我将查询明天的天气情况。#[{plugin:"weather",plugin-data:[{place:"北京",datetime:"2023年6月9日"}]}]#" 请不要忘记数组和json'},
            #
            #       #个人情况说明层
            #       {"role": "user",
            #        "content": "我是TKDS，你可以称呼我为主人、先生。我现在位于西安。下面的对话中如果我没有说明地区，则地点默认为我现在的地方，直接发送相关信息。现在的时间是"+date_string+"。"},
                  {"role": "user", "content": input_text}],
    )
    text_whole=""
    for chunk in completion_stream:
        if "choices" in chunk and chunk["choices"]:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                text_whole=text_whole+content
                print(content, end="")
                # process_streaming_data(content)

    return text_whole
#gpt返回信息解析
# 初始化缓冲区和JSON数组部分列表
gptstream_buffer = ""
gptstream_jsondata = []
def process_streaming_data(new_data):
    start_tag = "#["
    end_tag = "]#"

    global gptstream_buffer
    global gptstream_jsondata

    gptstream_buffer += new_data

    # 检查是否存在完整的开始标签和结束标签
    start_index = gptstream_buffer.find(start_tag)
    end_index = gptstream_buffer.find(end_tag)

    while end_index != -1:
        if start_index != -1 and start_index < end_index:
            # 输出文本部分
            print("文本部分:", gptstream_buffer[:start_index])
            text_part=gptstream_buffer[:start_index]
            # 截取已处理的部分
            gptstream_buffer = gptstream_buffer[start_index + len(start_tag):]
        else:
            # 提取JSON数组部分并处理
            gptstream_jsondata.append(gptstream_buffer[:end_index])
            # 截取已处理的部分
            gptstream_buffer = gptstream_buffer[end_index + len(end_tag):]
            # 输出JSON数组部分
            print("JSON数组部分:", gptstream_jsondata)
            json_part=gptstream_jsondata
            # 重置gptstream_jsondata列表
            gptstream_jsondata = []

        # 继续寻找下一个开始标签和结束标签的位置
        start_index = gptstream_buffer.find(start_tag)
        end_index = gptstream_buffer.find(end_tag)

    # return text_part, json_part

#json与text分离
def extract_text_and_json(input_string):
    start_tag = "#["
    end_tag = "]#"

    start_index = input_string.find(start_tag)
    end_index = input_string.find(end_tag)

    text_part = ""
    json_part = ""

    if start_index != -1 and end_index != -1:
        text_part = input_string[:start_index]
        json_part = input_string[start_index + len(start_tag):end_index]

    return text_part, json_part

#语音合成与朗诵 固定
# 传入进行朗诵，改进为线程
class TextToSpeechThread(threading.Thread):
    def __init__(self, text):
        super().__init__()
        self.text = text

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
            pass

        # 停止 pygame
        pygame.quit()

        # 删除临时音频文件
        os.remove(audio_file)

# gpt接口调用
def main():

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

    # 打开麦克风输入流
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length)

    # 初始化ChatGPT会话
    import openai
    openai.api_key = "xxxxxx"
    chat_history = []



    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm)
        #检测到关键词
        if keyword_index >= 0:
            text=wakeup_detected_callback()
            # result_gpt = gpt_input(text)



            user_input = text  # 获取用户输入
            chat_history.append(f"User: {user_input}")

            # 将用户输入和ChatGPT的对话历史合并为一个字符串
            chat_text = "\n".join(chat_history)
            # print(chat_text)

            # 调用ChatGPT获取助手回复
            response = openai.ChatCompletion.create(
                stream=True,
                model="gpt-3.5-turbo",
                messages=[
                    # {"role": "system", "content": "下面所有对话都讲中文,你将作为我的私人助手，为我提供各种服务。每次交流如果了解我的指令，需要回复我'好的'或者其他肯定性的回答，并回复我你将要执行的动作。"},
                    #       {"role": "user",
                    #        "content": "我询问你的可能包含一些实时性的数据或操作，这些必须调用接口来获取数据或执行操作。在正常的交流之外，如果涉及调用接口的回答必须在回复的末尾，使用数组和json格式回答。"},
                    #       # api能力介绍层
                    #       {"role": "user",
                    #        "content": '有关天气查询，可以使用 "#[{plugin:"weather",plugin-data:[{place:"北京",datetime:"2023年6月7日"}]}]#" 代表查询2023年6月7日北京的天气，place和datetime为必要参数，返回天气情况。'
                    #                   '当我询问天气相关的问题时，如"明天的天气怎么样?"，你可以回答"好的，我将查询明天的天气情况。#[{plugin:"weather",plugin-data:[{place:"北京",datetime:"2023年6月9日"}]}]#" 请不要忘记数组和json'},



                          {"role": "user", "content": chat_text}]
            )

            # tts_thread = TextToSpeechThread("")
            # tts_thread.start()

            text_whole = ""
            for chunk in response:
                if "choices" in chunk and chunk["choices"]:
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        # while tts_thread.is_alive():
                        #     time.sleep(0.1)  # 等待一小段时间，避免过多占用CPU资源
                        # tts_thread = TextToSpeechThread(content)
                        # tts_thread.start()
                        text_whole = text_whole + content
                        print(content, end="")
                        process_streaming_data(content)


            # assistant_response = response.choices[0].message['content']
            # print(assistant_response)
            #
            tts_thread = TextToSpeechThread(text_whole)
            tts_thread.start()
            # while tts_thread.is_alive():
            #     time.sleep(0.1)  # 等待一小段时间，避免过多占用CPU资源
            # tts_thread = TextToSpeechThread(text_whole)
            # tts_thread.start()


            # 解析助手回复...

            # text_part, json_part = extract_text_and_json(assistant_response)
            # print("print", text_part, json_part)
            # text_to_speech_and_play(text_part)

            chat_history.append(f"Assistant: {text_whole}")




            # print("result", result_gpt)
            #识别结果分离
            # text_part, json_part = extract_text_and_json(result_gpt)
            # print("print",text_part,json_part)
            # text_to_speech_and_play(text_part)











if __name__ == '__main__':
    main()