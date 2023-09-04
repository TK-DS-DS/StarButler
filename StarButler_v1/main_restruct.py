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


    result = model.transcribe("mlp.wav", language="Chinese")
    result_transcript = result["text"]
    print(result_transcript)
    return result_transcript

#语音合成与朗诵 固定
# 传入进行朗诵，改进为线程
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
            if(tts_thread.is_alive()):
                tts_thread.stop_playback()
            text=wakeup_detected_callback()

            user_input = text  # 获取用户输入
            chat_history.append(f"User: {user_input}")
            # 将用户输入和ChatGPT的对话历史合并为一个字符串
            chat_text = "\n".join(chat_history)

            # 调用ChatGPT获取助手回复
            response = openai.ChatCompletion.create(
                stream=True,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": chat_text}]
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
            tts_thread = TextToSpeechThread(text_whole)
            tts_thread.start()

            chat_history.append(f"chatGPT: {text_whole}")

if __name__ == '__main__':
    main()