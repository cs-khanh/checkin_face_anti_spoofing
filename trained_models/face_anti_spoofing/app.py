import os
import time
import sys
import cv2
import numpy as np
import gradio as gr
import IADG
import SASF

ModelD  = IADG.aFaceDetect()
Model1 = SASF.aSASF     (threshold=0.0094)
Model2 = IADG.aSpoofONNX('modelrgb',threshold=0.0553) # 10% 31.785% Threshold: 0.46294 20% 20.530% Threshold: 0.28085 30% 12.208% Threshold: 0.15554 40% 6.7358% Threshold: 0.08302 50% 4.5046% Threshold: 0.05533
Model3 = IADG.aSpoof    ('ICM2O',threshold=0.9980) # 10% 60.9210 Threshold: 0.9991  20% 41.1890 Threshold: 0.9980  30% 25.3207 Threshold: 0.9954  40% 13.6156 Threshold: 0.9871  50%  5.9737 Threshold: 0.9432
Model4 = IADG.aSpoof    ('IOM2C',threshold=0.9944) 


def prob(p,thre):
    if p<thre:
        return (thre-p)/thre
    return (p-thre)/(1-thre)

def run_image(input_image,text): # input_image - RGB
    thre=text.split('\n')
    Model1.threshold =float(thre[0])
    Model2.threshold =float(thre[1])
    Model3.threshold =float(thre[2])
    Model4.threshold =float(thre[3])
    bboxes,landmarks = ModelD(input_image)
    if len(landmarks)<1:
        return input_image,input_image,'Лицо не обнаружено или обнаружено несколько лиц'
    spoof1,spoof_prob1,img1 = Model1(input_image,bboxes[0],landmarks[0])
    spoof2,spoof_prob2,img2 = Model2(input_image,bboxes[0],landmarks[0])
    spoof3,spoof_prob3,img3 = Model3(input_image,bboxes[0],landmarks[0])
    spoof4,spoof_prob4,img4 = Model4(input_image,bboxes[0],landmarks[0])
    names=['Реальное фото','Подделка']
    text =f'SASF :\t P={spoof_prob1:.4f} ({names[spoof1]}). Уверенность: {prob(spoof_prob1,Model1.threshold)}\n'
    text+=f'FLRGB:\t P={spoof_prob2:.4f} ({names[spoof2]}). Уверенность: {prob(spoof_prob2,Model2.threshold)}\n'
    text+=f'ICM2O:\t P={spoof_prob3:.4f} ({names[spoof3]}). Уверенность: {prob(spoof_prob3,Model3.threshold)}\n'
    text+=f'IOM2C:\t P={spoof_prob4:.4f} ({names[spoof4]}). Уверенность: {prob(spoof_prob4,Model4.threshold)}\n'
    return img2,img3,text

def demo():
    with gr.Blocks(title="Тестирование детектора подделки лиц. Версия 2.") as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown('<h1><center>Тестирование детекторов подделки лиц</center></h1>')
                gr.Markdown('''
* Модель 1. Silent-Face-Anti-Spoofing (SASF). <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
* Модель 2. 人脸活体检测模型-RGB (**FLRGB**). <https://modelscope.cn/models/iic/cv_manual_face-liveness_flrgb>
* Модель 3. Instance-Aware Domain Generalization for Face Anti-Spoofing (**ICM2O**). <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf>
* Модель 4. Instance-Aware Domain Generalization for Face Anti-Spoofing (**IOM2C**). <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf>
* Интерпретация результата. Чем больше P порогового значения, тем больше уверенность, что фото - подделка.
* Пороговые значения нужно подобрать в ходе тестирования.
''')
                            
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image = gr.Image(type='numpy', label='Поддельное или реальное фото человека')
                with gr.Row():
                    input_text = gr.Textbox(label='Пороговые значения', value='0.0094\n0.2808\n0.9980\n0.9944', lines=4)
                with gr.Row():
                    submit = gr.Button('Сделать прогноз')
                    clear = gr.Button('Очистка')
                # out_download = gr.File(visible=False)
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                       output_image1 = gr.Image(type='numpy', label='Input image 1')
                    with gr.Column():
                       output_image2 = gr.Image(type='numpy', label='Input image 2')
                with gr.Row():
                    output_text=gr.TextArea( label="Результаты прогноза", lines=6,value="",)

        submit.click(run_image,[image,input_text],[output_image1,output_image2,output_text])
        clear.click(lambda: [None]*4, None,
                            [image, output_image1,output_image2,output_text])
        demo.launch()

if __name__ == '__main__':
    demo()
