import streamlit as st
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import os

RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__)) + '/runtime'
IMAGE_DIR = RUNTIME_DIR + '/images'
VIDEO_DIR = RUNTIME_DIR + '/videos'

if 'run_button' in st.session_state and st.session_state.run_button == True:
    st.session_state.running = True
else:
    st.session_state.running = False

@st.cache_resource
def init_model():
    return pipeline(task="image-to-video", model='damo/i2vgen-xl', model_revision='v1.1.5', device='cuda:0')

class App:
    def __init__(self):
        self.init_dir()
        self.image_to_video_pipe = init_model()
    
    def init_dir(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(VIDEO_DIR, exist_ok=True)

    def image_to_video(self, saved_image_path, text_in):
        if not os.path.exists(saved_image_path):
            raise st.error('请上传图片或等待图片上传完成')
        print(saved_image_path)
        output_video_path = self.image_to_video_pipe(saved_image_path, caption=text_in)[OutputKeys.OUTPUT_VIDEO]
        print(output_video_path)
        return output_video_path
    
    def run(self):
        st.title('I2VGen-XL')
        st.markdown('''
        I2VGen-XL可以根据用户输入的静态图像和文本生成目标接近、语义相同的视频，生成的视频具高清(1280 * 720)、宽屏(16:9)、时序连贯、质感好等特点。

        I2VGen-XL can generate videos with similar contents and semantics based on user input static images and text. The generated videos have characteristics such as high-definition (1280 * 720), widescreen (16:9), coherent timing, and good texture.
        ''')

        text_in = st.text_area('文本描述', height=100)
        image_in = st.file_uploader('图片输入', type='jpg', accept_multiple_files=False)
        saved_image_path = None
        if image_in is not None:
            # Save image to file
            saved_image_path = os.path.join(IMAGE_DIR, image_in.name)
            with open(saved_image_path, 'wb') as f:
                f.write(image_in.getvalue())
            st.image(image_in, caption='上传的图片', use_column_width=True)
        else:
            st.warning('请上传图片或等待图片上传完成')

        run_button = st.button("生成视频", disabled=st.session_state.running, key='run_button')
        if run_button:
            st.session_state.running = True
            output_video_path = self.image_to_video(saved_image_path, text_in)
            if output_video_path:
                st.session_state.video = output_video_path
            else:
                st.error("Failed to generate video.")
            st.session_state.running = False
        
        
        if "video" in st.session_state:
            st.video(st.session_state.video, caption='生成的视频', start_time=0)


        st.markdown('注：如果生成的视频无法播放，请尝试升级浏览器或使用chrome浏览器。')

if __name__ == '__main__':
    app = App()
    app.run()