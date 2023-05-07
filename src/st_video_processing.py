import numpy as np
import plotly.graph_objs as go
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from .db_connect import PGConnection
from .ml_model_processing import *

dotenv.load_dotenv()


class VideoProcessing:
    DEFAULT = 'init'
    VIDEO_UPLOADED = 'video_uploaded'
    VIDEO_EVALUATED = 'video_evaluated'
    available_states = [DEFAULT, VIDEO_UPLOADED, VIDEO_EVALUATED]

    def __init__(self, pgc: PGConnection):
        self._main_container: [DeltaGenerator, None] = None
        self._input_form_container: [DeltaGenerator, None] = None
        self._video_container: [DeltaGenerator, None] = None
        self._message_container: [DeltaGenerator, None] = None
        self.__video_path: [str, None] = None
        self.__video_file: [None] = None
        self.__state = self.DEFAULT
        self.__predict_model = TikTokAnalytics()
        self.__pgc: PGConnection = pgc

    def video_input_form(self):
        """
        Provides a form for users to upload a mp4 video file and saves it to a local directory using its hash as the
        file name. Once the video is uploaded, changes the state of the object to VIDEO_UPLOADED.

        :return: None
        """

        # Create a placeholder for web elements
        self._main_container = st.empty()
        # Add a header to our container
        with self._main_container.container():
            st.markdown("<h2 style='text-align: center; color: grey;'>Let's make a prediction</h2>",
                        unsafe_allow_html=True)

            # Create a sub-container for the form

            self._input_form_container, self._video_container = st.columns((4, 1))  # st.empty()
            self._video_container: DeltaGenerator

            with self._input_form_container.form('Upload video'):
                # Allow users to upload a video file
                self.__video_file = st.file_uploader("Upload your video ", type=["mp4", 'mov'])

                # Create a button for uploading the file
                click_upload_button = st.form_submit_button('Upload', use_container_width=True)

            if not click_upload_button:
                self._video_container.image(Image.open(os.getenv('IMAGES_DIR') + 'download_pic.png'))
                st.stop()

            if self.__video_file is None:
                self._video_container.empty()
                # Display a warning message if no file was uploaded
                st.warning('Please upload a video file')
                st.stop()

            if click_upload_button and self.__video_file is not None:
                # Generate a unique filename for the uploaded video file
                self.__video_path = os.getenv('VIDEO_DIR') + f'/{hash(str(self.__video_file)) % 10 ** 10}.mp4'

                # Save the video file to the local directory
                with open(self.__video_path, 'wb') as file:
                    file.write(self.__video_file.read())

                # Update the state of the object
                self.state = self.VIDEO_UPLOADED

    def evaluate_form(self):

        _, self._message_container, _ = st.columns([1, 5, 1], gap='medium')
        self._video_container: DeltaGenerator
        self._message_container: DeltaGenerator

        with self._video_container.container():
            st.video(self.__video_file)

        with self._message_container.container():
            with st.spinner():
                predict_out = self.__predict_model(self.__video_path)[0]
                self.delete_video_file()

            chart_gauge, like_value = self.gen_chart_gauge(float(predict_out[0]), float(predict_out[1]))
            st.plotly_chart(chart_gauge, use_container_width=True)

        self.__pgc.log_video_values(self.__video_file.name, like_value)
        self.state = self.VIDEO_EVALUATED

    def get_main_container(self):
        return self._main_container

    def get_video_container(self):
        return self._video_container

    def get_message_container(self):
        return self._message_container

    def get_input_form_container(self):
        return self._input_form_container

    def clear_main_container(self):
        """
        Clears the main container of any web elements.

        :return: The empty container once all elements have been cleared.
        """
        return self._main_container.empty()

    def get_file_path(self):
        """
        Returns the path of the video file associated with the object.

        :return: A string representing the path of the video file. Returns None if no file is associated.
        """
        return self.__video_path

    def delete_video_file(self):
        """
        Deletes the video file associated with the object, if it exists. Sets the file path to None once done.

        :return: None
        """
        if self.__video_path is not None:
            os.remove(self.__video_path)
            self.__video_path = None

    @property
    def state(self) -> str:
        """
        Returns the current state of the object.

        :return: A string representing the current state of the object.
        """
        return self.__state

    @state.setter
    def state(self, state: str) -> None:
        """
        Sets the state of the object to the given state.

        :param state: A string representing the new state of the object.
        """
        if state in self.available_states:
            # self.clear_main_container()
            self.__state = state

    @staticmethod
    def gen_chart_gauge(dislike_value: float = 2, like_value: float = 1) -> tuple[go.Figure, float]:

        # normalize to [0, 1]
        dislike_value, like_value = 1 / (1 + np.exp((dislike_value, like_value)))
        dislike_value, like_value = np.array((dislike_value, like_value)) / (dislike_value + like_value)

        current_color = {0 < like_value * 100 <= 50: '#ffb5a7',
                         50 < like_value * 100 <= 100: '#52b69a'}[True]

        current_color_sub = {0 < like_value * 100 <= 50: '#fcd5ce',
                             50 < like_value * 100 <= 100: '#a5d2c3'}[True]

        chart_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=like_value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of success", 'font': {'size': 24}},
            delta={'reference': dislike_value * 100, 'increasing': {'color': current_color_sub},
                   'decreasing': {'color': current_color_sub}},
            gauge={
                'axis': {'range': [None, 100],
                         'tickwidth': 1,
                         'tickcolor': "#222222",
                         'tickvals': [0, 10, 25, 50, 75, 90, 100],
                         # 'text':{'color': '#FF0000'}
                         },
                'bar': {'color': "rgba(255, 255, 255, 0.85)"},
                'bgcolor': "white",
                'borderwidth': 10,
                'bordercolor': "rgba(0, 0, 0, 0)",
                'steps': [
                    {'range': [0, 10], 'color': '#ffb5a7'},
                    {'range': [10, 25], 'color': '#fcd5ce'},
                    {'range': [25, 50], 'color': '#f8edeb'},
                    {'range': [50, 75], 'color': '#cfe0d7'},
                    {'range': [75, 90], 'color': '#a5d2c3'},
                    {'range': [90, 100], 'color': '#52b69a'}],
                'threshold': {
                    'line': {'color': "rgba(0, 0, 0, 0)", 'width': 2},
                    'thickness': 0.65,
                    'value': like_value * 100}}))

        chart_gauge.update_layout(paper_bgcolor="rgba(0, 0, 0, 0)", font={'color': current_color})

        return chart_gauge, like_value
