from src.st_main_page import *
from src.st_sign_in_processing import *
from src.st_sign_up_precessing import *
from src.st_video_processing import *

dotenv.load_dotenv()

os.chdir(os.getenv('MAIN_PATH'))


class AppMainBody:

    def __init__(self):
        self.__pgc = PGConnection()
        self._signin_process = SignInForm(self.__pgc)
        self._signup_process = SignUpForm(self.__pgc)
        self._evaluate_process = VideoProcessing(self.__pgc)
        self._main_page_process = MainPage()

        self._low_attention = -2
        self._high_attention = 2
        self._rgw = (0.4, 0.4, 0.25)

        st.session_state['signed_in']: str = SignInForm.INIT
        st.session_state['signed_up']: str = 'default'
        st.session_state['video_eval']: str = VideoProcessing.DEFAULT

    def conduct_process_signing_in(self):

        if 'signed_in' not in st.session_state:
            st.session_state['signed_in'] = SignInForm.INIT

            self.__pgc.log_new_session()
            self._signin_process.state = SignInForm.SIGNED_IN

        if self._signin_process.state in (SignInForm.INIT, SignInForm.NOT_SIGNED_IN):
            # provide a process of signing in
            self._signin_process.signin_form(self._signin_process.username, self._signin_process.password_hash)
            if self._signin_process.correct_signed_in():
                self.__pgc.log_new_session()
                st.session_state['signed_in'] = SignInForm.SIGNED_IN

        if self._signin_process.state in (SignInForm.SIGNED_IN,):
            self.__pgc.log_new_session()
            st.session_state['signed_in'] = SignInForm.SIGNED_IN

    def conduct_process_video_evaluating(self, **kwargs):

        if self._main_page_process.state in (MainPage.SINGED,):
            self._evaluate_process.video_input_form()

        if self._evaluate_process.state in (VideoProcessing.VIDEO_UPLOADED,):
            self._evaluate_process.evaluate_form(**kwargs)

    def conduct_process_signing_up(self):

        if self._signup_process.state in (SignUpForm.INIT, SignUpForm.NOT_SIGNED_UP):
            signed_up = self._signup_process.sign_up_form()
            if signed_up:
                self._signup_process.state = SignUpForm.SIGNED_UP
                self._main_page_process.state = MainPage.SINGED_UP
                st.experimental_rerun()

    def run(self):

        if self._main_page_process.state in (MainPage.INIT,):

            clicked_button = self._main_page_process.main_page()

            if clicked_button == MainPage.SING_IN_CLICK:
                self._main_page_process.clear_main_page()
                self.conduct_process_signing_in()
                self._main_page_process.state = MainPage.SINGED
                st.experimental_rerun()

            elif clicked_button == MainPage.SING_UP_CLICK:
                self._main_page_process.clear_main_page()
                self.conduct_process_signing_up()

        elif self._main_page_process.state in (MainPage.SINGED,) and self.__pgc.user_id_session is not None:

            with st.sidebar:
                self._low_attention = st.slider('low attention', -10.0, 0.0, -2.0)
                self._high_attention = st.slider('high attention', 0.0, 10.0, 2.0)
                r = st.slider('red threshold', 0.0, 1.0, 0.25)
                g = st.slider('green threshold', 0.0, 1.0, 0.25)
                w = st.slider('white threshold', 0.0, 1.0, 0.25)
                self._rgw = (r, g, w)

            self.conduct_process_video_evaluating(low_attention=self._low_attention,
                                                  high_attention=self._high_attention,
                                                  rgw_thr=self._rgw)

        elif self._main_page_process.state not in (MainPage.INIT, MainPage.SINGED):

            if self._main_page_process.state in (MainPage.SING_IN_CLICK,):
                self.conduct_process_signing_in()
                self._main_page_process.state = MainPage.SINGED
                st.experimental_rerun()

            elif self._main_page_process.state in (MainPage.SING_UP_CLICK,):
                self.conduct_process_signing_up()

    def get_main_page_state(self):
        return self._main_page_process.state


if 'app' not in st.session_state:
    st.session_state['app']: AppMainBody = AppMainBody()

elif st.session_state['app'].get_main_page_state() == MainPage.SINGED_UP:
    st.session_state['app']: AppMainBody = AppMainBody()

st.session_state['app'].run()
