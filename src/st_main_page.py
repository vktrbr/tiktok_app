import os

import dotenv
import streamlit as st

dotenv.load_dotenv()


class MainPage:
    SING_IN_CLICK = 'sign_in_click'
    SING_UP_CLICK = 'sign_up_click'
    INIT = 'init'
    SINGED = 'signed'
    SINGED_UP = 'signed_up'

    def __init__(self):
        self._main_container: st.delta_generator.DeltaGenerator = st.empty()
        self.__main_page_state = self.INIT

    def main_page(self):

        self._main_container = st.empty()

        with self._main_container.container():
            st.markdown("# Welcome to TikTokForecast")

            st.markdown("Below is a short explanation of how the application works")
            _, image_placeholder, _ = st.columns((1, 4, 1), gap='medium')
            image_placeholder.image(os.getenv('IMAGES_DIR') + 'circle_work.png')

            col1, col2 = st.columns(2, gap='medium')
            with col1:
                sign_in_button = st.button("Sign In", use_container_width=True)
                if sign_in_button:
                    self.__main_page_state = self.SING_IN_CLICK

            with col2:
                sign_up_button = st.button("Sign Up", use_container_width=True)
                if sign_up_button:
                    self.__main_page_state = self.SING_UP_CLICK

            st.markdown("# Instructions")
            st.markdown("To use TikTokForecast, sign up for an account or sign in if you already have one. "
                        "Once signed in, you will be able to access the forecasting tool.")

        return self.__main_page_state


    def clear_main_page(self):
        self._main_container.empty()
        del self._main_container

    @property
    def state(self):
        return self.__main_page_state

    @state.setter
    def state(self, state):
        self.__main_page_state = state
