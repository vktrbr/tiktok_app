from hashlib import sha256
from time import sleep

import streamlit as st

from .db_connect import PGConnection


class SingUpStates:
    INIT = 'init'
    SIGNED_UP = 'signed_up'
    NOT_SIGNED_UP = 'not_signed_up'
    available_states = [INIT, SIGNED_UP, NOT_SIGNED_UP]

    def __init__(self):
        super().__init__()
        self.__state = self.INIT
        self.available_states = [self.INIT, self.SIGNED_UP, self.NOT_SIGNED_UP]

    @property
    def state(self) -> str:
        """
        Returns current state
        :return: str with state name
        """
        return self.__state

    @state.setter
    def state(self, state: str):
        """
        Set specific state
        :param state:
        """
        assert state in self.available_states, f'Available states: {self.available_states}'
        self.__state = state


class SignUpForm(SingUpStates):

    def __init__(self, pgc: PGConnection):
        super().__init__()

        self.__username = None
        self.__password_hash = None
        self.__correct_user = None
        self.__correct_combination = None
        self.__pgc = pgc

        self.__signin_container: [st.delta_generator.DeltaGenerator, None] = None

    def sign_up_form(self) -> bool:
        """
        Displays a form for creating a new user and saves the user's information to the database.
        """
        self.state = self.NOT_SIGNED_UP

        self.__signin_container = st.empty()
        with self.__signin_container.container():
            st.markdown("<h2 style='text-align: center; color: grey;'>Create your TikTokForecast account</h2>",
                        unsafe_allow_html=True)

            _, incorrect_container, _ = st.columns((1, 4, 1))
            incorrect_container: st.delta_generator.DeltaGenerator

            _, center, _ = st.columns((1, 4, 1))
            center: st.delta_generator.DeltaGenerator

            with center.form('Sign up form'):
                self.__username = st.text_input(label='Username', key='signup_username')
                email_input = st.text_input(label='Email', key='signup_email')
                password_input = st.text_input(label='Password', type='password', key='signup_password')
                confirm_password = st.text_input(label='Confirm password', type='password',
                                                 key='confirm_signup_password')

                if password_input != confirm_password:
                    st.error('Passwords do not match.')

                self.__password_hash = sha256(password_input.encode('utf-8')).hexdigest()

                st.write('\n ')
                sign_up_submitted = st.form_submit_button('Create account', use_container_width=0)

            if sign_up_submitted and password_input == confirm_password:
                self.__correct_user = self.__pgc.check_user(self.__username, self.__username)

                if self.__correct_user:
                    incorrect_container.error('Username already exists.')
                    st.stop()

                self.__pgc.write_user(self.__username, email_input, self.__password_hash)
                center.success('Account created successfully!')
                sleep(3)
                return True
            else:
                st.stop()

        return False
