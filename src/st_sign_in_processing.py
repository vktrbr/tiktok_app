from hashlib import sha256

import streamlit as st

from .db_connect import PGConnection


class SingInStates:
    INIT = 'init'
    SIGNED_IN = 'signed_in'
    NOT_SIGNED_IN = 'not_signed_in'
    available_states = [INIT, SIGNED_IN, NOT_SIGNED_IN]

    def __init__(self):
        self.__state: str = 'init'

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


class SignInForm(SingInStates):

    def __init__(self, pgc: PGConnection):
        super().__init__()

        self.__username = None
        self.__password_hash = None
        self.__correct_user = None
        self.__correct_combination = None
        self.__pgc = pgc

        self.__signin_container: [st.delta_generator.DeltaGenerator, None] = None

    def signin_form(self, username: str = None, password_hash: str = None) -> bool:
        """
        Displays a sign-in form for TikTokForecast and validates user input.

        :param username: the username or email address input by the user (default None)
        :param password_hash: the hashed password input by the user (default None)
        :return: True if the user's input is valid, False otherwise
        """
        self.state = 'not_signed_in'

        if username is not None and password_hash is not None:
            self.__username = username
            self.__password_hash = password_hash

        if self.__pgc.check_user_n_password(self.__password_hash):
            return True

        else:
            self.__username = None
            self.__password_hash = None

        self.__signin_container = st.empty()
        with self.__signin_container.container():
            st.markdown("<h2 style='text-align: center; color: grey;'>Sign in to TikTokForecast</h2>",
                        unsafe_allow_html=True)

            _, incorrect_container, _ = st.columns((1, 4, 1))
            incorrect_container: st.delta_generator.DeltaGenerator

            _, center, _ = st.columns((1, 4, 1))
            center: st.delta_generator.DeltaGenerator

            with center.form('Sign in form'):
                self.__username = st.text_input(label='Username or email address')
                self.__password_hash = sha256(
                    st.text_input(label='Password', type='password').encode('utf-8')).hexdigest()

                st.write('\n ')
                sign_in_submitted = st.form_submit_button('Sign in', use_container_width=0)

            if sign_in_submitted:
                self.__correct_user = self.__pgc.check_user(self.__username, self.__username)

                if not self.__correct_user:
                    incorrect_container.error('The username you entered does not exist.')
                    st.stop()

                self.__correct_combination = self.__pgc.check_user_n_password(self.__password_hash)
                if not self.__correct_combination:
                    incorrect_container.error('Incorrect password or username.')
                    st.stop()

            else:
                st.stop()

        self.__pgc.log_new_session()
        return True

    def correct_signed_in(self) -> bool:
        """
        Checks if the user is currently signed in. If the user is not signed in, it will redirect the user to the
        sign-in page.
        :return:
        """
        if self.__correct_user is not None and self.__correct_combination is not None:
            if self.__correct_user and self.__correct_combination:
                self.__signin_container.empty()
                self.state = 'signed_in'
                return True

        return False

    @property
    def username(self):
        return self.__username

    @property
    def password_hash(self):
        return self.__password_hash

    @property
    def user_id_session(self):
        return self.__pgc.user_id_session
