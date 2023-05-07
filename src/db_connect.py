from __future__ import annotations

import hashlib
import os
from datetime import datetime

import dotenv
import psycopg2
from psycopg2.extensions import connection, cursor

dotenv.load_dotenv()


class PGConnection:

    def __init__(self,
                 host: str = os.getenv('PG_TT_WEBAPP_HOST'),
                 database: str = "default_db",
                 user: str = os.getenv('PG_TT_USER'),
                 password: str = os.getenv('PG_TT_WEBAPP_PASSWORD')
                 ):
        self.__db_host = host
        self.__db = database
        self.__db_user = user
        self.__db_password = password

        self.__conn: [connection, None] = None
        self.__cursor: [cursor, None] = None

        self.__user_name: str | None = None
        self.__user_email: str | None = None
        self.__user_id_user: str | None = None
        self.__user_id_session: str | None = None
        self.__user_password_hash: str | None = None
        self.__user_video_id: str | None = None
        self.__user_now: datetime = datetime.now()

    def __create_connection(self) -> connection:
        conn = psycopg2.connect(
            host=self.__db_host,  # os.environ['PG_TT_WEBAPP_HOST'],
            database=self.__db,
            user=self.__db_user,
            password=self.__db_password  # os.environ['PG_TT_WEBAPP_PASSWORD']
        )
        return conn

    def __execute_query(self, query: str) -> str:
        """

        :param query:
        :return:
        """

        self.__cursor.execute(query)
        record = self.__cursor.fetchone()
        return record

    def close(self):
        self.__cursor.close()
        self.__conn.close()

    def open(self):
        self.__conn = self.__create_connection()
        self.__cursor = self.__conn.cursor()

    def __stable_hash_for_username(self) -> str:
        """
        Calculates a stable hash value for the given username.

        :return: A stable hash value for the self.__user_name.
        """
        # Calculate the MD5 hash of the username and convert it to hexadecimal format
        md5_hash = hashlib.md5(self.__user_name.encode('utf-8')).hexdigest()

        # Convert the hexadecimal hash to an integer value
        hash_int = int(md5_hash, 16)

        # Return the float value as a string with 6 decimal places
        return f"{hash_int % 1_000_000_000_000}"

    def check_user(self, username: str = None, email: str = None) -> bool:
        """
        Checks if a user exists in the database with the given username or email.

        :param username: (optional) The username of the user to check.
        :param email: (optional) The email address of the user to check.
        :return: True if the user exists, False otherwise.
        """
        self.__user_name = username if username is not None else self.__user_name
        self.__user_email = email if email is not None else self.__user_email

        self.open()
        if self.__user_name is None and self.__user_email is None:
            # If both username and email are None, the function cannot check for the existence of a user.
            # Return False to indicate that no user exists.
            return False

        query = """
                    select exists (
                        select 1 from users.user 
                        where txt_username = %s or txt_email = %s
                    )
                """
        # Execute the query with parameters
        self.__cursor.execute(query, (self.__user_name, self.__user_email))
        # Fetch the result
        result = self.__cursor.fetchone()[0]
        self.close()

        if result:
            self.__user_name = username

        return result

    def check_user_n_password(self, password_hash: str = None) -> bool:
        """
        Checks if a user with the given username or email and password exists in the database.

        :param password_hash: The hashed password of the user to check.
        :return: True if the user exists and the password matches, False otherwise.
        """
        self.__user_password_hash = password_hash if password_hash is not None else self.__user_password_hash

        self.open()
        # Check if username or email is provided
        if self.__user_name is None and self.__user_email is None:
            # If not provided, return False
            return False

        # Define the query with placeholders for parameters to prevent SQL injection
        query = """
            select exists (
                select 1 
                from users.user
                where (txt_username = %s or txt_email = %s) and hash_pass = %s
            )
        """
        # Execute the query with parameters
        self.__cursor.execute(query, (self.__user_name, self.__user_email, self.__user_password_hash))
        # Fetch the result
        result = self.__cursor.fetchone()[0]
        self.close()
        return result

    def write_user(self, username: str = None, email: str = None, password_hash: str = None) -> bool:
        """
        Inserts a new user record into the database.

        :param username: The username of the new user.
        :param email: The email address of the new user.
        :param password_hash: The hashed password of the new user.
        :return: True if the user was successfully inserted, False otherwise.
        """
        self.__user_name = username if username is not None else self.__user_name
        self.__user_email = email if email is not None else self.__user_email
        self.__user_password_hash = password_hash if password_hash is not None else self.__user_password_hash

        if self.check_user(self.__user_name, self.__user_email):
            return False

        self.open()

        # Check if username or email is provided
        query = """
                    insert into users.user (id_user, txt_username, hash_pass, dt_registration, txt_email)
                    values (%s, %s, %s, %s, %s);
                """
        # Execute the query with parameters
        self.__cursor.execute(
            query,
            (
                f'u-{self.__stable_hash_for_username()}',
                self.__user_name,
                self.__user_password_hash,
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                self.__user_email
            )
        )

        # Get the number of rows affected by the query
        num_rows_affected = self.__cursor.rowcount

        # Commit the changes and close the connection
        self.__conn.commit()
        self.close()

        # Return True if at least one row was affected, False otherwise
        return num_rows_affected > 0

    def log_new_session(self) -> bool:
        """
        Inserts a new session record into the database.

        :return: True if the session was successfully inserted, False otherwise.
        """
        self.open()
        if self.__user_name is None and not self.check_user_n_password(self.__user_password_hash):
            return False

        if self.__user_id_session is not None:
            return False

        # Define the query with placeholders for parameters to prevent SQL injection
        query = """
            insert into users.session (id_user, id_session, dtime_session)
            values (%s, %s, %s);
        """
        # Execute the query with parameters
        now = datetime.now()
        self.__user_id_session = f'u-{self.__stable_hash_for_username()}-{now.day}-' \
                                 f'{now.hour}-{now.minute}-{now.second}-{now.microsecond}'
        self.__cursor.execute(
            query,
            (
                f'u-{self.__stable_hash_for_username()}',
                self.__user_id_session,
                now.strftime('%Y-%m-%d %H:%M:%S')
            )
        )

        # Get the number of rows affected by the query
        num_rows_affected = self.__cursor.rowcount

        # Commit the changes and close the connection
        self.__conn.commit()
        self.close()

        # Return True if at least one row was affected, False otherwise
        return num_rows_affected > 0

    def log_video_values(self, video_hash: str, like_value: float):
        """
        Inserts a new video record into the database with the given video hash and like value.

        :param video_hash: A unique identifier for the uploaded video.
        :type video_hash: str
        :param like_value: The fraction of likes for the uploaded video.
        :type like_value: float
        :return: True if the video record was successfully inserted, False otherwise.
        :rtype: bool
        """

        self.open()
        # Define the SQL query with placeholders for parameters to prevent SQL injection
        query = """
            insert into users.video (id_video, id_session, dtime_upload, num_metric_like, num_metric_dislike)
            values (%s, %s, %s, %s, %s);
        """

        # Generate the user ID session for the video
        now = datetime.now()
        self.__user_video_id = f"v-{video_hash}-" + f"{hashlib.md5(video_hash.encode('utf-8')).hexdigest()}"[:15]

        # Execute the query with parameters
        self.__cursor.execute(
            query,
            (
                self.__user_video_id,
                self.__user_id_session,
                now.strftime("%Y-%m-%d %H:%M:%S"),
                like_value,
                1.0 - like_value,
            ),
        )

        # Get the number of rows affected by the query
        num_rows_affected = self.__cursor.rowcount

        # Commit the changes and close the connection
        self.__conn.commit()
        self.close()

        # Return True if at least one row was affected, False otherwise
        return num_rows_affected > 0

    @property
    def user_id_session(self):
        return self.__user_id_session
