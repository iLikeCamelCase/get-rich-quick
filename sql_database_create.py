from getpass import getpass
from mysql.connector import connect, Error

def connect_mysql():
    '''
    Prompts user for credentials to connect to MySQL server
    '''
    try:
        with connect(                                   # "with" statement ensures connection terminated if an exception is raised
            host="localhost",                           # same as try: connect
            user=input("Enter username: "),             #         finally: close connection
            password=getpass("Enter password: "),
        ) as connection:
            # show already created databases
            show_db_query = "SHOW DATABASES"
            with connection.cursor() as cursor:
                cursor.execute(show_db_query)
                for db in cursor:
                    print(db)

            # create database named "historic_intraday"
            create_db_query = "CREATE DATABASE historic_intraday"
            with connection.cursor() as cursor:
                cursor.execute(create_db_query)

            # create table named "intraday_2019"
            create_2019_table_query = """
            CREATE TABLE intraday_2019(
                datetime DATETIME PRIMARY KEY
                open DECIMAL(10,2)
                high DECIMAL(10,2)
                low DECIMAL(10,2)
                close DECIMAL(10,2)
                volume INT
                ticker VARCHAR(5)
            )
            """
            with connection.cursor() as cursor:
                cursor.execute(create_2019_table_query)
                connection.commit()

            # show created table information for review
            show_table_query = "DESCRIBE intraday_2019"
            with connection.cursor() as cursor:
                cursor.execute(show_table_query)
                # fetch rows from last executed query 
                result = cursor.fetchall()
                for row in result:
                    print(row)

    except Error as e:
        print(e)

connect_mysql()