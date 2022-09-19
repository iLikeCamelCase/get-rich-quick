from getpass import getpass
from mysql.connector import connect, Error
from main import pathing

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

            # create database named "historic_intraday", in future edit it to do it automatically, assume users are stupid 
            path = pathing(input("Create new database labeled 'historic_intraday' (1)\nSkip database creation (2)"),1,2)    
            if (path == 1):
                create_db_query = "CREATE DATABASE historic_intraday"
                with connection.cursor() as cursor:     
                    cursor.execute(create_db_query)
            else:
                print("No database created.")

            connection.database = "historic_intraday"

            # create table named "intraday_2019"
            path = pathing(input("Create new table labeled 'intraday_2019'? (1)\nSkip table creation (2)"),1,2)
            if (path == 1):
                create_2019_table_query = """
                CREATE TABLE intraday_2019(
                    datetime DATETIME,
                    open DECIMAL(10,2),
                    high DECIMAL(10,2),
                    low DECIMAL(10,2),
                    close DECIMAL(10,2),
                    volume INT,
                    PRIMARY KEY (datetime)
                );
                """
                with connection.cursor() as cursor:
                    cursor.execute(create_2019_table_query)
                    connection.commit()
            else:
                print("No table created.")

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