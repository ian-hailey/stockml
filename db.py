import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np


class Db(object):
    def __init__(self, username='postgres', passwd='', host='localhost', database='stock'):
        self.username = username
        self.passwd = passwd
        self.host = host
        self.database = database
        self.connection = None
        try:
            self.connection = psycopg2.connect(user=username,
                                          password="eE{9NyUw,}l?",
                                          host=host,
                                          port="5432",
                                          database=database)
            cursor = self.connection.cursor()
            print(self.connection.get_dsn_parameters())
            cursor.execute("SELECT version();")
            record = cursor.fetchone()
            print("Connected to - {}".format(record))
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL host {} {}".format(host, error))
            self.connection = None
            raise error

    def __del__(self):
        if self.connection != None:
            self.connection.close()
            self.connection = None

    def check_symbol_exists(self, symbol):
        exists = False
        sql = "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_schema = 'public' AND table_name=LOWER('{}'));".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            if cursor.rowcount:
                exists = cursor.fetchone()[0]
        return exists

    def get_table_date_range(self, symbol):
        startDate = None
        endDate = None
        sql = "SELECT datetime FROM {} ORDER BY datetime ASC LIMIT 1".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            if cursor.rowcount:
                startDate = cursor.fetchone()[0]
        sql = "SELECT datetime FROM {} ORDER BY datetime DESC LIMIT 1".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            if cursor.rowcount:
                endDate = cursor.fetchone()[0]
        return startDate, endDate

    def delete_symbol(self, symbol):
        sql = "DROP TABLE IF EXISTS {} CASCADE;".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()
        sql = "DROP TABLE IF EXISTS {}_import CASCADE;".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()
        sql = "DROP TABLE IF EXISTS {}_days CASCADE;".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()
        sql = "DROP TABLE IF EXISTS {}_signals CASCADE;".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()

    def create_symbol_schema(self, symbol):
        sql = "CREATE TABLE IF NOT EXISTS {} ( \
                    datetime   TIMESTAMP  PRIMARY KEY, \
                    open       DECIMAL    NOT NULL, \
                    high       DECIMAL    NOT NULL, \
                    low        DECIMAL    NOT NULL, \
                    close      DECIMAL    NOT NULL, \
                    volume     INTEGER    NOT NULL);".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "CREATE TABLE IF NOT EXISTS {}_import ( \
                    datetime   TIMESTAMP  PRIMARY KEY, \
                    open       DECIMAL    NOT NULL, \
                    high       DECIMAL    NOT NULL, \
                    low        DECIMAL    NOT NULL, \
                    close      DECIMAL    NOT NULL, \
                    volume     INTEGER    NOT NULL);".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "CREATE TABLE IF NOT EXISTS {}_days ( \
                    datetime   TIMESTAMP  PRIMARY KEY, \
                    open       DECIMAL    NOT NULL, \
                    high       DECIMAL    NOT NULL, \
                    low        DECIMAL    NOT NULL, \
                    close      DECIMAL    NOT NULL, \
                    volume     INTEGER    NOT NULL);".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "CREATE TABLE IF NOT EXISTS {}_signals ( \
                    datetime   TIMESTAMP  PRIMARY KEY, \
                    zc         DECIMAL    NOT NULL);".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()
        sql = "SELECT create_hypertable('{}', 'datetime');".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "SELECT create_hypertable('{}_days', 'datetime');".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "SELECT create_hypertable('{}_signals', 'datetime');".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()
        pass

    def import_ohlcv_csv(self, filename, symbol):
        if self.check_symbol_exists(symbol) == False:
            print("Symbol {} does not exist - creating".format(symbol))
            self.create_symbol_schema(symbol)
        sql = "DELETE FROM {}_import;".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "COPY {}_import FROM STDIN WITH CSV HEADER DELIMITER AS ',';".format(symbol)
        f_import = open(filename)
        with self.connection.cursor() as cursor:
            cursor.copy_expert(sql, file=f_import)
        self.connection.commit()
        sql = "INSERT INTO {} SELECT * FROM {}_import ON CONFLICT DO NOTHING;".format(symbol, symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        sql = "DELETE FROM {}_import;".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()

    def insert_data_frame(self, df, table):
        df_columns = list(df)
        df_columns.insert(0, 'datetime')
        sql_columns = ",".join(df_columns)
        sql_values = "VALUES({})".format(",".join(["%s" for _ in df_columns]))
        sql = "INSERT INTO {} ({}) {} ON CONFLICT DO NOTHING;".format(table, sql_columns, sql_values)
        df_no_index = df.reset_index()
        with self.connection.cursor() as cursor:
            execute_batch(cursor, sql, df_no_index.values)
        self.connection.commit()

    def insert_signals(self, df_signals, symbol):
        table = "{}_signals".format(symbol)
        self.insert_data_frame(df_signals, table)

    def insert_ohlcv_days(self, df_days, symbol):
        table = "{}_days".format(symbol)
        self.insert_data_frame(df_days, table)

    def get_symbol_ohlcv(self, symbol, daterange, resolution='secs'):
        data = None
        if resolution == 'secs':
            table = symbol
        else:
            table = "{}_days".format(symbol)
        sql = "SELECT * FROM {} WHERE (datetime BETWEEN '{}' AND '{}') ORDER BY datetime;".format(table, daterange[0], daterange[1])
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            data = pd.DataFrame(cursor.fetchall(), columns=["Date", "Open", "High", "Low", "Close", "Volume"], dtype=np.float64)
            data['Volume'] = data['Volume'].astype(int)
            data = data.set_index('Date')
        return data

    def get_symbol_signals(self, symbol, daterange):
        data = None
        columns = []
        sql = "SELECT column_name FROM information_schema.columns WHERE table_name = LOWER('{}_signals');".format(symbol)
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                columns.append(row[0])
            columns[0] = 'Date'
        sql = "SELECT * from {}_signals where (datetime BETWEEN '{}' AND '{}') ORDER BY datetime;".format(symbol, daterange[0], daterange[1])
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            data = pd.DataFrame(cursor.fetchall(), columns=columns, dtype=np.float64)
            data = data.set_index('Date')
        return data

    def get_symbol_info(self, symbol):
        info = None
        if self.check_symbol_exists(symbol):
            info = {}
            info['start'], info['end'] = self.get_table_date_range(symbol)
        return info

