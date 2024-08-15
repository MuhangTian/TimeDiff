import pandas as pd
from sqlalchemy import create_engine


class DataBaseLoader():
    """
    For querying data from local postgreSQL database.

    Parameters
    ----------
    user : str
    password : str
    dbname : str, optional
        database name, by default "mimiciv"
    port : str, optional
        port you are using, by default "5432"
    schema : str, optional
        schema you are using, by default "mimiciv_hosp"
    host : str, optional
        host you are on, by default "localhost"
    """
    def __init__(self, user:str, password:str, dbname:str="mimiciv", port:str="5432", schema:str="mimiciv_hosp", host:str="localhost") -> None:
        assert all(isinstance(e, str) for e in [dbname, user, password, host, port]), "those parameters must be string!"
        assert isinstance(schema, str) or isinstance(schema, None), "\"schema\" must be str or None!"
        self.dbname = dbname
        if schema is not None:
            self.engine = create_engine(
                f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
                connect_args={'options': f"-c search_path={schema}"}
            )
        else:
            self.engine = create_engine(
                f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
            )
    
    def __getitem__(self, TableName:str) -> pd.DataFrame:
        assert isinstance(TableName, str), "\"TableName\" must be a string!"
        table = pd.read_sql_query(f"SELECT * FROM {TableName}", self.engine)
        
        return table
    
    def query(self, command:str) -> pd.DataFrame:
        '''for making query from database, with arbitrary SQL command'''
        assert isinstance(command, str), "\"command\" must be a string!"
        query = pd.read_sql_query(command, self.engine)
        
        return query