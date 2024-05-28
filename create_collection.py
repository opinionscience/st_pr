import streamlit as st
from lib import connection
from lib.connection import *
# from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd 
import os 
from langchain.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import BGEM3FlagModel

@st.cache_data()
def pickle_loader(path):
    return  pd.read_pickle(path)

@st.cache_resource()
def connect_chroma(configuration):
    conn = st.connection(
        "chromadb",
        type=ChromadbConnection,
        **configuration
        )
    return conn

@st.cache_resource()
def load_model():
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    return model

@st.cache_resource()
def connect(configuration = {"client_type": "PersistentClient", "path": "tmp"}):
    return connect_chroma(configuration)

@st.cache_resource()
def add_documents(_conn, collection_name, embeddings, lst_text, metadatas, lst_ids):
    _conn.upload_documents(
                collection_name=collection_name,
                embeddings = embeddings,
                documents=lst_text,
                metadatas=metadatas,
                ids=lst_ids
                )
    
@st.cache_resource()
def query_data(_conn, collection_name, query_embeddings, num_results_limit = 10, attributes=["documents", "embeddings", "metadatas", "data"], where_metadata_filter={}):
    queried_data = _conn.query_emb(collection_name=collection_name,
                            query_embeddings=query_embeddings,
                            num_results_limit=num_results_limit,
                            attributes=attributes,
                            where_metadata_filter=where_metadata_filter)
    return queried_data
    
@st.cache_resource()
def load_HF_embeddings(model_name):
    """
    create a HugginFace encoder
    """
    try:
        HF_encoder = HuggingFaceEmbeddings(model_name=model_name)
        return HF_encoder
    except Exception as e:
        pass
        print(e)

def HF_vectorize(HF_encoder, lst_txt):
    """
    Vectorize using a Huggingface encoder
    """
    embeddings = HF_encoder.embed_documents(lst_txt)

    return embeddings


def main():
    st.set_page_config(
        page_title="Search engine",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title('Settings')
    st.title('Analysis') 

    q = st.sidebar.text_area("query", value="macron")
    

    model = load_model()

    collection_name = "newest_collection3"

    conn = connect()
    
    try:
        huggingface_ef = {
            "api_key":"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",
            "model_name" :"BAAI/bge-m3"
         }
        conn.create_collection(collection_name=collection_name, embedding_function_name = "HuggingFaceEmbeddingFunction", embedding_config=huggingface_ef,  metadata={"hnsw:space": "cosine"})
    except:
        pass
    

    documents_collection_df = conn.get_collection_data(collection_name, attributes= ["documents", "embeddings", "metadatas"])
    # st.write(documents_collection_df)
    if len(documents_collection_df)< 1:
        try:
            df_to_upload = pickle_loader("data/df_prod_chroma_v2.pickle")
            df_to_upload['tweet_html'] = df_to_upload['tweet_html'].fillna(df_to_upload['translated_text'])
            df_to_upload[["message_id", "user_id", "user_name", "plateforme", "date", "translated_text", 'tweet_html']]=df_to_upload[["message_id", "user_id", "user_name", "plateforme", "date", "translated_text", 'tweet_html']].astype(str)
            df_to_upload[["views", "share", "comments", "likes", "engagements"]] = df_to_upload[["views", "share", "comments", "likes", "engagements"]].astype(int)
            # st.write(df_to_upload.head())
            lst_text = list(df_to_upload["text"])
            lst_ids = list(df_to_upload["message_id"])
            embeddings = list(df_to_upload["embeddings"])

            cols_metadata = ["user_id", "user_name", "views", "share", "comments", "likes", "engagements", "plateforme", "date", "translated_text", 'tweet_html']
            metadatas =  df_to_upload[cols_metadata].to_dict(orient="records")
            add_documents(conn, collection_name, embeddings, lst_text, metadatas, lst_ids)

        except Exception as e:
            pass
            st.write(e)
    else:
        
        query_embeddings = model.encode([q], 
                                batch_size=12, 
                                max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
        
        results_telegram = query_data(conn, collection_name, query_embeddings.tolist(), num_results_limit = 10, attributes=["documents", "metadatas", "embeddings"], where_metadata_filter={"plateforme": "Telegram"})


        results_twitter_1 = query_data(conn, collection_name, query_embeddings.tolist(), num_results_limit = 100, attributes=["documents", "metadatas", "embeddings"], where_metadata_filter={"plateforme": "Twitter"})
    
        results_twitter = query_data(conn, collection_name, results_telegram["embeddings"][0][0], num_results_limit = 100, attributes=["documents", "metadatas", "embeddings"], where_metadata_filter={"plateforme": "Twitter"})
        st.write(results_twitter)
        col1, col2, col3 = st.columns(3, gap="small")
        with col1:
            for i, r in enumerate(results_telegram["metadatas"][0]):
                st.write('*'*50)
                st.write(r['translated_text'])

        with col2:
            for i, r in enumerate(results_twitter_1["documents"][0]):
                st.write('*'*50)
                st.write(r)
        with col3:   

            for i, r in enumerate(results_twitter["documents"][0]):
                st.write('*'*50)
                st.write(r)

if __name__ == "__main__":
    main()