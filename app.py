import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from openai import OpenAI # この行は不要になるのでコメントアウトまたは削除

# --- APIキーの設定 ---
os.environ["OPENAI_API_KEY"] = "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"
# --- ここまで ---

# --- 設定項目 ---
DB_LOAD_PATH = "doshisha_faiss_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# --- アプリの初期設定 ---
st.set_page_config(page_title="同志社大学AIアバター", page_icon="🤖", layout="wide")

# --- 関数の定義 ---
@st.cache_resource
def load_dependencies():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(DB_LOAD_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever

# --- Webアプリの画面描画 ---
st.title("🤖 同志社大学AIアバター")
st.caption("同志社大学公式サイトの情報を基に、AIが質問に対話形式で回答します。")

if not ("OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-")):
    st.error("OpenAI APIキーが正しく設定されていません。コード内の該当箇所をもう一度ご確認ください。")
else:
    try:
        # 依存関係（DBとRetriever）をロード
        retriever = load_dependencies()

        # --- 【最重要修正点】LLMの初期化方法をLangChainの標準的な方法に戻します ---
        # これが最新のライブラリ間の連携で最も安定した書き方です
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        # --- ここまで修正 ---

        prompt = PromptTemplate(
            template="""あなたは同志社大学に関する質問に答える、親切で優秀なAIアシスタントです。
        提供された以下の「コンテキスト情報」だけを基にして、質問に日本語で回答してください。
        コンテキスト情報に答えがない場合は、「申し訳ありませんが、その情報は見つかりませんでした。」と回答してください。自身の知識で答えてはいけません。

        【コンテキスト情報】
        {context}

        【質問】
        {input}

        【回答】
        """,
            input_variables=["context", "input"],
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        question = st.text_input("質問を入力してください（例：建学の精神について教えてください）", key="user_question")

        if question:
            st.write(f"**質問:** {question}")
            
            with st.spinner("AIが回答を生成中です..."):
                response = retrieval_chain.invoke({"input": question})
                
                st.subheader("✅ 回答")
                st.write(response["answer"])

                st.subheader("--- 回答の根拠となった情報 ---")
                for i, doc in enumerate(response["context"]):
                    source_url = doc.metadata.get('source', '不明なソース')
                    with st.expander(f"【根拠{i+1}】 {doc.page_content[:50]}..."):
                        st.write(doc.page_content)
                        st.link_button("出典元ページを開く", url=source_url)

    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")