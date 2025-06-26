import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# --- 設定項目 ---
SOURCE_FILE = "doshisha_data.txt" # 収集したデータファイル
DB_SAVE_PATH = "doshisha_faiss_db"  # 作成するデータベースの保存先フォルダ名
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" # 日本語に強いベクトル化モデル

def build_vector_database():
    """
    テキストファイルを読み込み、チャンクに分割し、ベクトルデータベースを構築して保存する関数
    """
    print(f"'{SOURCE_FILE}' からデータを読み込んでいます...")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    # --- 1. テキストを意味のある塊（チャンク）に分割 ---
    print("テキストをチャンクに分割しています...")
    # ドキュメントをURLごとに分割
    documents = []
    # "--- URL: "で始まる行を基準に分割
    raw_docs = re.split(r'--- URL: (.+?) ---', full_text)
    
    # 最初の空要素を削除
    if not raw_docs[0].strip():
        raw_docs.pop(0)

    # URLと内容をペアにする
    for i in range(0, len(raw_docs), 2):
        url = raw_docs[i].strip()
        content = raw_docs[i+1].strip()
        documents.append({'page_content': content, 'metadata': {'source': url}})
    
    # LangChainの分割機能を利用
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # チャンクの最大文字数
        chunk_overlap=100, # チャンク間で重複させる文字数
    )

    # Documentオブジェクトに変換
    from langchain.docstore.document import Document
    docs_for_split = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents]
    
    all_splits = text_splitter.split_documents(docs_for_split)
    print(f"合計 {len(all_splits)} 個のチャンクに分割されました。")


    # --- 2. テキストをベクトルに変換（エンベディング） ---
    print(f"ベクトル化モデル '{EMBEDDING_MODEL}' を読み込んでいます...")
    # HuggingFaceのモデルを使って、テキストをベクトル化する準備
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # --- 3. ベクトルデータベースの作成と保存 ---
    print("ベクトルデータベースを構築しています...（時間がかかる場合があります）")
    # 分割したチャンクとベクトル化モデルを使って、FAISSデータベースを作成
    db = FAISS.from_documents(all_splits, embeddings)

    print(f"データベースを '{DB_SAVE_PATH}' に保存しています...")
    # 作成したデータベースをローカルに保存
    db.save_local(DB_SAVE_PATH)
    
    print("\n知識データベースの構築が完了しました！")

if __name__ == '__main__':
    build_vector_database()