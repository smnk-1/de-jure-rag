import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import ConstitutionRAG


def main():
    st.title("Конституция РФ - Юридический ассистент")

    rag_system = ConstitutionRAG()

    user_query = st.text_input("Введите ваш вопрос по Конституции РФ:")

    if st.button("Получить ответ"):
        if user_query:
            with st.spinner("Поиск релевантных статей и генерация ответа..."):
                result = rag_system.generate_answer(user_query)
            
            st.subheader("Ответ:")
            st.write(result["answer"])
            
            if result["used_articles"]:
                st.subheader("Использованные статьи:")
                for article in result["used_articles"]:
                    st.write(f"- Статья {article['number']} ({article['chapter']})")
        else:
            st.warning("Пожалуйста, введите вопрос.")

if __name__ == "__main__":
    main()
