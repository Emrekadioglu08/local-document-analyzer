import streamlit as st
import requests

st.set_page_config(page_title="Yerel Belge Analiz Asistanı", page_icon="📄", layout="centered")

st.title("📄 Yerel Belge Analiz Asistanı")
st.markdown("NotebookLM alternatifi, tamamen gizli ve yerel (Ollama & LlamaIndex) çalışan doküman analiz aracı.")

# 1. Dosya Yükleme Alanı (Çoklu dosya desteği ile)
st.subheader("1. Belge Yükleme")
uploaded_files = st.file_uploader("Analiz etmek istediğiniz PDF dosyalarını seçin", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Dosyaları Yükle ve İndeksle"):
        with st.spinner("Dosyalar sunucuya gönderiliyor ve LlamaIndex ile işleniyor..."):
            files_payload = [
                ("files", (file.name, file.getvalue(), "application/pdf")) 
                for file in uploaded_files
            ]
            try:
                response = requests.post("http://127.0.0.1:8000/upload", files=files_payload)
                if response.status_code == 200:
                    st.success("Dosyalar başarıyla yüklendi ve indekslendi!")
                else:
                    error_detail = response.json().get("detail", "Bilinmeyen hata")
                    st.error(f"Yükleme hatası: {error_detail}")
            except Exception as e:
                st.error(f"Backend sunucusuna bağlanılamadı: {e}")

st.markdown("---")

# 2. Akıllı Özet Alanı
st.subheader("2. 📑 Akıllı Belge Özeti ve İçindekiler")
if st.button("🚀 Belgeyi Analiz Et & Özet Çıkar"):
    with st.spinner("LlamaIndex belgeyi tarıyor, özet ve içindekiler hazırlanıyor..."):
        try:
            response = requests.get("http://127.0.0.1:8000/summary")
            if response.status_code == 200:
                data = response.json()
                if "summary" in data:
                    st.success("Özet başarıyla oluşturuldu!")
                    st.markdown("### 📋 Yönetici Özeti & İçindekiler")
                    st.write(data["summary"])
                else:
                    st.warning(data.get("error", "Bir hata oluştu."))
            else:
                st.error("Backend sunucusuna ulaşılamadı.")
        except Exception as e:
            st.error(f"Bağlantı hatası: {e}")

st.markdown("---")

# 3. Soru - Cevap Alanı
st.subheader("3. 💬 Belgeye Soru Sor")
user_question = st.text_input("Yüklediğiniz belge hakkında ne öğrenmek istiyorsunuz?")

if st.button("Soruyu Gönder"):
    if user_question:
        with st.spinner("Yapay zeka yanıtı arıyor..."):
            try:
                payload = {"question": user_question}
                response = requests.post("http://127.0.0.1:8000/query", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### 🤖 Yanıt:")
                    st.write(data.get("response"))
                    
                    sources = data.get("sources", [])
                    if sources:
                        st.caption(f"Kaynak Dosyalar: {', '.join(set(sources))}")
                else:
                    error_detail = response.json().get("detail", "Bilinmeyen hata")
                    st.error(f"Sorgu hatası: {error_detail}")
            except Exception as e:
                st.error(f"Bağlantı hatası: {e}")
    else:
        st.warning("Lütfen bir soru yazın.")