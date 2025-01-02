import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Streamlit Başlığı
st.title("Ürün Öneri Sistemi")
st.write("Bu uygulama, seçilen bir ürüne benzer ürünler önerir.")

# Veri setini yükle (örnek veri, dosyayı yüklemeniz gerekebilir)
Event = pd.read_csv('events.csv')

# Sadece etkileşimi olan (addtocart) ürünleri seçme
Event_filtered = Event[Event['event'] == 'addtocart']

# Ürün ID'lerini alıp unique hale getirme (sadece etkileşimde olanlar)
all_items = Event_filtered['itemid'].unique()

# Surprise formatına dönüştürme
reader = Reader(rating_scale=(1, 5))

# Retail Rocket veri setini uygun formatta alacak bir veri seti oluşturma
df_ratings = Event_filtered.groupby(['visitorid', 'itemid']).size().reset_index(name='rating')

dataset = Dataset.load_from_df(df_ratings[['visitorid', 'itemid', 'rating']], reader)

# Eğitim ve test setine ayırma
trainset, testset = train_test_split(dataset, test_size=0.2)

# Ürün tabanlı işbirlikçi filtreleme
sim_options = {
    'name': 'cosine',
    'user_based': False  # Ürün tabanlı filtreleme
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Streamlit session_state'i kullanarak seçilen ürün ID'sini saklamak
if "selected_product_id" not in st.session_state:
    st.session_state.selected_product_id = None

# Seçilebilir ürünleri gösterme
selected_product_id = st.selectbox(
    "Bir ürün seçin",
    options=all_items.tolist(),  # Etkileşime girilen ürünler listesi
    index=0 if st.session_state.selected_product_id is None else list(all_items).index(st.session_state.selected_product_id)
)

# Seçilen ürünü session_state'e kaydet
if selected_product_id != st.session_state.selected_product_id:
    st.session_state.selected_product_id = selected_product_id

# "Önerileri Göster" butonuna tıklanınca öneri oluşturma
if st.button("Önerileri Göster"):
    st.write(f"Seçilen ürün ID'si: {st.session_state.selected_product_id}")
    
    try:
        inner_id = trainset.to_inner_iid(st.session_state.selected_product_id)  # Orijinal ID'yi içsel ID'ye çevir
        neighbors = model.get_neighbors(inner_id, k=5)  # Seçilen ürüne en yakın 5 ürün önerisi
        suggestions = [trainset.to_raw_iid(neighbor) for neighbor in neighbors]  # İçsel ID'leri tekrar orijinal ID'ye çevir
        st.write(f"Seçilen ürüne benzeyen ürünler:", suggestions)
    except ValueError:
        st.write(f"Ürün ID'si {st.session_state.selected_product_id} geçersiz.")
