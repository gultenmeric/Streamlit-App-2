import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 3'er üründen oluşan veri seti
products = {
    'product_id': list(range(1, 10)),
    'product_name': ['Laptop A', 'Laptop B', 'Laptop C', 
                     'Telefon A', 'Telefon B', 'Telefon C', 
                     'USB A', 'USB B', 'USB C'],
    'category': ['Laptop', 'Laptop', 'Laptop', 
                 'Telefon', 'Telefon', 'Telefon', 
                 'USB', 'USB', 'USB'],
    'image_file': [
     'C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\asus ProArt.png'
, 
"C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Casper Nirvana z100.jpg",       
      "C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Hp Laptop 15-fc.png", 
"C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Iphone 11 Kırmızı.jpg", 
"C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Iphone 11 Sarı.jpg",

        "C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Iphone 11 Siyah.png", 
"C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Sandberg 3.0.jpg",
"C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\Sandisk usb 3.0.jpg", 
"C:\\Users\\meric\\OneDrive\\Masaüstü\\images\\TechMate 16Gb.jpg"
    ]
}

ratings = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
    'product_id': [1, 2, 3, 2, 3, 4, 1, 4, 5, 6, 7, 8, 1, 3, 5, 6, 7, 9],
    'rating': [5, 4, 3, 5, 4, 3, 5, 4, 5, 4, 3, 5, 5, 4, 5, 4, 3, 4]
}

# Verileri pandas DataFrame'e çevir
products_df = pd.DataFrame(products)
ratings_df = pd.DataFrame(ratings)

# Kullanıcı-ürün puanlama matrisi
user_product_matrix = ratings_df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# Ürünler arası benzerlik matrisi
product_similarity = cosine_similarity(user_product_matrix.T)

# Streamlit uygulaması başlığı
st.title("Kategorilere Dayalı Ürün Öneri Sistemi")

# Ürün seçimi
product_options = products_df['product_name'].tolist()
selected_product = st.selectbox("Bir ürün seçin:", product_options)

# Seçilen ürünün ID'sini ve kategorisini bul
selected_product_id = products_df[products_df['product_name'] == selected_product]['product_id'].values[0]
selected_category = products_df[products_df['product_name'] == selected_product]['category'].values[0]
selected_product_image = products_df[products_df['product_name'] == selected_product]['image_file'].values[0]

# Seçilen ürünü göster
st.subheader("Seçilen Ürün")
st.image(selected_product_image, width=200, caption=selected_product)

# Aynı kategorideki ürünleri filtrele
same_category_products = products_df[products_df['category'] == selected_category]

# Benzer ürünleri bul
similar_products = list(enumerate(product_similarity[selected_product_id - 1]))
similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)

# Sadece aynı kategorideki ürünlerden öneri seç
filtered_similar_products = [
    (product_id, similarity) for product_id, similarity in similar_products
    if product_id + 1 in same_category_products['product_id'].values and product_id + 1 != selected_product_id
][:3]

# Önerilen ürünleri göster
st.subheader("Benzer Ürünler")
if filtered_similar_products:
    for product_id, similarity in filtered_similar_products:
        product_info = products_df.iloc[product_id]
        st.image(product_info['image_file'], width=150, caption=product_info['product_name'])
else:
    st.write("Bu kategori için başka öneri bulunamadı.")
