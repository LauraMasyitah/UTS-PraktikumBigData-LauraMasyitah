# UTS-BigData-LauraMasyitah
Dashboard ini merupakan dashboard interaktif berbasis Streamlit yang dirancang untuk melakukan klasifikasi gambar dan deteksi objek pada alat tulis (stationery). Sistem ini memanfaatkan dua model machine learning yang telah dilatih sebelumnya, yaitu:

Model Klasifikasi (.h5) untuk mengidentifikasi jenis barang dari satu gambar yang diupload.

Model Deteksi Objek (.pt) berbasis YOLO untuk mendeteksi beberapa objek dalam satu gambar, lengkap dengan bounding box, label, dan confidence score.

Dashboard dikembangkan sebagai bagian dari UTS Pemrograman Big Data, dengan tujuan mempermudah pengujian model secara real-time melalui antarmuka yang sederhana, responsif, dan mudah digunakan. Aplikasi ini mendukung upload gambar berbagai format (JPG, JPEG, PNG), menampilkan hasil prediksi secara visual, serta menyediakan contoh gambar otomatis untuk memudahkan pengguna dalam mencoba fitur.

### cara menjalankan dashboard
```
conda create -n uts-laura python=3.9 
conda activate uts-laura
pip install -r requirements.txt 
```

### Run streamlit
```
python -m streamlit run dashboard.py 
```

### Link Dashboard Streamlit
[Klik untuk membuka Dashboard Streamlit](https://uts-praktikumbigdata-laura.streamlit.app/)

