# Instalasi Fast Api

### Install fast api

- pip install fastapi

### Install uvicorn

- pip install "uvicorn[standard]"

### Install python-multipart

- pip install python-multipart

### import FastAPI from FastApi



# GUIDE

### Deklarasi method
Deklarasi akses method bisa dilakukan menggunakan @app.method('url')

Setelah deklarasi definisikan fungsi yang akan dijalankan ketika endpoint diakses.


### Body Request
Untuk mengakses body request import BaseModel dari pydantic

Kemudian buat sebuah model yang extend ke BaseModel, kemudian isi attributnya sesuai dengan data yang dibutuhkan.

Jadikan model yang telah dibuat sebagai type pada parameter fungsi.

Untuk membuat sebuah attribut menjadi opsional bisa menggunakan type None = None


### Form field
