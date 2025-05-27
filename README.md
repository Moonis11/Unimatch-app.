# Certificate Type Classifier and University Matcher App

This Streamlit web application classifies uploaded certificate images to identify the certificate type and extract the score using a pre-trained machine learning model. After recognizing the score, the app compares it against a CSV database of universities and displays a list of universities that match the user's certificate scores.

The model is loaded dynamically from Google Drive to keep the repository size small and make updating easier. Users can upload certificate images (e.g., IELTS, TOEFL, Duolingo), get instant score extraction, and see which universities accept their scores.

## Features

- Upload certificate images in JPG or PNG format
- Automatically detect certificate type and extract score
- Load the ML model dynamically from Google Drive
- Compare scores with university requirements stored in a CSV file
- Display a list of matching universities based on your scores
- Built with Streamlit and Python for a fast and user-friendly experience

---

# Sertifikat Turi Aniqlovchi va Universitet Moslashtiruvchi Dastur

Ushbu Streamlit ilovasi foydalanuvchi yuklagan sertifikat rasmlarini turlariga ajratadi va sertifikatdagi ballarni oldindan o‘rgatilgan mashina o‘rganish modeli yordamida aniqlaydi. Keyin, ballarni CSV faylida saqlangan universitetlar talablariga solishtirib, foydalanuvchiga mos keluvchi universitetlar ro‘yxatini ko‘rsatadi.

Model Google Drive’dan dinamik yuklanadi, bu repository hajmini kichraytiradi va yangilashni osonlashtiradi. Foydalanuvchilar IELTS, TOEFL, Duolingo kabi sertifikat rasmlarini yuklab, ballarni olishlari va o‘z ballariga mos universitetlarni ko‘rishlari mumkin.

## Xususiyatlari

- Sertifikat rasmlarini JPG yoki PNG formatida yuklash
- Sertifikat turini aniqlash va ballarni chiqarish
- Mashina o‘rganish modelini Google Drive’dan avtomatik yuklash
- Ballarni CSV formatidagi universitet talablariga solishtirish
- Mos keluvchi universitetlar ro‘yxatini ko‘rsatish
- Streamlit va Python yordamida tez va qulay interfeys

