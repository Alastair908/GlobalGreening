import streamlit as st
from PIL import Image

# Streamlit uygulamasını başlat
st.title('Resim Uygulaması')

# Resmi yükleyin
image = Image.open('output.png')

# Resmi görüntüleyin
st.image(image, caption='Resim')



import streamlit as st
from PIL import Image

# Takım üyelerinin bilgilerini tanımlayın
team_members = [
    {"name": "John Doe", "image": "image.png"},
    {"name": "Jane Smith", "image": "image.png"},
    {"name": "Michael Johnson", "image": "image.png"},
    {"name": "Emily Davis", "image": "image.png"}
]

# Streamlit uygulamasını başlat
st.title('Takım Üyeleri')

# Dört resmi göstermek için 2x2 bir grid oluşturun
col1, col2 = st.columns(2)

# Takım üyelerini döngüye alın ve resimleri ve isimleri grid içine yerleştirin
for i, member in enumerate(team_members):
    with col1 if i < 2 else col2:
        image = Image.open(member["image"])
        st.image(image, caption=member["name"], use_column_width=True)




import streamlit as st
from PIL import Image

# Takım üyelerinin bilgilerini tanımlayın
team_members = [
    {"name": "John Doe", "image": "image.png"},
    {"name": "Jane Smith", "image": "image.png"},
    {"name": "Michael Johnson", "image": "image.png"},
    {"name": "Emily Davis", "image": "image.png"}
]

# Streamlit uygulamasını başlat
st.title('Takım Üyeleri')

# CSS stilini tanımlayın
css = """
<style>
.team-member-container {
    display: flex;
    flex-direction: row;
    justify-content: space-around;
    align-items: center;
    flex-wrap: wrap;
}

.team-member-image {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center;
}
</style>
"""

# CSS stilini Streamlit'e ekleyin
st.markdown(css, unsafe_allow_html=True)

# Takım üyelerini döngüye alın ve resimleri ve isimleri gösterin
st.markdown('<div class="team-member-container">', unsafe_allow_html=True)
for member in team_members:
    image = Image.open(member["image"])
    resized_image = image.resize((150, 150))  # Resmi istenen boyuta yeniden boyutlandırın
    st.image(resized_image, caption=member["name"], use_column_width=False, \
             output_format="JPEG", \
             key=member["name"], \
             output_container_class="team-member-image")
st.markdown('</div>', unsafe_allow_html=True)



from PIL import Image
import streamlit as st

# Takım üyelerinin bilgilerini tanımlayın
team_members = [
    {"name": "", "image": "image1.png"},
    {"name": "Jane Smith", "image": "image2.png"},
    {"name": "Michael Johnson", "image": "image3.png"},
    {"name": "Emily Davis", "image": "image4.png"}
]

# Yeniden boyutlandırılmış resimlerin boyutu
image_size = (300, 300)

# Streamlit uygulamasını başlat
st.title('Takım Üyeleri')

# Dört resmi göstermek için 2x2 bir grid oluşturun
col1, col2 = st.columns(2)

# Takım üyelerini döngüye alın ve resimleri ve isimleri grid içine yerleştirin
for i, member in enumerate(team_members):
    with col1 if i < 2 else col2:
        image = Image.open(member["image"])
        resized_image = image.resize(image_size)
        st.image(resized_image, caption=member["name"], use_column_width=True)
