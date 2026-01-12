import streamlit as st

def show():
    st.markdown("## Let's Connect")
    st.write(
        "Terima kasih sudah melihat project ini. "
        "Saya terbuka untuk diskusi, kolaborasi, maupun peluang di bidang **Data Analyst / Data Scientist**."
    )

    st.write("")
    st.markdown(
        """
        - **LinkedIn:** https://www.linkedin.com/in/dwi-amalia-/  
        - **GitHub:** https://github.com/dwiamalia123 
        - **Email :** dwiamalia228@gmail.com 
        """
    )

    st.divider()

    # =========================
    # Contact Form
    # =========================
    st.markdown("## 📝 Send a Message")

    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=160)
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not name or not email or not message:
            st.warning("Mohon lengkapi **Name**, **Email**, dan **Message**.")
        else:
            st.success("Terima kasih! Pesan kamu sudah tercatat ✅")
            with st.expander("Lihat ringkasan pesan"):
                st.write(
                    {
                        "name": name,
                        "email": email,
                        "subject": subject,
                        "message": message
                    }
                )

    # =========================
    # (3) Footer
    # =========================
    st.markdown(
        """
        <div class="footer">
            Built with Streamlit • Wine Quality Analysis & Prediction • January 2026
        </div>
        """,
        unsafe_allow_html=True
    )
