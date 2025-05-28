import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

#A. Konfigurasi Awal dan Pemuatan Model/Aset
st.set_page_config(
    page_title="Prediksi Potensi Dropout Siswa - Jaya Jaya Institut",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definisikan mapping status (untuk tampilan hasil)
STATUS_MAPPING = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
YES_NO_OPTIONS = {0: "Tidak", 1: "Ya"}

# Opsi untuk input selectbox berdasarkan unique values yang diberikan
S1_UNITS_ENROLLED_OPTIONS = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 26])
S1_UNITS_APPROVED_OPTIONS = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26])
S2_UNITS_ENROLLED_OPTIONS = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23])
S2_UNITS_APPROVED_OPTIONS = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20])

DEFAULT_INDEX_6_S1_ENROLLED = S1_UNITS_ENROLLED_OPTIONS.index(6) if 6 in S1_UNITS_ENROLLED_OPTIONS else 0
DEFAULT_INDEX_6_S1_APPROVED = S1_UNITS_APPROVED_OPTIONS.index(6) if 6 in S1_UNITS_APPROVED_OPTIONS else 0
DEFAULT_INDEX_6_S2_ENROLLED = S2_UNITS_ENROLLED_OPTIONS.index(6) if 6 in S2_UNITS_ENROLLED_OPTIONS else 0
DEFAULT_INDEX_6_S2_APPROVED = S2_UNITS_APPROVED_OPTIONS.index(6) if 6 in S2_UNITS_APPROVED_OPTIONS else 0

# List fitur asli yang dibutuhkan untuk rekayasa fitur dari 11 fitur final (standardized names)
ORIGINAL_FEATURES_FOR_FE = [
    '1st_Sem_Units_Approved', '1st_Sem_Units_Enrolled', '1st_Sem_Units_Grade',
    '2nd_Sem_Units_Approved', '2nd_Sem_Units_Enrolled', '2nd_Sem_Units_Grade',
    'Tuition_Fees_Up_To_Date', 'Debtor', 'Scholarship_Holder'
]

# List 11 fitur final yang digunakan untuk training model (standardized names)
FINAL_TRAINING_FEATURES = [
    'Approval_Rate_2nd_Sem', 'Overall_Approval_Rate', 'Approval_Rate_1st_Sem',
    '2nd_Sem_Units_Approved', 'Total_Units_Approved', '2nd_Sem_Units_Grade',
    'Average_Curricular_Grade', '1st_Sem_Units_Approved', '1st_Sem_Units_Grade',
    'Financial_Health_Score', 'Tuition_Fees_Up_To_Date'
]

# Column renaming map from notebook
# Keys are original names, Values are standardized names
NEW_FEATURE_NAMES_FROM_NOTEBOOK = {
    'Marital_status': 'Marital_Status',
    'Application_mode': 'Application_Mode',
    'Application_order': 'Application_Order',
    'Daytime_evening_attendance': 'Daytime_Evening_Attendance',
    'Previous_qualification': 'Previous_Qualification',
    'Previous_qualification_grade': 'Previous_Qualification_Grade',
    'Nacionality': 'Nationality', # Correcting the typo
    'Mothers_qualification': 'Mothers_Qualification',
    'Fathers_qualification': 'Fathers_Qualification',
    'Mothers_occupation': 'Mothers_Occupation',
    'Fathers_occupation': 'Fathers_Occupation',
    'Admission_grade': 'Admission_Grade',
    'Educational_special_needs': 'Educational_Special_Needs',
    'Tuition_fees_up_to_date': 'Tuition_Fees_Up_To_Date', # Relevant
    'Scholarship_holder': 'Scholarship_Holder',          # Relevant
    'Age_at_enrollment': 'Age_At_Enrollment',
    'Unemployment_rate': 'Unemployment_Rate',
    'Inflation_rate': 'Inflation_Rate',
    'Curricular_units_1st_sem_credited': '1st_Sem_Units_Credited',
    'Curricular_units_1st_sem_enrolled': '1st_Sem_Units_Enrolled', # Relevant
    'Curricular_units_1st_sem_evaluations': '1st_Sem_Units_Evaluations',
    'Curricular_units_1st_sem_approved': '1st_Sem_Units_Approved', # Relevant
    'Curricular_units_1st_sem_grade': '1st_Sem_Units_Grade',       # Relevant
    'Curricular_units_1st_sem_without_evaluations': '1st_Sem_Units_Without_Evaluations',
    'Curricular_units_2nd_sem_credited': '2nd_Sem_Units_Credited',
    'Curricular_units_2nd_sem_enrolled': '2nd_Sem_Units_Enrolled', # Relevant
    'Curricular_units_2nd_sem_evaluations': '2nd_Sem_Units_Evaluations',
    'Curricular_units_2nd_sem_approved': '2nd_Sem_Units_Approved', # Relevant
    'Curricular_units_2nd_sem_grade': '2nd_Sem_Units_Grade',       # Relevant
    'Curricular_units_2nd_sem_without_evaluations': '2nd_Sem_Units_Without_Evaluations',
}

# Pemuatan Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/model.joblib')
        return model
    except FileNotFoundError:
        st.error("File model 'model/model.joblib' tidak ditemukan. Pastikan sudah ada di folder 'model' yang benar.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

xgb_model = load_model()

#Fungsi Pra-pemrosesan Data (Disesuaikan)
def preprocess_data(df_input):
    df_processed = df_input.copy()
    unit_cols = ['1st_Sem_Units_Enrolled', '1st_Sem_Units_Approved', 
                 '2nd_Sem_Units_Enrolled', '2nd_Sem_Units_Approved']
    grade_cols = ['1st_Sem_Units_Grade', '2nd_Sem_Units_Grade']
    financial_cols = ['Tuition_Fees_Up_To_Date', 'Debtor', 'Scholarship_Holder']

    for col in unit_cols + grade_cols + financial_cols:
        if col not in df_processed.columns:
            # This error will now be more informative if it happens after delimiter fix
            st.error(f"Kolom yang dibutuhkan '{col}' tidak ada dalam data input untuk preprocessing. Kolom yang ada: {df_processed.columns.tolist()}")
            raise ValueError(f"Missing column for preprocessing: {col}")
        try:
            df_processed[col] = pd.to_numeric(df_processed[col])
        except ValueError:
            st.error(f"Kolom '{col}' mengandung nilai non-numerik yang tidak bisa diproses.")
            raise ValueError(f"Non-numeric value in column: {col}")

    df_processed['1st_Sem_Units_Enrolled_Safe'] = np.where(df_processed['1st_Sem_Units_Enrolled'] == 0, 1, df_processed['1st_Sem_Units_Enrolled'])
    df_processed['2nd_Sem_Units_Enrolled_Safe'] = np.where(df_processed['2nd_Sem_Units_Enrolled'] == 0, 1, df_processed['2nd_Sem_Units_Enrolled'])
    df_processed['Approval_Rate_1st_Sem'] = df_processed['1st_Sem_Units_Approved'] / df_processed['1st_Sem_Units_Enrolled_Safe']
    df_processed['Approval_Rate_2nd_Sem'] = df_processed['2nd_Sem_Units_Approved'] / df_processed['2nd_Sem_Units_Enrolled_Safe']
    df_processed['Total_Units_Enrolled'] = df_processed['1st_Sem_Units_Enrolled'] + df_processed['2nd_Sem_Units_Enrolled']
    df_processed['Total_Units_Approved'] = df_processed['1st_Sem_Units_Approved'] + df_processed['2nd_Sem_Units_Approved']
    df_processed['Total_Units_Enrolled_Safe'] = np.where(df_processed['Total_Units_Enrolled'] == 0, 1, df_processed['Total_Units_Enrolled'])
    df_processed['Overall_Approval_Rate'] = df_processed['Total_Units_Approved'] / df_processed['Total_Units_Enrolled_Safe']
    df_processed['Average_Curricular_Grade'] = (df_processed['1st_Sem_Units_Grade'].astype(float) + df_processed['2nd_Sem_Units_Grade'].astype(float)) / 2
    df_processed['Financial_Health_Score'] = df_processed['Tuition_Fees_Up_To_Date'] - df_processed['Debtor'] + df_processed['Scholarship_Holder']
    df_processed = df_processed.drop(columns=['1st_Sem_Units_Enrolled_Safe', '2nd_Sem_Units_Enrolled_Safe', 'Total_Units_Enrolled_Safe'], errors='ignore')
    
    missing_final_features = [f for f in FINAL_TRAINING_FEATURES if f not in df_processed.columns]
    if missing_final_features:
        st.error(f"Fitur final yang dibutuhkan untuk model tidak dapat dibuat: {', '.join(missing_final_features)}")
        raise ValueError(f"Missing final training features after preprocessing: {', '.join(missing_final_features)}")
    
    df_final_X = df_processed[FINAL_TRAINING_FEATURES].copy()
    return df_final_X

#B. Antarmuka Pengguna (UI)
st.title("🎓 Prediksi Potensi Dropout Siswa")
st.markdown("Aplikasi prototipe ini membantu Jaya Jaya Institut mengidentifikasi siswa yang berisiko tinggi untuk putus sekolah.")

input_method = st.sidebar.radio(
    "Pilih Metode Input Data:",
    ("Input Manual (Satu Siswa)", "Unggah File (Banyak Siswa)")
)

if input_method == "Input Manual (Satu Siswa)":
    st.header("Input Data Siswa Secara Manual")
    with st.form("manual_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detail Akademik Semester 1")
            s1_units_enrolled = st.selectbox("Unit Terdaftar Sem. 1", options=S1_UNITS_ENROLLED_OPTIONS, index=DEFAULT_INDEX_6_S1_ENROLLED, key="s1_enrolled")
            s1_units_approved = st.selectbox("Unit Disetujui Sem. 1", options=S1_UNITS_APPROVED_OPTIONS, index=DEFAULT_INDEX_6_S1_APPROVED, key="s1_approved")
            s1_units_grade = st.number_input("Nilai Rata-rata Sem. 1 (0.00 - 18.88)", min_value=0.0, max_value=18.875, value=10.64, step=0.01, format="%.2f", key="s1_grade")
        with col2:
            st.subheader("Detail Akademik Semester 2 & Keuangan")
            s2_units_enrolled = st.selectbox("Unit Terdaftar Sem. 2", options=S2_UNITS_ENROLLED_OPTIONS, index=DEFAULT_INDEX_6_S2_ENROLLED, key="s2_enrolled")
            s2_units_approved = st.selectbox("Unit Disetujui Sem. 2", options=S2_UNITS_APPROVED_OPTIONS, index=DEFAULT_INDEX_6_S2_APPROVED, key="s2_approved")
            s2_units_grade = st.number_input("Nilai Rata-rata Sem. 2 (0.00 - 18.57)", min_value=0.0, max_value=18.5714, value=10.23, step=0.01, format="%.2f", key="s2_grade")
            st.markdown("---")
            tuition_fees_up_to_date = st.radio("Pembayaran SPP Lancar?", options=[1, 0], format_func=lambda x: YES_NO_OPTIONS[x], key="spp_lancar", horizontal=True)
            debtor = st.radio("Status Debitur (Memiliki Hutang)?", options=[1, 0], format_func=lambda x: YES_NO_OPTIONS[x], key="debtor_status", horizontal=True, index=1)
            scholarship_holder = st.radio("Penerima Beasiswa?", options=[1, 0], format_func=lambda x: YES_NO_OPTIONS[x], key="scholarship", horizontal=True, index=1)
        
        submitted = st.form_submit_button("Prediksi Potensi Dropout")
        if submitted:
            input_data_dict = {
                '1st_Sem_Units_Approved': s1_units_approved, '1st_Sem_Units_Enrolled': s1_units_enrolled, '1st_Sem_Units_Grade': s1_units_grade,
                '2nd_Sem_Units_Approved': s2_units_approved, '2nd_Sem_Units_Enrolled': s2_units_enrolled, '2nd_Sem_Units_Grade': s2_units_grade,
                'Tuition_Fees_Up_To_Date': tuition_fees_up_to_date, 'Debtor': debtor, 'Scholarship_Holder': scholarship_holder,
            }
            input_df = pd.DataFrame([input_data_dict])
            try:
                with st.spinner("Memproses data dan membuat prediksi..."):
                    processed_input = preprocess_data(input_df.copy())
                    if not isinstance(processed_input, pd.DataFrame) or not all(col in processed_input.columns for col in FINAL_TRAINING_FEATURES):
                         st.error("Pra-pemrosesan gagal menghasilkan fitur yang dibutuhkan model.")
                    else:
                        processed_input = processed_input[FINAL_TRAINING_FEATURES] # Ensure column order
                        prediction = xgb_model.predict(processed_input)
                        prediction_proba = xgb_model.predict_proba(processed_input)
                        st.subheader("Hasil Prediksi:")
                        predicted_status = STATUS_MAPPING[prediction[0]]
                        if predicted_status == "Dropout": st.error(f"Siswa Diprediksi: **{predicted_status} (Risiko TINGGI)**")
                        elif predicted_status == "Enrolled": st.warning(f"Siswa Diprediksi: **{predicted_status} (Risiko Sedang/Lanjut Studi)**")
                        else: st.success(f"Siswa Diprediksi: **{predicted_status} (Risiko RENDAH/Lulus)**")
                        st.write(f"Probabilitas Prediksi:")
                        proba_df = pd.DataFrame({'Status': list(STATUS_MAPPING.values()), 'Probabilitas': prediction_proba[0]}).set_index('Status')
                        st.dataframe(proba_df.style.format("{:.2%}"))
            except ValueError as ve: st.error(f"Kesalahan Input atau Data: {ve}")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses atau memprediksi data: {e}")
                st.exception(e)

elif input_method == "Unggah File (Banyak Siswa)":
    st.header("Unggah File Data Siswa (.csv atau .xlsx)")
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Read the file content into a BytesIO object to allow sniffing or retrying parse
            file_content = uploaded_file.getvalue()
            from io import BytesIO, StringIO

            if uploaded_file.name.endswith('.csv'):
                try:
                    # Attempt to read with comma delimiter first (common default)
                    input_df_batch = pd.read_csv(BytesIO(file_content))
                    # Check if parsing was successful (e.g. more than 1 column)
                    if len(input_df_batch.columns) <= 1 and ';' in input_df_batch.columns[0]:
                         # If only one column and it contains semicolons, likely semicolon delimited
                         st.info("Terdeteksi potensi pemisah kolom titik-koma (semicolon), mencoba membaca ulang...")
                         input_df_batch = pd.read_csv(BytesIO(file_content), delimiter=';')
                    elif len(input_df_batch.columns) <=1 : # Still one column, try to sniff or raise error
                         st.warning("Gagal mem-parsing CSV dengan benar, hanya satu kolom terdeteksi. Pastikan delimiter (koma atau titik-koma) benar.")
                except pd.errors.ParserError as pe:
                    st.warning(f"Gagal membaca CSV dengan pemisah koma. Mencoba dengan titik-koma (semicolon)... Error: {pe}")
                    try:
                        input_df_batch = pd.read_csv(BytesIO(file_content), delimiter=';')
                    except Exception as e_semi:
                        st.error(f"Gagal membaca file CSV dengan pemisah koma maupun titik-koma. Pastikan format file benar. Error: {e_semi}")
                        st.stop() # Stop execution if parsing fails
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
                    st.stop()


            elif uploaded_file.name.endswith('.xlsx'):
                try:
                    input_df_batch = pd.read_excel(BytesIO(file_content))
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membaca file Excel: {e}")
                    st.stop()
            else:
                st.error("Format file tidak didukung. Harap unggah file .csv atau .xlsx")
                st.stop()
            
            st.write("Data yang Diunggah (Pratinjau 5 Baris Pertama sebelum standardisasi nama kolom):")
            st.dataframe(input_df_batch.head())

            # Rename columns based on the notebook mapping
            original_cols_before_rename = input_df_batch.columns.tolist()
            input_df_batch.rename(columns=NEW_FEATURE_NAMES_FROM_NOTEBOOK, inplace=True)
            renamed_cols = [col for col, orig_col in zip(input_df_batch.columns, original_cols_before_rename) if col != orig_col]
            if renamed_cols:
                st.info(f"Beberapa nama kolom distandarisasi. Contoh: {renamed_cols[:3]}")


            if st.button("Prediksi Dropout untuk File Ini"):
                with st.spinner("Memproses file dan membuat prediksi batch..."):
                    missing_cols = [col for col in ORIGINAL_FEATURES_FOR_FE if col not in input_df_batch.columns]
                    if missing_cols:
                        st.error(f"File yang diunggah (setelah potensi renaming) tidak memiliki kolom standar yang dibutuhkan: {', '.join(missing_cols)}")
                        st.warning(f"Pastikan file Anda memiliki kolom yang sesuai dengan {', '.join(ORIGINAL_FEATURES_FOR_FE)} atau nama asli yang dapat dipetakan (lihat panduan).")
                        st.write(f"Kolom yang terdeteksi setelah renaming: {input_df_batch.columns.tolist()}")

                    else:
                        try:
                            processed_input_batch = preprocess_data(input_df_batch.copy())
                            if not isinstance(processed_input_batch, pd.DataFrame) or not all(col in processed_input_batch.columns for col in FINAL_TRAINING_FEATURES):
                                st.error("Pra-pemrosesan batch gagal menghasilkan fitur yang dibutuhkan model.")
                            else:
                                processed_input_batch = processed_input_batch[FINAL_TRAINING_FEATURES] # Ensure column order
                                predictions_batch = xgb_model.predict(processed_input_batch)
                                predictions_proba_batch = xgb_model.predict_proba(processed_input_batch)
                                results_df = input_df_batch.copy()
                                results_df['Predicted_Status_Encoded'] = predictions_batch
                                results_df['Predicted_Status'] = results_df['Predicted_Status_Encoded'].map(STATUS_MAPPING)
                                for i, status_label in STATUS_MAPPING.items():
                                    results_df[f'Probabilitas_{status_label}'] = predictions_proba_batch[:, i]
                                st.subheader("Hasil Prediksi Batch:")
                                display_cols = [col for col in ORIGINAL_FEATURES_FOR_FE if col in results_df.columns][:2] + \
                                               ['Predicted_Status'] + \
                                               [f'Probabilitas_{label}' for label in STATUS_MAPPING.values()]
                                display_cols = [col for col in display_cols if col in results_df.columns] # Ensure all display_cols exist
                                st.dataframe(results_df[display_cols])
                                csv_output = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="Unduh Hasil Prediksi sebagai CSV", data=csv_output, file_name="hasil_prediksi_dropout_batch.csv", mime="text/csv")
                        except ValueError as ve: st.error(f"Kesalahan pada Data Batch: {ve}")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat memproses file batch: {e}")
                            st.exception(e)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca atau memproses file awal: {e}")
            st.exception(e)

#E. Tampilan Tambahan (Opsional)
st.sidebar.markdown("---")
st.sidebar.subheader("Tentang Model")
st.sidebar.info("Model Klasifikasi XGBoost digunakan untuk memprediksi status siswa (Dropout, Enrolled, Graduate). Model ini dilatih pada data historis dengan rekayasa fitur yang telah ditentukan.")
st.sidebar.subheader("Panduan Penggunaan")
st.sidebar.info(f"""
1.  **Pilih Metode Input**: Manual atau Unggah File.
2.  **Input Manual**: Isi semua field.
3.  **Unggah File**: File .csv/.xlsx. 
    * Untuk file CSV, aplikasi akan mencoba membaca dengan pemisah koma (,) terlebih dahulu, kemudian titik-koma (;) jika gagal atau hasilnya mencurigakan.
    * Pastikan memiliki kolom yang dibutuhkan. Aplikasi akan mencoba menstandarisasi nama kolom umum. Kolom inti yang dibutuhkan (setelah standardisasi) adalah:
        `{', '.join(ORIGINAL_FEATURES_FOR_FE)}`.
    * Nama kolom asli yang umum dikenali dan akan otomatis disesuaikan meliputi (namun tidak terbatas pada):
        `Curricular_units_1st_sem_enrolled`, `Tuition_fees_up_to_date`, dll.
4.  Klik tombol 'Prediksi'.
5.  Lihat hasil. Untuk unggah file, Anda dapat mengunduh hasilnya.
""")
st.sidebar.markdown("---")
st.sidebar.caption("Versi Aplikasi: 1.19") # Incremented version
