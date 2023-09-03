import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import pickle

st.set_page_config(page_title="BioTech Data Miners", page_icon="🧊",)
def new_line(n = 1 ): 

    """
    Function to add new lines to the app

    Parameters
    ----------
    n : int
        Number of new lines to add

    Returns
    -------
    None.

    """

    for _ in range(n): 
        st.markdown('\n')


st.markdown("<h1 align='center'>🧊 BioTech Data Miners</h1>", unsafe_allow_html=True)
new_line()

st.write("""
         
        The BioTech Data Miners team participated in a Kaggle competition to develop 
        models for RNA degradation, using a dataset of over 3000 RNA molecules. 
         The competition was organized by the Eterna community, which uses an 
         online video game platform to solve scientific problems. The team built a 
         model and a web application to make their solution available to everyone, 
         in the hope of contributing to the fight against COVID-19.
""")


st.divider()
st.markdown("<h2 align='center'> Upload Your CSV File </h3>", unsafe_allow_html=True)
new_line()

col1, col2, col3 = st.columns([0.5,1.5,0.5])
inp = col2.file_uploader("Upload your RNA sequence in .csv format", type = ['csv'])
new_line(2)


sequence_enc_map = {'A': 0, 'G': 1, 'C': 3, 'U': 2}
structure_enc_map = {'.': 0, '(': 1, ')': 1}
looptype_enc_map = {'S': 6, 'E': 2, 'H': 0, 'I': 5, 'X': 4, 'M': 3, 'B': 1}

enc_targets = ['sequence', 'a1_sequence', 'a2_sequence', 'a3_sequence',
               'b1_sequence', 'b2_sequence', 'b3_sequence', 'b4_sequence',
               'a4_sequence', 'b5_sequence', 'a5_sequence', 'structure', 'a1_structure',
               'b1_structure', 'b2_structure', 'a2_structure', 'b3_structure', 'a3_structure',
               'b4_structure', 'a4_structure', 'a5_structure', 'b5_structure', 'predicted_loop_type',
               'b1_predicted_loop_type', 'a1_predicted_loop_type', 'b2_predicted_loop_type', 'a2_predicted_loop_type',
               'b3_predicted_loop_type', 'a3_predicted_loop_type', 'b4_predicted_loop_type', 'a4_predicted_loop_type',
               'b5_predicted_loop_type', 'a5_predicted_loop_type', 'predicted_loop_type'
               ]
enc_maps = [sequence_enc_map, sequence_enc_map, sequence_enc_map, sequence_enc_map,
            sequence_enc_map, sequence_enc_map, sequence_enc_map,
            sequence_enc_map, sequence_enc_map, sequence_enc_map, sequence_enc_map,
            structure_enc_map, structure_enc_map, structure_enc_map, structure_enc_map,
            structure_enc_map, structure_enc_map, structure_enc_map, structure_enc_map,
            structure_enc_map, structure_enc_map, structure_enc_map, structure_enc_map,
            looptype_enc_map, looptype_enc_map, looptype_enc_map,
            looptype_enc_map, looptype_enc_map, looptype_enc_map,
            looptype_enc_map, looptype_enc_map, looptype_enc_map, looptype_enc_map, looptype_enc_map
            ]



if inp is not None:

    df = pd.read_csv(inp)
    model = pickle.load(open('best_model.pkl', 'rb'))

    
    st.markdown("<h4 align='center'> RNA sequence Features", unsafe_allow_html=True)
    new_line(1)
    st.write(df)

    for target_col, enc_map in zip(enc_targets, enc_maps):
        df[target_col] = df[target_col].apply(lambda x: enc_map.get(x, -1))
        
    st.divider()
    st.markdown("<h4 align='center'> Predictions", unsafe_allow_html=True)
    pred = model.predict(df)
    new_line(2)

    cola, colb, colc = st.columns(3)
    cola.markdown("<h5 align='center'> Predicted reactivity </h4>", unsafe_allow_html=True)
    cola.markdown(f"<h4 align='center'> {round(float(pred[0,0]), 4)} </h4>", unsafe_allow_html=True)

    colb.markdown("<h5 align='center'> Predicted deg_Mg_pH10 </h4>", unsafe_allow_html=True)
    colb.markdown(f"<h4 align='center'> {round(float(pred[0,1]), 4)} </h4>", unsafe_allow_html=True)

    colc.markdown("<h5 align='center'> Predicted deg_Mg_50C </h4>", unsafe_allow_html=True)
    colc.markdown(f"<h4 align='center'> {round(float(pred[0,2]), 4)} </h4>", unsafe_allow_html=True)

    
