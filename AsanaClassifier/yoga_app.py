import numpy as np
import pandas as pd
import base64

import streamlit as st


from utils.yogasana_inference import *

# https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json


########### SIDEBARS ########### 
#### FUNCTION TO LOAD CSS FILE ####
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("utils/style.css")


########### ALL HELPER FUNCTIONS ###########  
#### FUNCTION TO INFER ON VALIDATION FOLDER ####
def check_model_validation():
    t = "<div><span class='buttons'>Inference on Input Folder</div>"
    st.markdown(t, unsafe_allow_html=True)
    user_input = st.text_input("Please enter the full path of the image in this text box.")
    t = "<div><span class='fineprint'>The folder structure generally used in training any DL model in PyTorch **must** be followed.</div>"
    st.markdown(t, unsafe_allow_html=True)
    infer = False

    # Cache is used since inference on folders might take long time
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def expensive_computation(user_input):
        act, pred = yoga_inf.validate_batch(user_input, model="fine_tune")
        return act, pred
    
    if os.path.exists(user_input) and infer==False:
        act, pred = expensive_computation(user_input)
        infer = True
    elif user_input !='':
        st.error("ERROR: No folder has been found in the given path.")

    # Checking results as dataframe or confusion matrrix 
    if infer:
        option = st.selectbox('How would you like see the results?',('','Confusion Matrix', 'Data Frame'),\
                               format_func=lambda x: 'Select an option' if x == '' else x)

        if option == 'Confusion Matrix':        
            #Getting the confusion matrix
            cm = compute_confusion_matrix(class_names, act.numpy(), torch.argmax(pred, axis=1).numpy())
            conf_mat = make_confusion_matrix(cm, categories=class_names,figsize=(10,10))
            st.pyplot()

            t = "<div><span class='fineprint'>You can right click on the image and save if required</div>"
            st.markdown(t, unsafe_allow_html=True)

        if  option == 'Data Frame':
            def highlight(s):
                if s['actuals'] != s['predicted_1'] :
                    return ['background-color: yellow']*len(s)
                else:
                    return ['background-color: white']*len(s)

            df = analyse_preds(act, pred, class_names, error_only=False)
            st.dataframe(df.style.apply(highlight, axis=1))

            st.markdown("<div><span class='finebold'>NOTE</div>",unsafe_allow_html=True)
            t = "<div><span class='fineprint'>predicted_1 and predicted_2 are the best and second best classification value for a given image</div>"
            st.markdown(t, unsafe_allow_html=True)

            t = "<div><span class='fineprint'>The errors (if any) are highlighted in yellow</div>"
            st.markdown(t, unsafe_allow_html=True)

            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download as a csv file</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)


#### FUNCTION TO INFER ON SINGLE IMAGES ####
def check_model_inference():
    t = "<div><span class='buttons'>Inference on Input File</div>"
    st.markdown(t, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Add a valid image file")
    if uploaded_file != None:
        try:
            out, _ = yoga_inf.classify_asana(uploaded_file, model="fine_tune",batch=False)
            img = Image.open(uploaded_file)
            st.image(img, width=224)
            st.write('')
            output_string = out[0]            
            t = "<div>The predicted asana is <span class='ophighlight blue'><span class='bold'>" + output_string + "</span></span></div>"
            st.markdown(t, unsafe_allow_html=True)
        except:
            st.error("ERROR: Please make sure the uploaded file is a valid image file")


#### LOAD MODEL ####
classnames_path = "models/class_names.txt"
model_paths = ["models/finetuned_model.pth", "models/convfeat_model.pth"]
with open(classnames_path,"r") as op:
    class_names = [i.strip() for i in op.readlines()]

yoga_inf = YogaInference(model_paths, class_names)

#### HEADER AND OTHER INFO ####
t = "<div><span class='header'> ASANA CLASSFIER: GUESSING IT RIGHT.. </div>"
st.markdown(t, unsafe_allow_html=True)
t = "<div><span class='sider'> ...or wrong! </div>"
st.markdown(t, unsafe_allow_html=True)

st.write('')
st.write('')
st.write("The DL model being used is a purely fine-tuned ResNet-18 model, trained to classify 10 yoga asanas.") 
st.write("The supported asanas are _adhomukha-shwanasana,\
          balasana, paschimottanasana, phalakasana, sethu_bandhasana, tadasana, trikonasana, veerabhadrasana_1, veerabhadrasana_2\
          and vrikshasana._")



#### SELECT SINGLE IMAGE OR BATCH VALIDATION ####
t = "<div><span class='subtitle'>CHOOSE A METHOD TO EXPLORE MODEL PERFORMANCE</div>"
st.markdown(t, unsafe_allow_html=True)
model_validation = 'Model Validation (checks for model validation on a set of unseen data..)'
model_inference = 'Model Inference (yields model output for images or set of images..)'

validation = st.radio("",(model_validation, model_inference))

if validation == model_validation:
    check_model_validation()
else:
    check_model_inference()





#### SIDEBARS ####
#### These are all the way down since static links and elements are loaded wuickly in streamlit ####
t = "<div><span class='highlight blue'>Links to dataset used in model ⛽ <span class='bold'></div>"
st.sidebar.markdown(t, unsafe_allow_html=True)
st.sidebar.markdown('')
st.sidebar.markdown('[Oregon State Univeristy](https://oregonstate.app.box.com/s/4c5o6gilogogdm9m23tgtop7vw23e9vj)')
st.sidebar.markdown('[10 Yoga Poses](https://www.amarchenkova.com/2018/12/04/data-set-convolutional-neural-network-yoga-pose/)')

st.sidebar.markdown('')
t = "<div><span class='highlight blue'>Links that help in app creation 👓 <span class='bold'></div>"
st.sidebar.markdown(t, unsafe_allow_html=True)
st.sidebar.markdown('')
st.sidebar.markdown('[Landing page Streamlit](https://www.streamlit.io/)')
st.sidebar.markdown('[Streamlit 101](https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2)')
st.sidebar.markdown('[Awesome Streamlit](http://awesome-streamlit.org/)')

st.sidebar.markdown('')
t = "<div><span class='highlight blue'>Links to few awesome courses 📚 <span class='bold'></div>"
st.sidebar.markdown(t, unsafe_allow_html=True)
st.sidebar.markdown('')
st.sidebar.markdown('[CS229 -Andrew Ng](https://www.youtube.com/watch?v=jGwO_UgTS7I)')
st.sidebar.markdown('[CS231n Winter-2016](http://cs231n.stanford.edu/2016/)')
st.sidebar.markdown('[NYC DeepLearning](https://atcold.github.io/pytorch-Deep-Learning/)')

st.sidebar.markdown('')
t = "<div><span class='highlight blue'>Links to other great materials 🔧 <span class='bold'></div>"
st.sidebar.markdown(t, unsafe_allow_html=True)
st.sidebar.markdown('')
st.sidebar.markdown('[FastAI](https://course.fast.ai/)')
st.sidebar.markdown('[CTDS show](https://chaitimedatascience.com/)')

