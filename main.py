import fileinput
from msilib.schema import File
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import gauss
import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete
import os
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.ppo.policies import MlpPolicy

# Operations
def operations (Tab1,Tab2):
    sl = 3000
    bl = 1400
    totalheure1 = 0
    totalheure2 = 0
    
    for i in range(3):
        for j in range(30):
            if Tab1[i][j] == 1:
                totalheure1 = totalheure1 + 8
            j += 1
        i += 1 
    
    for k in range(3):
        for l in range(30):
            if Tab2[k][l] == 1:
                totalheure2 = totalheure2 + 8
            l += 1
        k += 1
    
    totalheure = totalheure1 + totalheure2
         
    cpsj1 = 0
    cpsa1 = 0
    cpsn1 = 0
    cpsj2 = 0
    cpsa2 = 0
    cpsn2 = 0
    
    i=0 
    for i in range(30):
        if Tab1[0][i] == 1:
            Q = 889
        elif Tab1[0][i] == 2:
            Q = 212
        elif Tab1[0][i] == 3:
            Q = 78
        elif Tab1[0][i] == 4:
            Q = 121
        elif Tab1[0][i] == 5:
            Q = 368
        else :
            Q = 0
        cpsj1 = cpsj1 + 0.811*8*Q
    
    i=0
    for i in range(30):
        if Tab1[1][i] == 1:
            Q = 889
        elif Tab1[1][i] == 2:
            Q = 212
        elif Tab1[1][i] == 3:
            Q = 78
        elif Tab1[1][i] == 4:
            Q = 121
        elif Tab1[1][i] == 5:
            Q = 368
        else :
            Q = 0
        cpsa1 = cpsa1 + 1.568*8*Q
    
    i=0
    for i in range(30):
        if Tab1[2][i] == 1:
            Q = 889
        elif Tab1[2][i] == 2:
            Q = 212
        elif Tab1[2][i] == 3:
            Q = 78
        elif Tab1[2][i] == 4:
            Q = 121
        elif Tab1[2][i] == 5:
            Q = 368
        else:
            Q = 0
        cpsn1 = cpsn1 + 0.543*8*Q
    
    cpm1 = cpsj1+cpsa1+cpsn1
    
    i=0 
    for i in range(30):
        if Tab2[0][i] == 1:
            Q = 945
        elif Tab2[0][i] == 2:
            Q = 225
        elif Tab2[0][i] == 3:
            Q = 83
        elif Tab2[0][i] == 4:
            Q = 128
        elif Tab2[0][i] == 5:
            Q = 391
        else :
            Q = 0
        cpsj2 = cpsj2 + 0.811*8*Q
    
    i=0
    for i in range(30):
        if Tab2[1][i] == 1:
            Q = 945
        elif Tab2[1][i] == 2:
            Q = 225
        elif Tab2[1][i] == 3:
            Q = 83
        elif Tab2[1][i] == 4:
            Q = 128
        elif Tab2[1][i] == 5:
            Q = 391
        else :
            Q = 0
        cpsa2 = cpsa2 + 1.568*8*Q
    
    i=0
    for i in range(30):
        if Tab2[2][i] == 1:
            Q = 945
        elif Tab2[2][i] == 2:
            Q = 225
        elif Tab2[2][i] == 3:
            Q = 83
        elif Tab2[2][i] == 4:
            Q = 128
        elif Tab2[2][i] == 5:
            Q = 391
        else:
            Q = 0
        cpsn2 = cpsn2 + 0.543*8*Q
    
    cpm2 = cpsj2+cpsa2+cpsn2
    
    # Cost per month
    cpm = cpm1 + cpm2
    # Cost per hour
    cph = cpm/(totalheure)
    #nombre de pieces ligne 1 
    nb1 = sl*totalheure1/8
    #nombre de pieces ligne 2
    nb2 = bl*totalheure2/8
    #nombre de pieces total
    nb = nb1 + nb2
    
    # Ratio nombre de pieces cout par heure 
    R = nb/cph
    
    return totalheure,nb,cph,R

# Color filter
def cell_color(val):
    if val == 1 :
        color = 'green'
    elif val == 2:
        color = 'red'
    elif val == 3:
        color = 'yellow'
    elif val == 4:
        color = 'cyan'
    elif val == 5:
        color = 'darkblue'
    else:
        color = 'black'
    return color

#env
class TabEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        # Actions PR = 1, S7 = 2, S3 = 3, R3 = 4, R0 = 5, OFF = 0
        self.action_space = MultiDiscrete([6]*180)
        self.observation_space = Box(low=0, high=3000, shape=(1,), dtype=np.float32)
        # Initialisation
        #self.Tab1 = np.zeros((3,30), dtype= int)
        #self.Tab2 = np.zeros((3,30), dtype= int)
        self.Tab1 = np.random.randint(6 , size = (3,30))
        self.Tab2 = np.random.randint(6 , size = (3,30))
        
        # Set start temp
        self.state = operations(self.Tab1, self.Tab2)[2]
        

        self.nb_iter = 10000
        self.state_init = 0


    def step(self, action):
        for i in range(1,181,1):
            if i<=30:
                self.Tab1[0][i-1] = action[i-1]
            elif i<=60:
                self.Tab1[1][i-31] = action[i-1]
            elif i<=90:
                self.Tab1[2][i-61] = action[i-1]
            elif i<=120:
                self.Tab2[0][i-91] = action[i-1]
            elif i<=150:
                self.Tab2[1][i-121] = action[i-1]
            elif i<=180:
                self.Tab2[2][i-151] = action[i-1]
        
        # Elimination du 4ème shift
        self.weekends = [6,13,20,27,37,44,51,58,67,74,81,88,97,104,111,118,127,134,141,148,157,164,171,178]
        for i in self.weekends:
            if i<30:
                self.Tab1[0][i] = 0
            elif i<60:
                self.Tab1[1][i-31] = 0
            elif i<90:
                self.Tab1[2][i-61] = 0
            elif i<120:
                self.Tab2[0][i-91] = 0
            elif i<150:
                self.Tab2[1][i-121] = 0
            elif i<180:
                self.Tab2[2][i-151] = 0
            
            
        self.state = operations(self.Tab1, self.Tab2)[2]

        self.nb_iter -= 1
        
        # Calculate reward
        reward = 0
        
        if self.state>700:
            reward += 2.17*(810 - self.state )/10000
        
        if operations(self.Tab1, self.Tab2)[0] >= 1200:
            reward = -5000
            done = True
            self.nb_iter = 0
        
        for numj in range(60):
            for nums in range(3):
                if nums == 0 or nums == 2 :
                    if numj<=29:
                        if numj!=5 or numj!=6 or numj!=12 or numj!=13 or numj!=19 or numj!=20 or numj!=26 or numj!=27:
                            if self.Tab1[nums][numj] == 1:
                                reward += 7/10000
                    else:
                        if numj!=35 or numj!=36 or numj!=42 or numj!=43 or numj!=49 or numj!=50 or numj!=56 or numj!=57:
                            if self.Tab2[nums][numj-30] == 1:
                                reward += 7/10000
        
        for numj in range(60):
            for nums in range(3):
                if numj<=29:
                    if numj==6 or numj==5 or numj==12 or numj==13 or numj==20 or numj==21 or numj==26 or numj==27:
                        if self.Tab1[nums][numj] == 1:
                            reward -= 6/10000
                else:
                    if numj==36 or numj==35 or numj==43 or numj==42 or numj==50 or numj==49 or numj==56 or numj==57:
                        if self.Tab2[nums][numj-30] == 1:
                            reward -= 6/10000
                            
        for j in range (60):
            for i in range (3):
                if j <=29:
                    if self.Tab1[i][j] == 1:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 30:
                                numj = j
                        if self.Tab1[nums][numj] == 5 or self.Tab1[nums][numj] == 4 :
                            reward -= 10/10000
                            
                    if self.Tab1[i][j] == 0:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 30:
                                numj = j
                        if self.Tab1[nums][numj] == 1 or self.Tab1[nums][numj] == 2 or self.Tab1[nums][numj] == 3 or self.Tab1[nums][numj] == 4 :
                            reward -= 26/10000
                    
                    if self.Tab1[i][j] == 2:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 30:
                                numj = j
                        if self.Tab1[nums][numj] == 0 or self.Tab1[nums][numj] == 3 or self.Tab1[nums][numj] == 5 or self.Tab1[nums][numj] == 4 :
                            reward -= 10/10000
                            
                    if self.Tab1[i][j] == 3:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 30:
                                numj = j
                        if self.Tab1[nums][numj] == 0 or self.Tab1[nums][numj] == 1 or self.Tab1[nums][numj] == 2 or self.Tab1[nums][numj] == 5 :
                            reward -= 18/10000
                            
                    if self.Tab1[i][j] == 4:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 30:
                                numj = j
                        if self.Tab1[nums][numj] == 0 or self.Tab1[nums][numj] == 3 or self.Tab1[nums][numj] == 4 or self.Tab1[nums][numj] == 5 :
                            reward -= 10/10000
                            
                    if self.Tab1[i][j] == 5:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 30:
                                numj = j
                        if self.Tab1[nums][numj] == 0 or self.Tab1[nums][numj] == 4 or self.Tab1[nums][numj] == 5 :
                            reward -= 10/10000
                            
                    
                else:
                    if self.Tab2[i][j-30] == 1:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 1
                            numj = j+1
                            if numj == 60:
                                numj = j
                        if self.Tab2[nums][numj-30] == 5 or self.Tab2[nums][numj-30] == 4 :
                            reward -= 10/10000
                            
                    if self.Tab2[i][j-30] == 0:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 60:
                                numj = j
                        if self.Tab2[nums][numj-30] == 1 or self.Tab2[nums][numj-30] == 2 or self.Tab2[nums][numj-30] == 3 or self.Tab2[nums][numj-30] == 4 :
                            reward -= 26/10000
                    
                    if self.Tab2[i][j-30] == 2:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 60:
                                numj = j
                        if self.Tab2[nums][numj-30] == 0 or self.Tab2[nums][numj-30] == 3 or self.Tab2[nums][numj-30] == 5 or self.Tab2[nums][numj-30] == 4 :
                            reward -= 10/10000
                            
                    if self.Tab2[i][j-30] == 3:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 60:
                                numj = j
                        if self.Tab2[nums][numj-30] == 0 or self.Tab2[nums][numj-30] == 1 or self.Tab2[nums][numj-30] == 2 or self.Tab2[nums][numj-30] == 5 :
                            reward -= 18/10000
                            
                    if self.Tab2[i][j-30] == 4:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 60:
                                numj = j
                        if self.Tab2[nums][numj-30] == 0 or self.Tab2[nums][numj-30] == 3 or self.Tab2[nums][numj-30] == 4 or self.Tab2[nums][numj-30] == 5 :
                            reward -= 11/10000
                            
                    if self.Tab2[i][j-30] == 5:
                        nums = i+1
                        numj = j
                        if nums > 2:
                            nums = 0
                            numj = j+1
                            if numj == 60:
                                numj = j
                        if self.Tab2[nums][numj-30] == 0 or self.Tab2[nums][numj-30] == 4 or self.Tab2[nums][numj-30] == 5 :
                            reward -= 11/10000
                            
                               
                
        #actualisation       
        self.state_init = self.state
        
        # Check if all cases was done
        if self.nb_iter != 0 :
            done = False
        else:
            done = True
            
        info = {}
        # Return step information
        return np.array([self.state]).astype(np.float32), reward, done, info


    def reset(self):
        self.Tab1 = np.random.randint(6 , size = (3,30))
        self.Tab2 = np.random.randint(6 , size = (3,30))
        self.state = operations(self.Tab1, self.Tab2)[2]
        self.nb_iter = 10000
        self.state_init = 0
        return np.array([self.state]).astype(np.float32)


    def render(self):
        # Implement viz
        df1 = pd.DataFrame(data = self.Tab1 , columns=('day %d' % (i+1) for i in range(30)), index = ('Jour','Aprem','Nuit'))
        df2 = pd.DataFrame(data = self.Tab2 , columns=('day %d' % (i+1) for i in range(30)), index = ('Jour','Aprem','Nuit'))
        style1 = df1.style.applymap(lambda x: f"background-color: {cell_color(x)}")
        style2 = df2.style.applymap(lambda x: f"background-color: {cell_color(x)}")
        
        with pd.ExcelWriter('data/output.xlsx') as writer:  
            style1.to_excel(writer, sheet_name='Ligne_1')
            style2.to_excel(writer, sheet_name='Ligne_2')
        
        # print(self.Tab1)
        # print(self.Tab2,'\n')
        pass

env = TabEnv()

#option menu
choose = option_menu("RL_Planning",["Home","Quick plan","Custom plan","Load model","About"],
icons=['house','pie-chart-fill','graph-up','pin-map','person lines fill'],
menu_icon = "list", default_index=0,
styles={
"container": {"padding": "5!important", "background-color": ""},
"icon": {"color": "orange", "font-size": "18px"},
"nav-link": {"font-size": "10px", "text-align": "left", "margin":"5px", "--hover-color": ""},
"nav-link-selected": {"background-color": ""},
},orientation = "horizontal"
)


if choose == "Home":
    st.markdown("")
    st.title("Welcome to Reinforcement Learning planning Dashboard")
    st.markdown("#### Select :")
    st.text("Quick plan : to quickly generate a planing without any input")
    st.text("custom plan : to generate a planing using a pretrained model")
    st.text("Load model : to generate a planing using your own model")
    st.text("About : for infos about this dashboard.")
    image1 = Image.open('data/Legend.JPG')
    st.markdown("#### Legend :")
    st.image(image1, caption='')
    st.text("")
    _,_,t,_ = st.columns(4)
    t.text(";)")


if choose == "Quick plan":
    st.text("")

    _,_,t,_,_ = st.columns(5)
    plan = t.button("Quick plan")


    if plan :
        df1 = pd.read_excel('data/output1.xlsx', sheet_name='Ligne_1')
        df2 = pd.read_excel('data/output1.xlsx', sheet_name='Ligne_2')
        style1 = df1.style.applymap(lambda x: f"background-color: {cell_color(x)}")
        style2 = df2.style.applymap(lambda x: f"background-color: {cell_color(x)}")
        st.text('Line 1')
        st.dataframe(style1)
        st.text('Line 2')
        st.dataframe(style2)

        st.text('')
        st.text(f'Energy cost (MAD per hour): 839')
        st.text('')
        with open("data/output1.xlsx", "rb") as file:
            btn = st.download_button(
                    label="Download Planning",
                    data=file,
                    file_name="Planning.xlsx",
                    mime="excel/xlsx"
                )


if choose == "Custom plan":

    st.markdown("#### Here is some stats")
    
    image = Image.open('data/tsb.PNG')
    st.image(image, caption='Tensorboard')

    t1,t2= st.columns(2)
    algo = t1.selectbox(
     'Select an algorithm',
     ('A2C', 'PPO'))
    timestep = t2.selectbox(
     'Select a timestep',
     ('5000', '10000','15000','20000','25000','30000','35000','40000','45000','50000','55000','60000','65000','70000','75000','80000','85000','90000','95000','100000','105000','110000','115000','120000'))
    
    plan = st.button("Predict planning")

    if plan:
        st.text('!This will take a while ;)')
        if algo == 'A2C':
            models_dir = "models/A2C"
            model_path = f"{models_dir}/{timestep}.zip"
            model = A2C.load(model_path, env = env)

        elif algo == 'PPO':
            models_dir = "models/PPO"
            model_path = f"{models_dir}/{timestep}.zip"
            model = PPO.load(model_path, env = env)
        
        obs = env.reset()
        done = False
        score = 0 

        while not done:
            action = model.predict(obs, deterministic=False)
            action = action[0]
            obs, reward, done, info = env.step(action)
            score+=reward
        # print('cph:{} Score:{}'.format(obs, score))
        env.render()

        df1 = pd.read_excel('data/output.xlsx', sheet_name='Ligne_1')
        df2 = pd.read_excel('data/output.xlsx', sheet_name='Ligne_2')
        style1 = df1.style.applymap(lambda x: f"background-color: {cell_color(x)}")
        style2 = df2.style.applymap(lambda x: f"background-color: {cell_color(x)}")
        st.text('Line 1')
        st.dataframe(style1)
        st.text('Line 2')
        st.dataframe(style2)
        st.text('')

        st.text(f'Energy cost (MAD per hour): {obs}')
        st.text('')

        with open("data/output.xlsx", "rb") as file:
            btn = st.download_button(
                    label="Download planing",
                    data=file,
                    file_name="Planning.xlsx",
                    mime="excel/xlsx"
                )


if choose == "Load model":

    t1,t2= st.columns(2)
    algo = t1.selectbox(
     'Select an algorithm',
     ('A2C', 'PPO'))
    uploaded_file = t2.file_uploader("Load your model here", type= "zip")
    plan = st.button("Predict planning")


    if plan:
        if uploaded_file is None:
            st.text('Upload a model first')
        else:
            if algo == 'A2C':
                model = A2C.load(uploaded_file, env = env)
            if algo == 'PPO':
                model = PPO.load(uploaded_file, env = env)

            obs = env.reset()
            done = False
            score = 0 

            while not done:
                action = model.predict(obs, deterministic=False)
                action = action[0]
                obs, reward, done, info = env.step(action)
                score+=reward
            # print('cph:{} Score:{}'.format(obs, score))
            env.render()

            df1 = pd.read_excel('data/output.xlsx', sheet_name='Ligne_1')
            df2 = pd.read_excel('data/output.xlsx', sheet_name='Ligne_2')
            style1 = df1.style.applymap(lambda x: f"background-color: {cell_color(x)}")
            style2 = df2.style.applymap(lambda x: f"background-color: {cell_color(x)}")
            st.text('Line 1')
            st.dataframe(style1)
            st.text('Line 2')
            st.dataframe(style2)
            st.text('')

            st.text(f'Energy cost (MAD per hour): {obs}')
            st.text('')

            with open("data/output.xlsx", "rb") as file:
                btn = st.download_button(
                        label="Download planing",
                        data=file,
                        file_name="Planning.xlsx",
                        mime="excel/xlsx"
                    )


if choose == 'About':
    st.text('Version: 1.1')
    st.text('Date: 07-21-2022')
    st.text('License: ENSASM-Meknes')




    # def convert_df(df):
    # # IMPORTANT: Cache the conversion to prevent computation on every rerun
    #     return df.to_csv().encode('utf-8')

    # csv1 = convert_df(dataframe1)

    # st.download_button(
    #     label="Download data as CSV",
    #     data=csv1,
    #     file_name='large_df.csv',
    #     mime='text/csv',
    # )



    
# header = st.container()
# dataset = st.container()
# features = st.container()
# modelTraining = st.container()

# while header:
#     st.title("welcome to this project")


    # d, e,f,g = st.columns(4)
    # q = d.text_input('Movie title')
    # st.write(q)
    # a, b = st.sidebar.columns(2)
    # a.text_input('Movie')
    # b.text_input(q)
    # option = st.multiselect(
    #  'How would you like to be contacted?',
    #  ('Email', 'Home phone', 'Mobile phone'))

    # st.write('You selected:', option)
    # if option == "Email":
        # st.header("EmAil innnnn")

    # fig,ax = plt.subplots()
    # plt.plot([1,2,3], [10,4,9])
    # plt.show()
    # st.pyplot(fig)




    # uploaded_file = st.file_uploader("Choose a file", type= "csv")
    # if uploaded_file is not None:
    #     # To read file as bytes:
    #     bytes_data = uploaded_file.getvalue()
    #     # st.write(bytes_data)

    #     # Can be used wherever a "file-like" object is accepted:
    #     dataframe = pd.read_csv(uploaded_file)
    #     st.dataframe(dataframe)
    #     def convert_df(df):
    #  # IMPORTANT: Cache the conversion to prevent computation on every rerun
    #         return df.to_csv().encode('utf-8')

    #     csv = convert_df(dataframe)

    #     st.download_button(
    #         label="Download data as CSV",
    #         data=csv,
    #         file_name='large_df.csv',
    #         mime='text/csv',
    #     )