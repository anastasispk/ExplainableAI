import streamlit as st
from PIL import Image
import requests
st.set_page_config(layout="wide")
import sys
sys.path.append('../GaitMixer/src')
from datasets.gait import CasiaQueryDataset
from datasets.augmentation import *
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import streamlit.components.v1 as components
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

url = 'https://lawgame.app.squaredev.io/v1.4/'

def display_animate(data, w=320, h=240):
    def animate(frame_num):
        ax.clear()
        x = data[frame_num,:,0]
        y = -data[frame_num,:,1]
        ax.set_xlim(0, w)
        ax.set_ylim(-h, 0)
        ax.scatter(x,y)
        for i, bone in enumerate(coco_bones):
            ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], 'r')
    fig, ax = plt.subplots()
    animate(0)
    return FuncAnimation(fig, animate, frames=data.shape[0], interval=20)

def create_session(scenario, task):
    API_ENDPOINT = url + 'session/' + '?scenario={}&task={}'.format(scenario, task)

    data = {
    'user_id': '024b4b9c-1b9a-4c3e-8c9b-2c8597c9d6f3',
    'username': 'johndoe'
    }
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    }

    response = requests.post(API_ENDPOINT, headers=headers, json=data)

    st.text(f"Status Code: {response.status_code}")
    st.json(response.json())
    j = response.json()
    if ['id'] not in st.session_state:
        st.session_state['id'] = j['session_id']

def delete_session():
    headers = {
    'accept': 'application/json',
    }
    response = requests.delete(url + 'session/' + st.session_state['id'], headers=headers)
    st.text(f"Status Code: {response.status_code}")
    st.json(response.json())

def upload_image(file_path):

    headers = {
    'accept': 'application/json',
    }

    API_ENDPOINT = url + 'processing/' + st.session_state['id'] + '/' + 'image'

    files = {
    'file': (file_path, open(file_path, 'rb'), 'image/jpeg'),
    }

    response = requests.post(API_ENDPOINT, headers=headers, files=files)
    print(response.json())

def load_casia():
    dataset = CasiaQueryDataset(
        '/home/anastasispk/Dev/GaitMixer/data/casiab_npy',
        id_range='75-75',
        duplicate_bgcl=False,
        transform=transforms.Compose([
            SelectSequenceCenter(60),
            remove_conf(enable=True),
            normalize_width,
            ToTensor()]))
        
    data_target = {}
    # subject_id, walking_status, sequence_num, view_angle
    for data in dataset:
        data_target[tuple(data[1])] = data[0]

    global coco_bones
    coco_bones = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
    ]

    animation = display_animate(data_target[(75,2,1,36)]*320)
    writervideo = FFMpegWriter(fps=10) 
    animation.save('file_name.mp4', writer=writervideo)
    sys.exit(0)

def display():
    col1, col2 = st.columns([0.5, 0.5], gap='large')

    with col1:
        # components.html(animation.to_jshtml(), height=500)
        st.video('file_name.mp4')
    with col2:
        # img = cv2.imread('Figure1.png')
        # img = cv2.resize(img, (800,600))
        # st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = Image.open('Figure1.png')
        st.image(img)


def main():

    st.title("Responsible and Explainable AI for Fair and Unbiased Biometric Technologies​")

    st.sidebar.image('./assets/logo.png')

    st.sidebar.title("Settings")

    # upload model
    scenario = st.sidebar.radio("Select scenario.", ["Gait recognition", "Behavior recognition"])

    st.subheader(scenario, divider='rainbow')

    # load_casia()

    display()

    # animation.to_jshtml()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
