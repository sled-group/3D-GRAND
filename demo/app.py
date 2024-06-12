import spaces
import os
import gradio as gr
from time import sleep
from signal import SIGTERM
from psutil import process_iter
from settings import GRAND3D_Settings
from utils import list_dirs
import open3d as o3d
from copy import deepcopy
import numpy as np
import re
from bs4 import BeautifulSoup
import trimesh.transformations as tf

import logging


# The following line sets the root logger level as well.
# It's equivalent to both previous statements combined:
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from session import Session
from model import load_model_and_dataloader, get_model_response

# Load model and tokenizer once at the start
model_path = "checkpoints/merged_weights_grounded_obj_ref"
model_base = None
load_8bit = False
load_4bit = False
load_bf16 = True
scene_to_obj_mapping = "data/predicted_scene_data_update_5.json"
# scene_to_obj_mapping = "data/scanrefer_ground_truth_scene_graph.json"
max_new_tokens = 5000
obj_context_feature_type = "text"


tokenizer, model, data_loader = load_model_and_dataloader(
    model_path=model_path,
    model_base=model_base,
    load_8bit=load_8bit,
    load_4bit=load_4bit,
    load_bf16=load_bf16,
    scene_to_obj_mapping=scene_to_obj_mapping,
    device_map="cpu",
)  # Huggingface Zero-GPU has to use .to(device) to set the device, otherwise it will fail

model.to("cuda")  # Huggingface Zero-GPU requires explicit device placement


def get_chatbot_response(user_chat_input, scene_id):
    # Get the response from the model
    prompt, response = get_model_response(
        model=model,
        tokenizer=tokenizer,
        data_loader=data_loader,
        scene_id=scene_id,
        user_input=user_chat_input,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
    )
    return scene_id, prompt, response


# def get_chatbot_response(user_chat_input):
#     # Get the response from the chatbot
#     scene_id = "scene0643_00"
#     scene_graph = """
#     Object-centric context: <obj_0>: {'category': 'door', 'centroid': '[0.35, 1.99, 1.11]', 'extent': '[0.68, 0.65, 2.11]'}; <obj_1>: {'category': 'ceiling', 'centroid': '[1.04, -1.39, 2.68]', 'extent': '[0.18, 0.90, 0.05]'}; <obj_2>: {'category': 'ceiling', 'centroid': '[0.77, 2.09, 2.65]', 'extent': '[0.94, 0.86, 0.11]'}; <obj_3>: {'category': 'trash can', 'centroid': '[-0.61, -2.16, 0.21]', 'extent': '[0.42, 0.36, 0.41]'}; <obj_4>: {'category': 'chair', 'centroid': '[0.35, -1.35, 0.50]', 'extent': '[0.46, 0.47, 0.94]'}; <obj_5>: {'category': 'trash can', 'centroid': '[-0.22, -2.13, 0.24]', 'extent': '[0.40, 0.28, 0.39]'}; <obj_6>: {'category': 'cabinet', 'centroid': '[-1.24, 0.00, 0.58]', 'extent': '[0.61, 0.57, 0.79]'}; <obj_7>: {'category': 'cup', 'centroid': '[0.62, 0.23, 0.77]', 'extent': '[0.14, 0.14, 0.08]'}; <obj_8>: {'category': 'window', 'centroid': '[-0.35, -2.87, 1.13]', 'extent': '[2.05, 0.60, 1.07]'}; <obj_9>: {'category': 'file cabinet', 'centroid': '[0.40, -1.97, 0.39]', 'extent': '[0.40, 0.66, 0.73]'}; <obj_10>: {'category': 'monitor', 'centroid': '[0.92, -1.51, 0.97]', 'extent': '[0.25, 0.57, 0.47]'}; <obj_11>: {'category': 'chair', 'centroid': '[0.34, 0.59, 0.43]', 'extent': '[0.65, 0.64, 0.94]'}; <obj_12>: {'category': 'desk', 'centroid': '[0.64, 0.75, 0.57]', 'extent': '[0.76, 1.60, 0.82]'}; <obj_13>: {'category': 'chair', 'centroid': '[0.55, -0.33, 0.48]', 'extent': '[0.60, 0.60, 0.87]'}; <obj_14>: {'category': 'office chair', 'centroid': '[-0.28, 1.56, 0.46]', 'extent': '[0.67, 0.55, 1.02]'}; <obj_15>: {'category': 'office chair', 'centroid': '[-0.86, -1.53, 0.43]', 'extent': '[0.54, 0.64, 0.97]'}; <obj_16>: {'category': 'chair', 'centroid': '[-0.28, 1.56, 0.46]', 'extent': '[0.67, 0.55, 1.02]'}; <obj_17>: {'category': 'monitor', 'centroid': '[0.98, 0.56, 1.05]', 'extent': '[0.21, 0.60, 0.54]'}; <obj_18>: {'category': 'doorframe', 'centroid': '[-0.17, 2.42, 1.01]', 'extent': '[0.16, 0.18, 1.70]'}; <obj_19>: {'category': 'chair', 'centroid': '[-0.86, -1.53, 0.43]', 'extent': '[0.54, 0.64, 0.97]'}; <obj_20>: {'category': 'bookshelf', 'centroid': '[0.93, 2.00, 1.34]', 'extent': '[0.73, 0.99, 2.60]'}; <obj_21>: {'category': 'office chair', 'centroid': '[0.35, -1.35, 0.50]', 'extent': '[0.46, 0.47, 0.94]'}; <obj_22>: {'category': 'desk', 'centroid': '[-1.23, 1.60, 0.70]', 'extent': '[0.80, 2.01, 0.51]'}; <obj_23>: {'category': 'book', 'centroid': '[0.91, 1.31, 0.89]', 'extent': '[0.34, 0.32, 0.30]'}; <obj_24>: {'category': 'desk', 'centroid': '[-1.24, -1.12, 0.54]', 'extent': '[0.79, 1.88, 0.85]'}; <obj_25>: {'category': 'desk', 'centroid': '[0.63, -1.51, 0.53]', 'extent': '[0.81, 1.97, 0.85]'}; <obj_26>: {'category': 'calendar', 'centroid': '[-1.72, -0.44, 1.40]', 'extent': '[0.07, 0.88, 0.83]'}; <obj_27>: {'category': 'office chair', 'centroid': '[0.34, 0.59, 0.43]', 'extent': '[0.65, 0.64, 0.94]'}; <obj_28>: {'category': 'file cabinet', 'centroid': '[-1.02, -0.76, 0.47]', 'extent': '[0.58, 0.75, 0.81]'}; <obj_29>: {'category': 'cup', 'centroid': '[-1.26, -1.65, 0.78]', 'extent': '[0.10, 0.12, 0.04]'}; <obj_30>: {'category': 'keyboard', 'centroid': '[0.55, 0.84, 0.73]', 'extent': '[0.22, 0.15, 0.03]'}
#     """
#     response = """
#     <detailed_grounding>a <p>brown wooden office desk</p>[<obj_12>] on the left to the <p>gray shelf</p>[<obj_20>].</detailed_grounding> <refer_expression_grounding>These sentences refer to <p>the brown wooden office desk</p>[<obj_12>].</refer_expression_grounding>
#     """
#     return scene_id, scene_graph, response


# Resetting to blank
def reset_textbox():
    return gr.update(value="")


# to set a component as visible=False
def set_visible_false():
    return gr.update(visible=False)


# to set a component as visible=True
def set_visible_true():
    return gr.update(visible=True)


def change_scene_or_system_prompt(dropdown_scene_selection: str):
    # reset model_3d, chatbot_for_display, chat_counter, server_status_code
    new_session_state = Session.create_for_scene(dropdown_scene_selection)
    file_name = f"{dropdown_scene_selection}.obj"
    print(os.path.join(GRAND3D_Settings.data_path, dropdown_scene_selection, file_name))
    return (
        new_session_state,
        os.path.join(GRAND3D_Settings.data_path, dropdown_scene_selection, file_name),
        None,
        new_session_state.chat_history_for_display,
    )


def cylinder_frame(p0, p1):
    """Calculate the transformation matrix to position a unit cylinder between two points."""
    direction = np.asarray(p1) - np.asarray(p0)
    length = np.linalg.norm(direction)
    direction /= length
    # Computing rotation matrix using Rodrigues' formula
    rot_axis = np.cross([0, 0, 1], direction)
    rot_angle = np.arccos(np.dot([0, 0, 1], direction))
    rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)

    # Translation
    translation = (np.asarray(p0) + np.asarray(p1)) / 2

    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix
    transformation[:3, 3] = translation
    scaling = np.eye(4)
    scaling[2, 2] = length
    transformation = np.matmul(transformation, scaling)
    return transformation


def create_cylinder_mesh(p0, p1, color, radius=0.04, resolution=20, split=1):
    """Create a colored cylinder mesh between two points p0 and p1."""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=1, resolution=resolution, split=split
    )
    transformation = cylinder_frame(p0, p1)
    cylinder.transform(transformation)
    # Apply color
    cylinder.paint_uniform_color(color)
    return cylinder


def prettify_mesh_for_gradio(mesh):
    # Define the transformation matrix
    T = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    # Apply the transformation
    mesh.transform(T)

    mesh.scale(10.0, center=mesh.get_center())

    bright_factor = 1  # Adjust this factor to get the desired brightness
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors) * bright_factor, 0, 1)
    )

    return mesh


def create_bbox(center, extents, color=[1, 0, 0], radius=0.02):
    """Create a colored bounding box with given center, extents, and line thickness."""
    # ... [The same code as before to define corners and lines] ...
    print(extents)
    print(type(extents))
    extents = extents.replace("[", "").replace("]", "")
    center = center.replace("[", "").replace("]", "")
    extents = [float(x.strip()) for x in extents.split(",")]
    center = [float(x.strip()) for x in center.split(",")]
    angle = -np.pi / 2  # 90 degrees
    axis = [1, 0, 0]  # Rotate around x-axis
    R = tf.rotation_matrix(angle, axis)
    center_homogeneous = np.append(center, 1)
    extents_homogeneous = np.append(extents, 1)

    # Apply the rotation to the center and extents
    rotated_center = np.dot(R, center_homogeneous)[:3]
    rotated_extents = np.dot(R, extents_homogeneous)[:3]

    sx, sy, sz = rotated_extents
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + float(rotated_center[0])
    corners_3d[1, :] = corners_3d[1, :] + float(rotated_center[1])
    corners_3d[2, :] = corners_3d[2, :] + float(rotated_center[2])
    corners_3d = np.transpose(corners_3d)

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    cylinders = []
    for line in lines:
        p0, p1 = corners_3d[line[0]], corners_3d[line[1]]
        cylinders.append(create_cylinder_mesh(p0, p1, color, radius))
    return cylinders


def highlight_clusters_in_mesh(
    centroids_extents_detailed,
    centroids_extends_refer,
    mesh,
    output_dir,
    output_file_name="highlighted_mesh.obj",
):
    print("*" * 50)
    # Visualize the highlighted points by drawing 3D bounding boxes overlay on a mesh
    old_mesh = deepcopy(mesh)
    output_path = os.path.join(output_dir, "mesh_vis")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a combined mesh to hold both the original and the bounding boxes
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh += old_mesh

    # Draw bounding boxes for each centroid and extent
    for center, extent in centroids_extents_detailed:
        print("center: ", center)
        print("extent: ", extent)
        bbox = create_bbox(
            center, extent, color=[1, 1, 0]
        )  # yellow color for all boxes
        for b in bbox:
            combined_mesh += b

    for center, extent in centroids_extends_refer:
        bbox = create_bbox(center, extent, color=[0, 1, 0])
        for b in bbox:
            combined_mesh += b

    # Save the combined mesh
    output_file_path = os.path.join(output_path, output_file_name)
    o3d.io.write_triangle_mesh(
        output_file_path, combined_mesh, write_vertex_colors=True
    )
    print("*" * 50)
    return output_file_path


def extract_objects(text):
    return re.findall(r"<obj_\d+>", text)


# Parse the scene graph into a dictionary
def parse_scene_graph(scene_graph):
    scene_dict = {}
    matches = re.findall(r"<obj_(\d+)>: (\{.*?\})", scene_graph)
    for match in matches:
        obj_id = f"<obj_{match[0]}>"
        obj_data = eval(match[1])
        scene_dict[obj_id] = obj_data
    return scene_dict


def get_centroids_extents(obj_list, scene_dict):
    centroids_extents = []
    for obj in obj_list:
        if obj in scene_dict:
            centroid = scene_dict[obj]["centroid"]
            extent = scene_dict[obj]["extent"]
            centroids_extents.append((centroid, extent))
    return centroids_extents


@spaces.GPU
def language_model_forward(
    session_state, user_chat_input, top_p, temperature, dropdown_scene
):
    session_state = Session.create_for_scene(dropdown_scene)
    session_state.chat_history_for_display.append(
        (user_chat_input, None)
    )  # append in a tuple format, first is user input, second is assistant response

    yield session_state, None, session_state.chat_history_for_display

    # Load in a 3D model
    file_name = f"{session_state.scene}.obj"
    original_model_path = os.path.join(
        GRAND3D_Settings.data_path, session_state.scene, file_name
    )
    print("original_model_path: ", original_model_path)

    # Load the GLB mesh
    mesh = o3d.io.read_triangle_mesh(original_model_path)

    # get chatbot response
    scene_id, scene_graph, response = get_chatbot_response(
        user_chat_input, session_state.scene
    )

    assert scene_id == session_state.scene  # Ensure the scene ID matches

    # use scene_graph and response to get centroids and extents
    # Parse the scene graph into a dictionary
    scene_dict = parse_scene_graph(scene_graph)
    print("Model Input: " + str(scene_dict))
    print("=" * 50)
    print("Model Response: " + response)

    # Parse the response to get detailed and refer expression groundings
    soup = BeautifulSoup(response, "html.parser")
    detailed_grounding_html = str(soup.find("detailed_grounding"))
    refer_expression_grounding_html = str(soup.find("refer_expression_grounding"))

    # Extract objects from both sections
    detailed_objects = extract_objects(detailed_grounding_html)
    refer_objects = extract_objects(refer_expression_grounding_html)

    # Extract objects from both sections
    print("detailed_objects: ", detailed_objects)
    print("refer_objects: ", refer_objects)

    # Perform set subtraction to get remaining objects
    remaining_objects = list(set(detailed_objects) - set(refer_objects))
    print("remaining_objects: ", remaining_objects)

    centroids_extents_detailed = get_centroids_extents(remaining_objects, scene_dict)
    print("centroids_extents_detailed: ", centroids_extents_detailed)
    centroids_extents_refer = get_centroids_extents(refer_objects, scene_dict)
    print("centroids_extents_refer: ", centroids_extents_refer)
    # Define your centroids and extents here (example data)
    # Highlight clusters in the mesh and save it
    session_output_dir = session_state.get_session_output_dir()
    highlighted_model_path = highlight_clusters_in_mesh(
        centroids_extents_detailed,
        centroids_extents_refer,
        mesh,
        session_output_dir,
        output_file_name="highlighted_model.obj",
    )

    # Update the chat history with the response
    last_turn = session_state.chat_history_for_display[
        -1
    ]  # first is user input, second is assistant response
    last_turn = (last_turn[0], response)
    session_state.chat_history_for_display[-1] = last_turn
    session_state.save()  # save the session state

    yield session_state, highlighted_model_path, session_state.chat_history_for_display


title = """<h1 align="center">🏠💬  3D-GRAND: Towards Better Grounding and Less Hallucination for 3D-LLMs 🚀</h1>
<p><center>
<a href="https://3d-grand.github.io/" target="_blank">[Project Page]</a>
<a href="https://www.dropbox.com/scl/fo/5p9nb4kalnz407sbqgemg/AG1KcxeIS_SUoJ1hoLPzv84?rlkey=weunabtbiz17jitfv3f4jpmm1&dl=0" target="_blank">[3D-GRAND Data]</a>
<a href="https://www.dropbox.com/scl/fo/inemjtgqt2nkckymn65rp/AGi2KSYU9AHbnpuj7TWYihs?rlkey=ldbn36b1z6nqj74yv5ph6cqwc&dl=0" target="_blank">[3D-POPE Data]</a>
</center></p>
"""

# Modifying existing Gradio Theme
# theme = gr.themes.Soft(
#     primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.pink
# )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    session_state = gr.State(Session.create)

    gr.HTML(title)

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=5):
                dropdown_scene = gr.Dropdown(
                    choices=list_dirs(GRAND3D_Settings.data_path),
                    value=GRAND3D_Settings.default_scene,
                    interactive=True,
                    label="Select a scene",
                )
                model_3d = gr.Model3D(
                    value=os.path.join(
                        GRAND3D_Settings.data_path,
                        GRAND3D_Settings.default_scene,
                        f"{GRAND3D_Settings.default_scene}.obj",
                    ),
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model",
                    camera_position=(-50, 65, 10),
                    zoom_speed=10.0,
                )
                gr.HTML(
                    """<center><strong>
                    👆 SCROLL or DRAG on the 3D Model
                    to zoom in/out and rotate. Press CTRL and DRAG to pan.
                    </strong></center>
                    """
                )
                gr.HTML(
                    """<center><strong>
                    👇 When grounding finishes,
                    the grounding result will be displayed below.
                    </strong></center>
                    """
                )
                model_3d_grounding_result = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="Grounding Result",
                    zoom_speed=15.0,
                )
                gr.HTML(
                    """<center><strong>
                    <div style="display:inline-block; color:green">&#9632;</div> = Chosen Target &nbsp;
                    <div style="display:inline-block; color:yellow">&#9632;</div> = Landmarks
                    </strong></center>
                    """
                )
            with gr.Column(scale=5):
                chat_history_for_display = gr.Chatbot(
                    value=[(None, GRAND3D_Settings.INITIAL_MSG_FOR_DISPLAY)],
                    label="Chat Assistant",
                    height=510,
                    render_markdown=False,
                    sanitize_html=False,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        user_chat_input = gr.Textbox(
                            placeholder="I want to find the chair near the table",
                            show_label=False,
                        )
                    with gr.Column(scale=1, min_width=0):
                        send_button = gr.Button("Send", variant="primary")
                    with gr.Column(scale=1, min_width=0):
                        clear_button = gr.Button("Clear")
                with gr.Row():
                    with gr.Accordion(label="Examples for user message:", open=True):
                        gr.Examples(
                            examples=[
                                ["The TV on the drawer, opposing the bed."],
                                ["the desk next to the window"],
                            ],
                            inputs=user_chat_input,
                        )

        with gr.Accordion("Parameters", open=False, visible=False):
            top_p = gr.Slider(
                minimum=0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                interactive=True,
                label="Top-p (nucleus sampling)",
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
    # gr.Markdown("### Terms of Service")
    # gr.HTML(
    #     """By using this service, users are required to agree to the following terms:
    #         The service is a research preview intended for non-commercial use only.
    #         The service may collect user dialogue data for future research."""
    # )

    # Event handling
    dropdown_scene.change(
        fn=change_scene_or_system_prompt,
        inputs=[dropdown_scene],
        outputs=[
            session_state,
            model_3d,
            model_3d_grounding_result,
            chat_history_for_display,
        ],
    )
    clear_button.click(
        fn=change_scene_or_system_prompt,
        inputs=[dropdown_scene],
        outputs=[
            session_state,
            model_3d,
            model_3d_grounding_result,
            chat_history_for_display,
        ],
    )
    user_chat_input.submit(
        fn=language_model_forward,
        inputs=[session_state, user_chat_input, top_p, temperature, dropdown_scene],
        outputs=[session_state, model_3d_grounding_result, chat_history_for_display],
    )
    send_button.click(
        fn=language_model_forward,
        inputs=[session_state, user_chat_input, top_p, temperature, dropdown_scene],
        outputs=[session_state, model_3d_grounding_result, chat_history_for_display],
    )

    send_button.click(reset_textbox, [], [user_chat_input])
    user_chat_input.submit(reset_textbox, [], [user_chat_input])

sleep_time = 2
port = 7011
for x in range(1, 10):  # try 8 times
    try:
        # put your logic here
        gr.close_all()
        demo.queue(
            max_size=20,
        ).launch(
            # debug=True,
            # server_name="0.0.0.0",
            # server_port=port,
            share=True
        )
    except OSError:
        for proc in process_iter():
            for conns in proc.connections(kind="inet"):
                if conns.laddr.port == port:
                    proc.send_signal(SIGTERM)  # or SIGKILL
        print(f"Retrying {x} time...")
        pass

    sleep(sleep_time)  # wait for 2 seconds before trying to fetch the data again
    sleep_time *= 2  # exponential backoff
