import mujoco_py as mjp
import numpy as np
import pickle

# Specify the path to your XML file
xml_file = "/home/zsufiyan/TD3/custom_gym_envs/envs/reacher/xml/ReacherEnv_v1_StructDamage2.xml"
model = mjp.load_model_from_path(xml_file)
sim = mjp.MjSim(model)

# List to store the states
states = []

# Simulate and save states
for i in range(1000):
    sim.step()
    # Save the entire state, or extract only the necessary values
    states.append(sim.get_state())

# Save states to a file using pickle
with open('simulation_states.pkl', 'wb') as f:
    pickle.dump(states, f)

print("Simulation states saved.")
