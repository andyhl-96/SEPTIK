import septik
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import viser
from viser.extras import ViserUrdf
import pyroki as pk
import time

def circle(t, T):
    x = jnp.array([0.000000, 0.707107, 0.000000, 0.707107, 0.4, 0.2 * jnp.sin(t / T * 2 * jnp.pi), 0.2 * jnp.cos(t / T * 2 * jnp.pi) + 0.80])
    return x

# sample poses on the path
path_nodes = septik.compute_points(circle, 5, 5)

# project points on each IK manifold
franka_urdf, franka, franka_coll = septik.load_robot("fr3_franka_hand.urdf")
X = septik.sample(septik.primes[0:7], jnp.arange(1, 1001))
X = septik.scale_points(X, franka.joints.lower_limits_all[0:7], franka.joints.upper_limits_all[0:7])
print(franka.joints.upper_limits_all[0:7])
print(septik.primes[0:7])

# initialize viser with robot and IK path
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
urdf_vis = ViserUrdf(server, franka_urdf, root_node_name="/robot")
server.scene.add_point_cloud(
        name="/multiple_points",
        points=np.array(path_nodes[:, 4:]),
        colors=np.repeat(np.array([[1, 0.5, 0]]), len(path_nodes), axis=0),
        point_size=0.01,
        point_shape="circle"
)

# project configs onto each ik pose
print("jitting functions")
ik_funcs = []
for i in range(len(path_nodes)):
    ik_func = lambda q, robot: jnp.linalg.norm(jnp.array(robot.forward_kinematics(q)[-1] - path_nodes[i]))
    X_proj = septik.jacobi_stein_proj(ik_func, 1, 1, X, franka)
    ik_funcs.append(ik_func)
print("Done jitting")

layers = []
ts = time.perf_counter()
for i in range(len(path_nodes)):
    X_proj = septik.jacobi_stein_proj(ik_funcs[i], 10, 10, X, franka)
    layers.append(X_proj)
tf = time.perf_counter()
print(tf - ts)

coeffs = septik.compute_hermite_poly4([layers[0][0], layers[1][0], jnp.zeros(7), jnp.zeros(7), jnp.zeros(7)], 0.0, 1)
ts = time.perf_counter()
coeffs = septik.compute_hermite_poly4([layers[0][0], layers[1][0], jnp.zeros(7), jnp.zeros(7), jnp.zeros(7)], 0.0, 1)
tf = time.perf_counter()
print(tf - ts)
exit()

while True:
    for layer in layers:
        urdf_vis.update_cfg(np.array(layer[50]))
        time.sleep(0.1)
    time.sleep(1)




