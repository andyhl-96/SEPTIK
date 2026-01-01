# SEPTIK: Stein-Enhanced Pathwise Inverse Kinematics

SEPTIK is a method for pathwise inverse kinematics (IK) that leverages JAX JIT compilation and vectorization, as well as stein-variational projection onto IK manifolds. The procedure will use a multipartite graph structure similar to STAMPEDE[1], but mainly differes in the sampling strategy for IK solutions and enforcing C^2 continuity between timesteps. We will hopefully be able to outperform existing methods in both runtime and accuracy.
