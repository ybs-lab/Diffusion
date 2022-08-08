from importData import import_all, generate_diffuse_tether_trajectories

if __name__ == '__main__':
    df = import_all(get_latest=False, is_parallel=True, break_trajectory_at_collisions=True, assign_traj_states=True)
