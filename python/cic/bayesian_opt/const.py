SIMULATION = True
GITHUB_BRANCH = 'cleanup'
DIFFICULTY_LEVEL = 4
EVALUATE_TRAJ = 'standart'
PATH_TO_IMAGE = '/home/funk/production_test1.sif'
USERNAME = 'emptyheron'
PWD = ''

RELATIVE_MOVEMENT = True
SAMPLE_FCT = 'sample_lift_w_orientations_directions'#'sample_rot_ground_directions'

NUM_INIT_SAMPLES = 4
NUM_ROLLOUTS_PER_SAMPLE = 4
NUM_LOCAL_THREADS = 15 # only has an effect when running in simulation
NUM_ITERATIONS = 50
NUM_ACQ_RESTARTS = 500
ACQ_SAMPLES = 1000

EPISODE_LEN_SIM = 25000
EPISODE_LEN_REAL = 60000

# EVALUATION SPECIFIC PARAMETERS - ONLY CARE ABOUT THEM WHEN RUNNING eval_code.py
SAMPLE_NEW = True
SPLIT_JOBS_EVENLY = True # tells whether jobs should be split evenly across robots
MODELS_TO_RUN = ['model_1','model_2']
# MODELS_TO_RUN = ['tud_align','ttic_align','uw_align']
# MODELS_TO_RUN = ['tud_with_tud_grasp','tud_with_ttic_grasp','tud_with_uw_grasp','ttic_with_tud_grasp','ttic_with_ttic_grasp','ttic_with_uw_grasp','uw_with_tud_grasp','uw_with_ttic_grasp','uw_with_uw_grasp']
ROBOTS_AVAILABLE = [None]#['roboch5','roboch1'] #[None] # ['roboch5','roboch1'] #