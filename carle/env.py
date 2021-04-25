import numpy as np
import matplotlib.pyplot as plt
import os
import time 

import skimage
import skimage.io

import torch
import torch.nn as nn
import torch.nn.functional as F

class CARLE(nn.Module):

    def __init__(self, **kwargs):
        super(CARLE, self).__init__()

        self.inner_env = None
        self.width = kwargs["width"] if "width" in kwargs.keys() else 256
        self.height = kwargs["height"] if "height" in kwargs.keys() else 256
        self.use_cuda = kwargs["use_cuda"] \
                if "use_cuda" in kwargs.keys() else False

        self.action_width = 64 
        self.action_height = 64  

        self.alive_rate = kwargs["alive_rate"] if "alive_rate" in kwargs.keys()\
                else 0.0
        
        # instances define how many CA universes to run in parallel via vectorization 
        self.instances = kwargs["instances"] if "instances" in kwargs.keys()\
                else 1

        # keep track of universe development
        self.logging = kwargs["logging"] if "logging" in kwargs.keys() else False

        self.set_neighborhood()
        self.set_action_padding()

        # Conway's GoL rules
        self.allowed_rules = [str(num) for num in range(9)]
        self.birth = [3]
        self.survive = [2,3]


    def birth_rule_from_string(self, my_string="B3"):
        self.birth = []
        for element in my_string:
            if element in self.allowed_rules:
                self.birth.append(int(element))

    def survive_rule_from_string(self, my_string="S23"):
        self.survive = []
        for element in my_string:
            if element in self.allowed_rules:
                self.survive.append(int(element))

    def rules_from_string(self, my_string="B3/S23"):

        temp = my_string.split("/")

        self.birth_rule_from_string(temp[0])
        self.survive_rule_from_string(temp[1])

    def set_neighborhood(self):
        """
        Establish the neighborhood function as a convolutional layer
        Moore neighborhoods are used in Life-like CA
        """
        
        circular = True

        moore_kernel = torch.tensor([[1.,1.,1.], [1.,0.,1.], [1.,1.,1.]],\
                requires_grad=False)

        if circular:
            my_mode = "circular"
        else:
            my_mode = "zeros"

        self.neighborhood = nn.Conv2d(1, 1, 3, padding=1,\
                padding_mode=my_mode, bias=False)


        if torch.cuda.is_available() and self.use_cuda:
            self.my_device = "cuda"
            self.neighborhood.to(self.my_device)
            self.to(self.my_device)

            # run on multiple gpus if possible
            #self.neighborhood = nn.DataParallel(self.neighborhood)
        else:
            self.my_device = "cpu"
            self.neighborhood.to(self.my_device)

        for param in self.neighborhood.parameters():
            param.requres_grad = False

        for param in self.neighborhood.named_parameters():
            param[1][0] = moore_kernel

        for param in self.neighborhood.parameters():
            param.requres_grad = False


    def set_action_padding(self):


        assymetry_width = (self.width - self.action_width) % 2
        assymetry_height = (self.height - self.action_height) % 2

        self.action_width -= (self.width % 2)
        self.action_height -= (self.height % 2)

        width_padding = (self.width - self.action_width) // 2 
        height_padding = (self.height - self.action_height) // 2

        self.action_padding = nn.ZeroPad2d(padding=\
                (height_padding, height_padding + assymetry_height,\
                width_padding, width_padding + assymetry_width))

    def reset(self):
        
        self.universe = torch.zeros(self.instances, 1, self.height, self.width)

        self.universe = self.universe.to(self.my_device)
        observation = self.universe

        self.instance_id = str(int(time.time()))
        self.step_number = 0

        # used to determine when logging universe rle is necessary
        self.steps_since_action = 0
        self.log = []

        return observation

    def apply_action(self, action):

        if type(action) is not torch.Tensor:
            action = torch.Tensor(action)

        while len(action.shape) < 4:
            action = action.unsqueeze(0)

        action = action.to(self.my_device)

        # this may be better as an assertion line to avoid silent failures
        action = action[0, 0, :self.action_width, :self.action_height]

        action = self.action_padding(action)

        # toggle cells according to actions
        self.universe = 1.0 * torch.logical_xor(self.universe, action)

    def get_observation(self):

        return self.universe

    def step(self, action):
        
        if torch.sum(action):
            self.action = action

            if self.logging:
                self.log_universe()

            self.apply_action(action)

        else:
            self.steps_since_action += 1

        if torch.mean(action) == 1.0:
            """
            If all toggles are toggled, perform a universe reset.
            This gives agents a means to 'clear the slate' without relying on
            external interference i.e. hand-coded resets when an agent is 
            'stuck.'
            """

            observation = self.reset()
        else:

            my_neighborhood = self.neighborhood(self.universe)

            universe_1 = torch.zeros_like(self.universe) 

            for b in self.birth:
                universe_1[((1-self.universe) * (my_neighborhood == b)) == 1] = 1

            for s in self.survive:
                universe_1[(self.universe * (my_neighborhood == s)) == 1] = 1

            
            self.universe = universe_1
            self.step_number += 1

            # This environment is open-ended free from exogenous reward,
            # giving no done signal and a reward of 0.0
            # episodic constraints and endogenous rewards have to be implemented
            # by wrappers or agents themselves.
            observation = self.get_observation()

        reward = torch.zeros(self.instances, 1).to(self.my_device)
        done = torch.zeros(self.instances, 1)
        info = [{}] * self.instances

        return observation, reward, done, info

    def render(self):

        os.system("clear")
        print("\n CA Universe")

        for ii in range(self.universe.shape[2]):
            print("")
            for jj in range(self.universe.shape[3]):
                if self.universe[0,0,ii,jj]:
                    print("o", end="")
                else:
                    print(" ", end="")

        time.sleep(0.125)
                #print(self.universe[0,0,ii,jj], end="\r")

    def rle_to_grid(self, rle):

        ii = 0
        jj = 0

        line_count = 0
        total_count = 0
        rle_length = len(rle)

        my_grid = torch.zeros(self.height, self.width) 

        temp = ""
        while total_count < rle_length:

            temp += rle[total_count]

            if temp[-1] == "\n":
                temp = temp[:-1]
            else:

                if temp[-1].lower() == "b":

                    if temp == "b":
                        run = 1
                    else:
                        run = int(temp[:-1])

                    my_grid[ii,jj:jj+run] = 0

                    jj += run
                    temp = ""

                elif temp[-1].lower() == "o":

                    if temp == "o":
                        run = 1
                    else:
                        run = int(temp[:-1])

                    my_grid[ii,jj:jj+run] = 1

                    jj += run
                    temp = ""

                elif temp[-1] == "$":

                    if len(temp) > 1:
                        row_run = int(temp[:-1])

                        # +1 so that current row doesn't get overwritten 
                        my_grid[ii+1:ii+row_run,:] = 0

                        ii += row_run
                    else:
                        my_grid[ii,jj:] = 0
                        
                        # advance to next row
                        ii += 1


                    jj = 0
                    temp = ""
                    
                elif temp[-1] == "!":
                    temp = ""

            total_count += 1

        return my_grid

    def read_rle(self, filepath):

        rle = ""

        add_to_rle = False
        with open(filepath, "r") as f:

            for temp_line in f.readlines():
                    
                if add_to_rle:

                    rle += temp_line 

                if "rule" in temp_line:

                    rules = temp_line.split("/")

                    proto_birth = rules[0].split()[-1]

                    if ":" in rules:
                        # assuming the rle includes dimensions 
                        # (i.e. has a colon) for now
                        proto_survive_dim = rules[1].split(":")

                        proto_survive = proto_survive_dim[0]
                        proto_dim = proto_survive_dim[1]
                    else: 
                        proto_survive = rules[-1]
                        proto_dim = None
                        

                    self.birth = []
                    self.survive = []

                    for bb in proto_birth:
                        if bb.lower() != "b":
                            self.birth.append(int(bb))
                    for ss in proto_survive:
                        if ss.lower() != "s" and ss.lower() != '\n':
                            self.survive.append(int(ss))

                    # ignore dimensions (and corner) for now (assuming rle files come from CARLE)

                    # set 
                    add_to_rle = True

        return rle

    def read_csv(self, filepath):

        print("warning, read_csv not implemented yet")

        return ""

    def load_universe(self, filepath, universe_index=0):


        if "rle" in filepath[-4:]:

            my_rle = self.read_rle(filepath)

        else:
            my_rle = self.read_csv(filepath)

        my_universe = self.rle_to_grid(my_rle)

        assert self.universe.shape[2] == my_universe.shape[0]\
                and self.universe.shape[3] == my_universe.shape[1],\
                "tried to load the wrong size universe"

        self.universe[universe_index,0,:,:] = my_universe

    def get_rle(self, universe, action=False):

        """
        compute run-length encoding for given universe
        expects one 2D CA universe
        """

        universe = universe.squeeze()

        #write header
        rle = "#C exp_id={} \n".format(self.instance_id)
        if action:
            rle += "#C step={} (action) \n".format(self.step_number)
        else:
            rle += "#C step={} (universe) \n".format(self.step_number)

        rle += "x = 0, y = 0, rule = B" 
        for bb in self.birth: rle += str(bb)
        rle += "/S"  
        for ss in self.survive: rle += str(ss)
        rle += ":T{}, {}\n".format(self.height, self.width)

        state_string = ["b", "o"]

        run_length = ""
        for ii in range(universe.shape[0]):
            jj = 0
            current_state = universe[ii, jj]
            run_count = 1
            while jj < universe.shape[1] - 1:

                jj +=1

                if universe[ii, jj] == current_state:
                    run_count += 1
                else:
                    
                    run_length += str(run_count) + state_string[int(current_state)]

                    if len(run_length) > 69:
                        rle +=  run_length + "\n"
                        run_length = ""

                    current_state = universe[ii,jj]
                    run_count = 1

            
            run_length += str(run_count) + state_string[int(current_state)]
            run_length += "$"
            if len(run_length) > 69:
                rle +=  run_length + "\n"
                run_length = ""

        # end of pattern signal
        rle += "!"

        return rle

    def log_universe(self, universe_index=0):
        """
        save universe rle to log list
        [[action, rle]
        ]
        """

        rle_universe = self.get_rle(self.universe[universe_index,0,:,:])
        rle_action = self.get_rle(self.action[universe_index,0,:,:], action=True)

        self.log.append([rle_action, rle_universe])

        
    def save_log(self):
        """
        save log as csv file
        """
        
        with open("./logs/carle_log{}.csv".format(self.instance_id), "w") as f:

            f.write('action,universe,\n')
            for log_entry in self.log:

                for entry in log_entry:
                    f.write('"' + entry + '"' + ",")
                f.write("\n")



    def save_rle(self, rle):

        with open("./logs/universe{}_step{}.rle"\
                .format(self.instance_id, self.step_number), "w") as f:

            f.write(rle)



    def save_frame(self):
        """
        save frames from the first instance of 'universe' 
        (at index [0,0,:,:] from self.universe tensor)
        """

            
        skimage.io.imsave("./frames/frame{}_step{}.png"\
                .format(self.instance_id, self.step_number), \
                np.uint8(255 * self.universe[0,0,:,:].detach().cpu().numpy()))



if __name__ == '__main__':

    env = CARLE(logging=True)

    obs = env.reset()
    

    my_steps = 3

    action = torch.zeros(env.instances, \
            1, env.action_height, \
            env.action_width)
    action[:,:,14, 16] = 1.0
    action[:,:,15, 16:18] = 1.0
    action[:,:,16, 15:18:2] = 1.0

    t0 = time.time()
    for step in range (my_steps):
        #env.render()
        _ = env.step(action)
        action *= 0.0


    rle = env.get_rle(env.universe[0,0,:,:])

    env.save_rle(rle)
    env.save_frame()

    env.save_log()


    env = CARLE()
    obs = env.reset()
    env.load_universe("./logs/universe1616019257_step3.rle")

    env.save_frame()

    t1 = time.time()
    print("CA updates per second with {}x vectorization = {} and saving frames"\
            .format(env.instances, my_steps * env.instances/(t1-t0)))

    if(0):


        for instances in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            action = torch.ones(env.instances,1,32,32)
            env.instances = instances
            obs = env.reset()
            t2 = time.time()

            for step in range(my_steps):
                _ = env.step(action)
            

            t3 = time.time()
            print("CA updates per second with {}x vectorization = {}"\
                    .format(env.instances, my_steps * env.instances/(t3-t2)))




