
import numpy as np
import casadi as cs

from config import CONSTRAINTS_X, CONSTRAINTS_U
from mpc.mpc import MPC
from obstacles.obstacles_motion import ObstacleMotion
from envs.env import env
from rl.optim import AdamOptimizer
from rl.lrscheduler import LearningRateScheduler

from config import NUM_INPUTS, NUM_STATES


class RLagent:

        def __init__(
        self,
        params_init,
        seed,
        alpha,
        gamma,
        decay_rate,
        layers_list,
        noise_scalingfactor,
        noise_variance,
        patience_threshold,
        lr_decay_factor,
        horizon,
        positions,
        radii,
        modes,
        mode_params,
        slack_penalty_MPC_L1, 
        slack_penalty_MPC_L2,
        slack_penalty_RL_L1,
        slack_penalty_RL_L2,
        violation_penalty,
        ):
            # Store random seed for reproducibility
            self.seed = seed

            # Create the environment
            self.env = env()

            # Penalty in RL stagecost on slacks
            self.slack_penalty_MPC_L1 = slack_penalty_MPC_L1
            self.slack_penalty_MPC_L2 = slack_penalty_MPC_L2
            self.slack_penalty_RL_L1 = slack_penalty_RL_L1
            self.slack_penalty_RL_L2 = slack_penalty_RL_L2

            self.violation_penalty = violation_penalty

            # Initialize MPC and obstacle‐motion classes
            self.mpc = MPC(
                layers_list,
                horizon,
                positions,
                radii,
                self.slack_penalty_MPC_L1,
                self.slack_penalty_MPC_L2,
                mode_params,
                modes,
            )
            self.obst_motion = ObstacleMotion(positions, modes, mode_params)

            # layer list of input
            self.nn_input_size = layers_list[0]
            self.layers_list = layers_list

            # Parameters of experiments and states
            self.ns = self.mpc.ns
            self.na = self.mpc.na
            self.horizon = self.mpc.horizon
            self.params_init = params_init

            # Learning‐rate for parameter updates
            self.alpha = alpha

            # Build state bounds repeated over the horizon
            # np.tile takes an array and “tiles” (i.e. repeats) it to fill a larger array.
            self.X_lower_bound = -np.tile(CONSTRAINTS_X, self.horizon)
            self.X_upper_bound = np.tile(CONSTRAINTS_X, self.horizon)

            # Equality‐constraint bounds (Ax+Bu==0) - all zeros
            self.state_const_lbg = np.zeros(1 * self.ns * (self.horizon))
            self.state_const_ubg = np.zeros(1 * self.ns * (self.horizon))

            # CBF safety constraints: h(x_{k+1})-h(x_k)+alpha*h(x_k) + s >= 0 → we invert so g =< 0
            # means the cbf constraint is bounded between -inf and zero --> for the g =< 0
            self.cbf_const_lbg = -np.inf * np.ones(self.mpc.nn.obst.obstacle_num * (self.horizon))
            self.cbf_const_ubg = np.zeros(self.mpc.nn.obst.obstacle_num * (self.horizon))

            # Discount factor for TD updates
            self.gamma = gamma

            # RNG for adding exploration noise
            self.np_random = np.random.default_rng(seed=self.seed)
            self.noise_scalingfactor = noise_scalingfactor
            self.noise_variance = noise_variance

            # Create CasADi mpc solver instances once for reuse
            self.solver_inst = self.mpc.MPC_solver()  # deterministic MPC solver
            self.solver_inst_random = self.mpc.MPC_solver_rand()  # noisy MPC (the MPC with exploration noise)

            # Symbolic function to get the gradient of the MPC Lagrangian
            self.qlagrange_fn_jacob = self.mpc.generate_symbolic_mpcq_lagrange()
            print(f"generated the function")

            # Create CasADi qp solver instance once for reuse
            self.qp_solver = self.mpc.qp_solver_fn()  # QP for constrained parameter updates
            print(f"casadi func created")

            # Learning‐rate scheduling
            self.decay_rate = decay_rate
            self.patience_threshold = patience_threshold
            self.lr_decay_factor = lr_decay_factor
            self.best_stage_cost = np.inf
            self.best_params = params_init.copy()
            self.current_patience = 0

            # ADAM
            theta_vector_num = cs.vertcat(self.params_init["nn_params"])
            self.adam = AdamOptimizer(dim=int(theta_vector_num.shape[0]))

            print(f"before make nn step")
            # hidden state 1 function
            self.get_hidden_func = self.mpc.nn.make_nn_step()

            print(f"before flat input")
            self.flat_input_fn = self.mpc.make_flat_input_fn()

            # Warmstart variables storage
            self._reset_warmstart_caches()

            print(f"before make phi fn")
            # construct function to access h(x_{k+1},k+1)-alpha*h(x_{k},k)
            self.phi_func = self.mpc.make_phi_fn()

            print(f"before make h functions")
            # construct list to extract h(x_{k},k)
            self.h_func_list = self.mpc.nn.obst.make_h_functions()
            print(f"initialization done")
            
            #learning rate scheduler data class (so its just a conainer for data)
            self.lr_scheduler = LearningRateScheduler(
            alpha=alpha,
            patience_threshold=patience_threshold,
            lr_decay_factor=lr_decay_factor,
            best_params=params_init.copy(),
        )
        
        def _reset_warmstart_caches(self):
            """
            Reset the warmstart caches for the MPC solver.
            """
            self.x_prev_VMPC        = cs.DM()  
            self.lam_x_prev_VMPC    = cs.DM()  
            self.lam_g_prev_VMPC    = cs.DM()  

            self.x_prev_VMPCrandom     = cs.DM()  
            self.lam_x_prev_VMPCrandom = cs.DM()  
            self.lam_g_prev_VMPCrandom = cs.DM()
            
            self.x_prev_QMPC        = cs.DM()  
            self.lam_x_prev_QMPC    = cs.DM()  
            self.lam_g_prev_QMPC    = cs.DM()  
        
        @staticmethod
        def noise_scale_by_distance(x, y, max_radius=1.0): #maxradius was 2
            
            
            """
            Compute a scaling factor for exploration noise based on distance from the origin. 
            Close to the origin, noise is scaled down; at max_radius, it is 1.
            
            Returns:
                float: a factor in [0, 1] by which to multiply noise.
            """
            dist = np.sqrt(x**2 + y**2)
            if dist >= max_radius:
                return 1
            else:
                return (dist / max_radius)**2
            
        def v_mpc(self, params, x, xpred_list, ypred_list):
            """
            Solve the value-function MPC problem for the current state.

            Args:
                params (dict):  
                    Dictionary of system and NN parameters
                x (ns,):  
                    Current state of the system.
                xpred_list (m*(horizon+1),):  
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):  
                    Predicted obstacle y-positions over the horizon.
                hidden_in:  
                    Current hidden-state vectors for the NN layers.

            Returns:
                u_opt (na,):  
                    The first optimal control action.
                V_val (solution["f"]):  
                    The optimal value function V(x).
                hidden_t1 :  
                    Updated hidden states after one NN forward pass.
            """
            
            # bounds

            # input bounded between 1 and -1
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            # state constraints (first state is bounded to be x0), omega cannot be 0
            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  
                                  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(), self.X_upper_bound, U_upper_bound, 
                                  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])

            #lower and upper bound for state and cbf constraints 
            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten to put it into the solver 
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"],
                                                       P_diag, Q_flat, R_flat,  params["nn_params"], 
                                                       xpred_list, ypred_list),
                x0    = self.x_prev_VMPC,
                lam_x0 = self.lam_x_prev_VMPC,
                lam_g0 = self.lam_g_prev_VMPC,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )

            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # warmstart variables for next iteration
            self.x_prev_VMPC     = solution["x"]
            self.lam_x_prev_VMPC = solution["lam_x"]
            self.lam_g_prev_VMPC = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the evaluation stage cost)
            self.S_VMPC = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]
            
            #check output  of NN
            alpha_list = []
            h_func_list = [h_func for h_func in self.mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
            alpha_list.append(cs.DM(self.fwd_func(x, h_func_list, xpred_list[:self.mpc.nn.obst.obstacle_num], 
                                                  ypred_list[:self.mpc.nn.obst.obstacle_num], params["nn_params"])))

            return u_opt, solution["f"], alpha_list
        
        def v_mpc_rand(self, params, x, rand, xpred_list, ypred_list):
            """
            Solve the value-function MPC problem with injected randomness.

            This is identical to V_MPC, but includes a random noise term in the optimization
            to encourage exploration.

            Args:
                params (dict):
                    Dictionary of system and NN parameters:
                x (ns,):
                    Current system state vector.
                rand (na,1):
                    Random noise vector added to first control action in MPC objective
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current NN hidden-state from previous time step.

            Returns:
                u_opt (na,):
                    The first optimal control action (with randomness).
                hidden_t1 (list of MX):
                    Updated NN hidden-state 
            """
            
            
            # bounds
            U_lower_bound = -np.ones(self.na * (self.horizon))
            U_upper_bound = np.ones(self.na * (self.horizon))

            lbx = np.concatenate([np.array(x).flatten(), self.X_lower_bound, U_lower_bound,  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.array(x).flatten(),self.X_upper_bound, U_upper_bound,  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])
            

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)

            solution = self.solver_inst_random(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag, Q_flat, R_flat, params["nn_params"], rand, xpred_list, ypred_list),
                x0    = self.x_prev_VMPCrandom,
                lam_x0 = self.lam_x_prev_VMPCrandom,
                lam_g0 = self.lam_g_prev_VMPCrandom,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            #extract first optimal control action to apply (MPC)
            u_opt = solution["x"][self.ns * (self.horizon+1):self.ns * (self.horizon+1) + self.na]
            
            # warmstart variables for next iteration
            self.x_prev_VMPCrandom = solution["x"]
            self.lam_x_prev_VMPCrandom = solution["lam_x"]
            self.lam_g_prev_VMPCrandom = solution["lam_g"]
            
            # remember the slack variables for stage cost computation (in the RL stage cost)
            self.S_VMPC_rand = solution["x"][self.na * (self.horizon) + self.ns * (self.horizon+1):]
            
            #check output  of NN
            alpha_list = []
            h_func_list = [h_func for h_func in self.mpc.nn.obst.h_obsfunc(x, xpred_list, ypred_list)]
            alpha_list.append(cs.DM(self.fwd_func(x, h_func_list, xpred_list[:self.mpc.nn.obst.obstacle_num], 
                                                  ypred_list[:self.mpc.nn.obst.obstacle_num],  params["nn_params"])))

            return u_opt, alpha_list

        def q_mpc(self, params, action, x, xpred_list, ypred_list):
            
            """"
            
            Solve the Q-value MPC problem for current state and current action.
            
            Similar to V_MPC, but includes the action in the optimization and computes the Q-value.
            
            Args:
                params (dict):
                    Dictionary of system and NN parameters.
                action (na,):
                    Current control action vector.
                x (ns,):
                    Current state of the system.
                xpred_list (m*(horizon+1),):
                    Predicted obstacle x-positions over the horizon.
                ypred_list (m*(horizon+1),):
                    Predicted obstacle y-positions over the horizon.
                hidden_in (list of MX):
                    Current hidden-state vectors for the NN layers.
            Returns:
                x_opt (ns*(horizon+1),):
                    Optimal state trajectory over the horizon.
                Q_val (solution["f"]):
                    Optimal Q-value for the current state and action.
                lagrange_mult_g (solution["lam_g"]):
                    Lagrange multipliers for the constraints.
                lam_lbx (solution["lam_x"]):
                    Lagrange multipliers for the lower bounds on x.
                lam_ubx (solution["lam_x"]):
                    Lagrange multipliers for the upper bounds on x.
                lam_p (solution["lam_p"]):
                    Lagrange multipliers for the parameters.
                hidden_t1 (list of MX):
                    Updated hidden states after one NN forward pass.
            """

            # Build input‐action bounds (note horizon−1 controls remain free after plugging in `action`)
            U_lower_bound = -np.ones(self.na * (self.horizon-1))
            U_upper_bound = np.ones(self.na * (self.horizon-1))

            #Assemble full lbx/ubx: [ x0; X(1…H); action; remaining U; slack ]
            lbx = np.concatenate([np.asarray(x).flatten(), self.X_lower_bound, 
                                  np.asarray(action).flatten(), U_lower_bound,  
                                  np.zeros(self.mpc.nn.obst.obstacle_num *self.horizon)])  
            ubx = np.concatenate([np.asarray(x).flatten(), self.X_upper_bound,
                                  np.asarray(action).flatten(), U_upper_bound, 
                                  np.inf*np.ones(self.mpc.nn.obst.obstacle_num *self.horizon)])

            lbg = np.concatenate([self.state_const_lbg, self.cbf_const_lbg])  
            ubg = np.concatenate([self.state_const_ubg, self.cbf_const_ubg])

            #flatten
            A_flat = cs.reshape(params["A"] , -1, 1)
            B_flat = cs.reshape(params["B"], -1, 1)
            P_diag = cs.diag(params["P"])#cs.reshape(params["P"], -1, 1)
            Q_flat = cs.reshape(params["Q"], -1, 1)
            R_flat = cs.reshape(params["R"], -1, 1)
            
            solution = self.solver_inst(p = cs.vertcat(A_flat, B_flat, params["b"], params["V0"], P_diag,
                                                       Q_flat, R_flat, params["nn_params"],
                                                       xpred_list, ypred_list),
                x0    = self.x_prev_QMPC,
                lam_x0 = self.lam_x_prev_QMPC,
                lam_g0 = self.lam_g_prev_QMPC,
                ubx=ubx,  
                lbx=lbx,
                ubg=ubg,
                lbg=lbg
            )
            
            # Extract lagrange multipliers needed for the lagrangian:
            lagrange_mult_g = solution["lam_g"]
            lam_lbx = -cs.fmin(solution["lam_x"], 0)
            lam_ubx = cs.fmax(solution["lam_x"], 0)
            lam_p = solution["lam_p"]
            
            # warmstart variables for next iteration
            self.lam_g_prev_QMPC = solution["lam_g"]
            self.x_prev_QMPC = solution["x"]
            self.lam_x_prev_QMPC = solution["lam_x"]

            return solution["x"], solution["f"], lagrange_mult_g, lam_lbx, lam_ubx, lam_p
        
            
        def stage_cost(self, action, state, S, hx):
            """
            Computes the stage cost : L(s,a).
            
            Args:
                action: (na,):
                    Control action vector.
                state: (ns,):
                    Current state vector of the system
                S: (m*(horizon+1),):
                    Slack variables for the MPC problem, used in the stage cost.
                    Slacks that were used for relaxing CBF constraints in the MPC problem.
            
            Returns:
                float:
                    The computed stage cost value.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])
            hx = np.array(hx)
            
            violations = np.clip(-hx, 0, None)
            
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action +self.slack_penalty_RL*(np.sum(S)/(self.horizon+self.mpc.nn.obst.obstacle_num)) #+ np.sum(4e5*violations)
            )
            
        def stage_cost_validation(self, action, state, hx):
            """
            Computes the stage cost : L(s,a).
            
            Args:
                action: (na,):
                    Control action vector.
                state: (ns,):
                    Current state vector of the system
                S: (m*(horizon+1),):
                    Slack variables for the MPC problem, used in the stage cost.
                    Slacks that were used for relaxing CBF constraints in the MPC problem.
            
            Returns:
                float:
                    The computed stage cost value.
            """
            # same as the MPC ones
            Qstage = np.diag([10, 10, 10, 10])
            Rstage = np.diag([1, 1])
            hx = np.array(hx)
            
            violations = np.clip(-hx, 0, None)
            
            return (
                state.T @ Qstage @ state
                + action.T @ Rstage @ action + np.sum(3e5*violations)
            )
        
        def parameter_updates(self, params, B_update_avg):

            """
            function responsible for carryin out parameter updates after each episode
            """
            P_diag = cs.diag(params["P"])

            #vector of parameters which are differenitated with respect to
            # theta_vector_num = cs.vertcat(P_diag, params["nn_params"])
            
            theta_vector_num = cs.vertcat(params["nn_params"])

            # L  = self.cholesky_added_multiple_identity(A_update_avg)
            # A_update_chom = L @ L.T
            
            identity = np.eye(theta_vector_num.shape[0])


            # alpha_vec is resposible for the updates
            # alpha_vec = cs.vertcat(self.alpha*np.ones(3), self.alpha, self.alpha, self.alpha*np.ones(theta_vector_num.shape[0]-5)*1e-2)
            alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0])*1e-2)
            # alpha_vec = cs.vertcat(self.alpha*np.ones(4), self.alpha*np.ones(theta_vector_num.shape[0]-4)*1e-2)
            # alpha_vec = cs.vertcat(self.alpha*np.ones(theta_vector_num.shape[0]-2), self.alpha,self.alpha*1e-5)
            print(f"B_update_avg:{B_update_avg}")

            dtheta, self.exp_avg, self.exp_avg_sq = self.ADAM(self.adam_iter, B_update_avg, self.exp_avg, self.exp_avg_sq, alpha_vec, 0.9, 0.999)
            self.adam_iter += 1 

            print(f"dtheta: {dtheta}")

            # uncostrained update to compare to the qp update
            y = np.linalg.solve(identity, B_update_avg)
            theta_vector_num_toprint = theta_vector_num - (y)#self.alpha * y
            print(f"theta_vector_num no qp: {theta_vector_num_toprint}")

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.01*np.abs(theta_vector_num[5:]))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.01*np.abs(theta_vector_num[5:]))

            # lbx = cs.vertcat(-np.inf*np.ones(5), -0.0001*np.ones(theta_vector_num.shape[0]-5))
            # ubx = cs.vertcat(np.inf*np.ones(5), 0.0001*np.ones(theta_vector_num.shape[0]-5))
            
            # constrained update qp update
            solution = self.qp_solver(
                    p=cs.vertcat(theta_vector_num, dtheta),
                    # lbg = cs.vertcat(np.zeros(4), -np.inf*np.ones(theta_vector_num.shape[0]-4)),
                    # ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                    lbg= cs.vertcat(-np.inf*np.ones(theta_vector_num.shape[0])),
                    ubg = cs.vertcat(np.inf*np.ones(theta_vector_num.shape[0])),
                    # ubx = ubx,
                    # lbx = lbx
                )
            stats = self.qp_solver.stats()


            if stats["success"] == False:
                print("QP NOT SUCCEEDED")
                theta_vector_num = theta_vector_num
            else:
                theta_vector_num = theta_vector_num + solution["x"]

            print(f"theta_vector_num: {theta_vector_num}")

            # P_diag_shape = self.ns*1
            # # #constructing the diagonal posdef P matrix 
            # P_posdef = cs.diag(theta_vector_num[:P_diag_shape])

            # params["P"] = P_posdef
            # params["nn_params"] = theta_vector_num[P_diag_shape:]       
            params["nn_params"] = theta_vector_num  

            return params
        
