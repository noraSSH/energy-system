T = 24

p = cp.Variable((generation.shape[0],T))
r = cp.Variable((generation.shape[0],T))
theta = cp.Variable((N,T))
u = cp.Variable((generation.shape[0],T),boolean = True) # Commitment variable
v = cp.Variable((generation.shape[0],T),boolean = True) # Startup variable
w = cp.Variable((generation.shape[0],T),boolean = True) # Shutdown variable

obj = cp.Minimize(cp.sum(linear_cost.T@p + startup_cost.T@v + shutdown_cost.T@w + noload_cost.T@u + reserve_cost.T@u))
# obj = cp.Minimize(cp.sum(linear_cost.T@p))

power_flow_constraints = [Egen@p - d == B@theta]
flow_limits = [fmin@np.ones((1,T)) <= np.diag(b)@M.T@theta, np.diag(b)@M.T@theta <= fmax@np.ones((1,T))]
generator_limits = [np.diag(pmin.T[0])@u + r <= p, p <= np.diag(pmax.T[0])@u - r]
ramp_limits_startup = [p[:,i] - p[:,i-1] <= np.diag(hourly_ramp.T[0])@u[:,i-1] + np.diag(startup_shutdown_ramp.T[0])@v[:,i] for i in range(1,T)]
ramp_limits_shutdown = [p[:,i-1] - p[:,i] <= np.diag(hourly_ramp.T[0])@u[:,i] + np.diag(startup_shutdown_ramp.T[0])@w[:,i] for i in range(1,T)]
commitment_constraints = [v[:,i] - w[:,i] == u[:,i] - u[:,i-1] for i in range(1,T)]

min_up_time_constraints = []
for idx, utime in enumerate(min_up_time):
    for t in range(utime-1,T):
        min_up_time_constraints += [cp.sum(v[idx,t-utime+1:t+1]) <= u[idx,t]]

min_down_time_constraints = []
for idx, dtime in enumerate(min_down_time):
    for t in range(dtime-1,T):
        min_down_time_constraints += [cp.sum(w[idx,t-dtime+1:t+1]) <= 1-u[idx,t]]

reserve_constraints = [cp.sum(r[:,t]) >= 0.07*cp.sum(d[:,t]) for t in range(T)]
reserve_constraints += [cp.sum(r[:,t]) >= p[idx,t] + r[idx,t] for t in range(T) for idx,_ in enumerate(pmax)]
reserve_constraints += [r <= np.diag(pmax.T[0])@u, r >= 0]

# constraints = power_flow_constraints + flow_limits + generator_limits + commitment_constraints + min_up_time_constraints + min_down_time_constraints + reserve_constraints + ramp_limits_startup + ramp_limits_shutdown
constraints = power_flow_constraints + flow_limits + generator_limits + commitment_constraints + reserve_constraints + ramp_limits_startup + ramp_limits_shutdown + min_up_time_constraints + min_down_time_constraints

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.GUROBI,verbose=True,MIPgap = 0.01)