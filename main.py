import numpy as np
from heapq import *
from scipy.special import zeta
from scipy.optimize import minimize_scalar, minimize
import sys
import time
import argparse

class Simulator(object):

	def __init__(self, num_arms, arm_means, arm_stds, delay, partial_stds=None, batch_size=1):
		"""
		Arguments:
			num_arms: total number of arms
			arm_means: vector of arm means to simulate Gaussian reward feedback
			arm_stds: vector of arm stds to simulate Gaussian reward feedback and compute confidence interval
			delay: vector of delays associated with every arm pull
			partial_stds: vector of stds associated with every partial feedback reward
			batch_size: number of arms that can be simulated for parallel MAB
		"""

		self.n = num_arms
		self.means = arm_means
		self.stds = arm_stds
		self.partial_stds = partial_stds
		self.batch_size = batch_size

		if partial_stds is not None:
			self.next_score = np.zeros(batch_size)
		self.delay = delay

	def sim(self, arm, t, slot=0):
		# Simulate a full delayed feedback
		score = np.random.normal(self.means[arm], self.stds[arm])
		if self.partial_stds is not None:
			self.next_score[slot] = score
		return (self.delay[arm] + t, arm, score)

	def pu_sim(self, arm, slot=0):
		# Simulate a partial delayed feedback
		return np.random.normal(self.next_score[slot], self.partial_stds[arm])


class RacingDF(object):

	def __init__(self, k, delta, simulator, a=0.6, c=1.1, partial_feedback='none'):
		"""
		Arguments:
			k: number of best arms we want to select
			delta: 1-target confidence
			simulator: the MAB simulator
			a, c: parameters needed for calculating the confidence intervals 
				  (see Zhao et al. "Adaptive Concentration Inequalities for Sequential Decision Problems." NIPS, 2016 for details.)
			partial_feedback: 'none' - using only full delayed feedback
				  'unbiased' - using unbiased partial feedback
		"""

		assert(partial_feedback.lower() == 'none' or partial_feedback.lower() == 'unbiased'), 'Invalid mode. Please set \'none\', or \'unbiased\' as mode.'
		self.n = simulator.n
		self.sigma = simulator.stds 
		self.k = k
		self.delta = delta
		self.simulator = simulator
		self.mode = partial_feedback.lower()

		self.a = a
		self.c = c

		self.t = 0
		self.bsize = simulator.batch_size
		self.last_feedback_time = np.zeros(self.bsize)
		
		self.available_arms = np.ones(self.bsize)
		self.surviving = {j:0 for j in range(self.n)}
		self.accepted = {}
		self.rejected = {}

		self.N = np.zeros(self.n)
		self.F = np.zeros(self.n)
		self.mu = np.zeros(self.n)

		if self.mode == 'unbiased':
			self.mu_pa = np.zeros(self.bsize)
			self.mu_bias = np.zeros(self.n)

		self.ub = np.ones(self.n) * np.inf
		self.lb = np.ones(self.n) * -np.inf
		self.feedbacks = [[] for i in range(self.bsize)]


	
	def C(self, sigma, t, delta):
		"""
		Calculates the radius of the confidence interval
		Arguments:
			sigma: standard deviation
			t: number of feedbacks received
			delta: error likelihood (for the whole process, not just this individual interval)
		"""

		if t == 0:
			return np.inf
		a = self.a
		c = self.c
		b = 0.5 * c * (np.log(zeta(2 * a / c, 1)) + np.log(2 / delta))
		A = max(len(self.surviving), 1)
		return 2 * sigma * ((a * np.log(1 + np.log(t) / np.log(c)) + b + c * np.log(A) / 2) / t) ** 0.5

	def UB(self, arm):
		return self.mu[arm] + self.C(self.sigma[arm], self.F[arm], self.delta)

	def LB(self, arm):
		return self.mu[arm] -  self.C(self.sigma[arm], self.F[arm], self.delta)

	
	def receive_feedback(self, batch_num):
		"""
		Receives feedback from a given batch slot. Updates empirical mean and confidence intervals.
		Rejects or accepts arms based on the confidence intervals
		"""

		_, arm, score = heappop(self.feedbacks[batch_num])
		# Update empirical mean
		self.mu[arm] = (self.mu[arm] * self.F[arm] + score) / (self.F[arm] + 1)
		if self.mode == 'unbiased':
			# Set partial feedback empirical mean to 0 since we now have the full delayed feedback
			self.mu_pa[batch_num] = 0.0
		
		self.F[arm] += 1

		kt = self.k - len(self.accepted)
		# Calculate new confidence intervals
		C = self.C(self.sigma[arm], self.F[arm], self.delta)
		self.ub[arm] = self.mu[arm] + C
		self.lb[arm] = self.mu[arm] - C
		ub = []
		lb = []
		for new_arm, pulls in self.surviving.items():
			ub.append((self.ub[new_arm], new_arm))
			lb.append((self.lb[new_arm], new_arm))
		ub = sorted(ub,reverse=True)
		lb = sorted(lb,reverse=True)

		# Update surviving arms
		for i in range(len(self.surviving)):
			if lb[i][0] > ub[kt][0]:
				self.accepted[lb[i][1]] = self.surviving.pop(lb[i][1])
			if ub[i][0] < lb[kt-1][0]:
				self.rejected[ub[i][1]] = self.surviving.pop(ub[i][1])

		self.available_arms[batch_num] = 1

	def new_pull(self, batch_num):
		"""
		Makes a new pull at a given batch slot.
		"""

		min_pull_count = np.inf
		next_arm = -1
		for arm, pulls in self.surviving.items():
			if pulls < min_pull_count:
				min_pull_count = pulls
				next_arm = arm

		heappush(self.feedbacks[batch_num], self.simulator.sim(next_arm, self.t, batch_num))
		self.surviving[next_arm] += 1
		self.available_arms[batch_num] = 0
		self.last_feedback_time[batch_num] = self.t

	def increment_time(self):
		"""
		Increments time by one. Makes new pulls if any batch slots are open. Receives any feedbacks which are due
		to arrive at the new time. If mode != 'none', receives partial feedbacks from all batch slots that have
		arms running and whose feedbacks aren't arriving at the new time. 
		"""
		if (self.t+1) % 1e5 == 0:
			print("t = %d; number of surviving arms: %d" % (self.t, len(self.surviving)))

		for b in range(self.bsize):
			if self.available_arms[b] == 1:
				self.new_pull(b)

		self.t += 1

		for b in range(self.bsize):
			if self.t == self.feedbacks[b][0][0]:
				self.receive_feedback(b)
			else:
				arm = self.feedbacks[b][0][1]

				if self.mode == 'unbiased':
					score = self.simulator.pu_sim(arm, b)
					d = self.t - self.last_feedback_time[b]
					# Update partial feedback empirical mean
					self.mu_pa[b] = (self.mu_pa[b] * (d-1) + score) / d

					# Calculate the alternative confidence intervals when using partial feedback

					if self.mode == 'unbiased':	
						def Cp_func(delta):
							return self.C(self.sigma[arm], self.F[arm]+1, delta[0]) + self.C(self.simulator.partial_stds[arm], d, delta[1]) / (self.F[arm] + 1)
						res = minimize(Cp_func, [self.delta/2, self.delta/2], bounds=((1e-6,self.delta) , (1e-6,self.delta)), 
									   constraints = ({'type':'ineq', 'fun': lambda x: self.delta - x[0] - x[1] }))
						delta_p = res.x[1]

						C_combined = Cp_func([1-delta_p, delta_p])

					# Calculate the regular confidence interval without partial feedback
					C = self.C(self.sigma[arm], self.F[arm], self.delta)


					# If the alternative confidence interval is tighter than the regular one, use it to calculate new upper and lower bounds.
					partial_used = False
					if C_combined < C:
						partial_used = True
						new_mu = (self.mu[arm] * self.F[arm] + self.mu_pa[b] + self.mu_bias[arm]) / (self.F[arm] + 1)
						self.ub[arm] = new_mu + C_combined
						self.lb[arm] = new_mu - C_combined
					else:
						new_mu = self.mu[arm]
						self.ub[arm] = new_mu + C
						self.lb[arm] = new_mu - C
				else:
					new_mu = self.mu[arm]

				kt = self.k - len(self.accepted)
				ub = []
				lb = []
				for arm2, pulls in self.surviving.items():
					ub.append((self.ub[arm2], arm2))
					lb.append((self.lb[arm2], arm2))
				ub = sorted(ub,reverse=True)
				lb = sorted(lb,reverse=True)

				# Update surviving arms
				update = False
				for i in range(len(self.surviving)):
					if lb[i][0] > ub[kt][0]:
						self.accepted[lb[i][1]] = self.surviving.pop(lb[i][1])
						update = True
					if ub[i][0] < lb[kt-1][0]:
						self.rejected[ub[i][1]] = self.surviving.pop(ub[i][1])
						update = True

				if arm not in self.surviving:
					# If arm was accepted / rejected and is running on another batch slot, kill that run to free
					# up the batch slot
					heappop(self.feedbacks[b])
					self.mu[arm] = new_mu
					if self.mode == 'unbiased':
						self.mu_pa[b] = 0.0
					self.available_arms[b] = 1

	def run(self):
		"""
		Runs the racing algorithm and identifies the top k arms
		"""

		rem_arms = self.n
		while rem_arms > 0:
			self.increment_time()
			rem_arms = len(self.surviving)

		return self.accepted

def parse_args():
	"""
	Specifies command line arguments for the program.
	"""

	parser = argparse.ArgumentParser(description='Best arm identification with partial and full delayed feedback.')
	
	parser.add_argument('--exp_type', default='none',
                      choices=['free_means', 'bounded_means', 'delay', 'none'],
                      type=str, help='choose none for simulating your own exp \
                      or use one of the other options to replicate results in the paper')

	parser.add_argument('--seed', default=1, type=int,
						help='Seed for random number generators')
	parser.add_argument('--num_tries', default=10, type=int,
						help='number of re')

	# default best-arm options
	parser.add_argument('--n', default=100, type=int,
						help='number of total arms')
	parser.add_argument('--delta', default=0.05, type=float,
						help='1 - target confidence')
	parser.add_argument('--k', default=20, type=int,
						help='number of top arms to be identified')
	parser.add_argument('--std', default=0.5, type=float,
						help='standard deviation for full feedback')

	# parallel MAB option
	parser.add_argument('--bsize', default=10, type=int,
						help='batch size for parallel MAB')

	# delay options
	parser.add_argument('--d', default=100, type=int,
						help='delay of feedback.')
	parser.add_argument('--pstd', default=0.01, type=float,
						help='standard deviation for partial feedback')

	return parser.parse_args()


if __name__ == '__main__':

	args = parse_args()
	np.random.seed(args.seed)
	np.set_printoptions(threshold=np.inf)

	if args.exp_type == 'free_means' or args.exp_type == 'bounded_means':
		nvals = [10, 20, 50, 100]
		delta = 0.05
		bsize = 10
		for nval in nvals:
			print('number of arms:', nval)
			k = int(0.2 * nval)
			full_delayed_timesteps = 0.
			partial_feedback_timesteps = 0.

			for try_idx in range(args.num_tries):
				print('random try:', try_idx+1)
				if args.exp_type == 'free_means':
					sim_full_delayed = Simulator(num_arms=nval, arm_means=np.random.permutation(2*nval - np.arange(nval)), arm_stds=0.5 * np.ones(nval), delay=100 * np.ones(nval), batch_size=bsize)
				else:
					sim_full_delayed = Simulator(num_arms=nval, arm_means=np.random.permutation(5 - (np.arange(nval) / float(nval))**0.6), arm_stds=0.03 * np.ones(nval), delay=50 * np.ones(nval), batch_size=bsize)
				agent_full_delayed = RacingDF(k, delta, sim_full_delayed)
				best_arm = agent_full_delayed.run()
				full_delayed_timesteps += agent_full_delayed.t
				print('full delayed timesteps:', agent_full_delayed.t)

				if args.exp_type == 'free_means':
					sim_partial_feedback = Simulator(num_arms=nval, arm_means=np.random.permutation(2*nval - np.arange(nval)), arm_stds=0.5 * np.ones(nval), delay=100 * np.ones(nval), partial_stds=0.25 * np.ones(nval), batch_size=bsize)
				else:
					sim_partial_feedback = Simulator(num_arms=nval, arm_means=np.random.permutation(5 - (np.arange(nval) / float(nval))**0.6), arm_stds=0.03 * np.ones(nval), delay=50 * np.ones(nval), partial_stds=0.03 * np.ones(nval), batch_size=bsize)
				agent_partial_feedback = RacingDF(k, delta, sim_partial_feedback, partial_feedback='unbiased')
				best_arm = agent_partial_feedback.run()
				partial_feedback_timesteps += agent_partial_feedback.t
				print('partial feedback timesteps', agent_partial_feedback.t)

			print('average full delayed timesteps:', full_delayed_timesteps/args.num_tries)
			print('average partial feedback timesteps:', partial_feedback_timesteps/args.num_tries)
			print()

	elif args.exp_type == 'delay':
		dvals = [10, 20, 50, 100, 200, 500, 1000]
		delta = 0.05
		bsize = 10
		n = 100
		k = int(0.2 * n)
		for dval in dvals:
			print('reward feedback delay:', dval)
			full_delayed_timesteps = 0.
			partial_feedback_timesteps = 0.

			for try_idx in range(args.num_tries):
				print('random try:', try_idx+1)
				sim_full_delayed = Simulator(num_arms=n, arm_means=np.random.permutation(2*n - np.arange(n)), arm_stds=0.5 * np.ones(n), delay=dval * np.ones(n), batch_size=10)
				agent_full_delayed = RacingDF(k, delta, sim_full_delayed)
				best_arm = agent_full_delayed.run()
				full_delayed_timesteps += agent_full_delayed.t
				print('full delayed timesteps:', agent_full_delayed.t)

				sim_partial_feedback = Simulator(num_arms=n, arm_means=np.random.permutation(2*n - np.arange(n)), arm_stds=0.5 * np.ones(n), delay=dval * np.ones(n), partial_stds=0.25 * np.ones(n), batch_size=10)
				agent_partial_feedback = RacingDF(k, delta, sim_partial_feedback, partial_feedback='unbiased')
				best_arm = agent_partial_feedback.run()
				partial_feedback_timesteps += agent_partial_feedback.t
				print('partial feedback timesteps', agent_partial_feedback.t)

			print('average full delayed timesteps:', full_delayed_timesteps/args.num_tries)
			print('average partial feedback timesteps:', partial_feedback_timesteps/args.num_tries)
			print()

	else:
		n = args.n
		k = args.k 
		bsize = args.bsize
		delta = args.delta

		# these parameters can also be set individually for each arm
		arm_means = np.random.permutation(2*n - np.arange(n))
		arm_stds = args.std * np.ones(n)
		delay = args.d * np.ones(n)
		pstd = args.pstd * np.ones(n)

		sim = Simulator(num_arms=n, arm_means=arm_means, arm_stds=arm_stds, delay=delay, partial_stds=pstd, batch_size=bsize)

		agent_partial_feedback = RacingDF(k, delta, sim, partial_feedback='unbiased')
		top_arms = list(agent_partial_feedback.run().keys())
		print('top-k arms:', top_arms)
		print('timesteps elapsed:', agent_partial_feedback.t)

