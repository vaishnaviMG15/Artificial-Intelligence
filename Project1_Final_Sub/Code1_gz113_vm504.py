import numpy as np
import random
from Queue import PriorityQueue
import math
from decimal import Decimal
import matplotlib.pyplot as plt
from time import time


#helper function: prints a two dimensional array in a readable format
def helper_print(arr):

	for i in arr:
		print i

#helper function: checks if 2 tuples (each one represents a state: (row, column)) are equal
def areTuplesEqual (t1, t2):

	if (t1[0] == t2 [0]) and (t1[1] == t2[1]) :
		return True
	else:
		return False

#helper function: returns a list of all the valid states from this point in the maze
#from the neighboring positions within the array bounds, it excludes the filled positions
def validChildren (maze, point):
	
	validStates = []
	(row, col) = point	
	dim = len(maze)
	p1 = row+1
	p2 = row-1
	p3 = col+1
	p4 = col-1

	#adding all the neighboring states as long as they are not out of bounds
	if row != dim - 1 :
		validStates.append((p1, col))
	if row != 0:
		validStates.append((p2, col))
	if col != 0:
		validStates.append((row, p4))
	if col != dim - 1 :
		validStates.append((row, p3))

	#filtering out the spots which are filled and cannot be reached
	#print validStates
	illegal = []
	for x in validStates:
		(a,b) = x
		if maze[a, b] == 'f':
			illegal.append(x)
	for y in illegal:
		validStates.remove(y)
	return validStates

#Question 1
#returns a maze with dimension 'dim' and probability of each cell being occupied 'p'
#The maze is a 2D square array of characters
#character 'S' represents upper left corner
#character 'G' represents lower right corner
#character 'e' represents the unoccupied states
#character 'f' represents the filled states/obstacles
def create_maze (dim, p) :
	
	#checking that the dimension is valid
	if dim <= 1:
		print("Choose a dimension greater than 1. For a size of 1 the source and goal are same so there is automatically a path")
		exit()

	#creating an np array
	arr = np.array([['e']*dim]*dim)

	#initializing start and goal
	arr[0,0] = 'S'
	arr[dim-1, dim-1] = 'G'

	#traversing the array to create obstacles based on probability
	for i in range(dim):
		for j in range(dim):
			if i == 0 and j == 0:
				continue
			if i == (dim-1) and j == (dim - 1):
				continue
			if random.uniform(0,1) <= p:
				arr[i,j] = 'f'
	return arr

#Question 2: using DFS to detrmine if there is a path between 2 points or not
#each state can be represented as a two tuple of the form (row #, column #)
#point1 is the initial state
#point2 is the goal state
#from each state the children states will be legal neighbor states
#illegal states are those whixh have 'f' (filled) or an 'x' (on fire). 
#Note: I added conditions for fire to make this method helpful for the second part of the project
#it is possible to loop back to a previous state, so I will use Graph search
#DFS: fringe is a stack (which I will implement using a python list)
#the closed set represents all the states we have explored.
#by exploring a state I mean getting all the neighbour states

def isPath (maze, point1, point2) :

	(a,b) = point1
	(e1,e2) = point2

	#checking that point 1 and 2 are not illegal
	if maze[a,b] == 'f' or maze[a,b] == 'x':
		return False
	if maze[e1,e2] == 'f' or maze[e1,e2] == 'x':
		return False

	#initializing the fringe and closed set
	fringe = []
	fringe.append(point1)
	closed_set = []


	while fringe :
		#removing the newest state in the fringe
		current_state = fringe.pop()

		#checking if the state is the destination
		if areTuplesEqual (current_state, point2) :
			return True

		#if not, explore the children of this state, in case this state has not been explored
		if current_state not in closed_set :
			
			childStates = validChildren (maze, current_state)
			#filtering out children who have already been explored or who are on fire
			for x in childStates:
				(c,d) = x
				if x not in closed_set and maze[c,d] != 'x':
 					fringe.append(x)
			closed_set.append(current_state)
		
	
	return False

#Question 2: Graph of P(G can be reached from S) vs obstactcle density 'p' for the isPath method [DFS]
#p: 0, 0.05, 0.1, 0.15,... 1
#the dim value we used: 100
def part2Graph (dim) :
	
	#prev is used to store all the points in this graph in (x, y) format
	prev = {}
	runs = 20
	p = 0
	while p <= 1:
		count = 0.0
		for i in range(runs):
			maze = create_maze(dim, p)
			if isPath(maze, (0,0), (dim-1, dim-1)):
				count += 1.0
		prev[p] = count/runs
		p += 0.05
		p = round(p,2)

	#This function is coded at a later part. It basically displays the grapg represented by points in prev
	printGraph(prev)

#helper_function: takes a dictionary and converts it into a list which represents the path
#this is a useful function to retrieve the shortest path from source to destination in upcoming questions
def getPath(prev,G):
	
	reverse_list = [G]
	parent = prev[G]
	while parent is not None :
		reverse_list.append(parent)
		parent = prev[parent]
	reverse_list.reverse()
	return reverse_list

#Question3: BFS to determine shortest path between two points (start and end)
#start and end are tuples of the form (r,c) representing location in maze	
#fringe: queue
#implementing this in the form of a graph search where a closed set is used to prevent exploring already explored states
#output: 
#if there is a path: True, list representing the path (this list includes the start and end states), # of nodes explored
#if no path : False, None, # of nodes explored
			 
def shortestBFS (maze, start, end) :
	
	(a,b) = start
	(e1,e2) = end

	#making sure start and end are valid
	if maze[a,b] == 'f' or maze[a,b] == 'x':
		return False, None, 0

	if maze[e1,e2] == 'f' or maze[e1,e2] == 'x':
		return False, None, 0

	dim = len(maze)
	
	#initializing fringe, closed sets, prev, explored
	#prev: used to keep track of the path
	#explored: to keep track of explored nodes

	fringe = []
	fringe.append(start)
	closed_set = []
	prev = {}
	prev[start] = None
	explored = 0
	
	#explore the state space as long as fringe is not empty meaning we cannot explore further
	while fringe :
		
		#pulling out the oldest state in the fringe
		current_state = fringe.pop(0)
		
		#checking if we have reached the end
		if areTuplesEqual (current_state, end) :
			return True, getPath(prev,end), explored+1 
		
		#exploring current_state if it hasnt benn explored before
		if current_state not in closed_set :
			
			explored += 1
			childStates = validChildren (maze, current_state)

			#filtering out the child states that have already been explored or that are on fire
			for x in childStates:
				(c,d) = x
				if x not in closed_set and maze[c,d] != 'x':
					prev[x] = current_state
					fringe.append(x)
 			#now that state has been explored it can be added to the closed set.
			closed_set.append(current_state)
		
	#if we did not visit the 'end' after exploring as much as possible, we return unsuccessfully	
	return False, None, explored

#helper for question 3 A star : construct heuristic table from maze
#returns a 2D arraywith each position holding the euclidean distance from itself to the destination node
def heuristicGenerator(maze, end):
	
	(a,b) = end
	dim = len(maze)
	result = np.array([[0.0]*dim]*dim)
	for i in range(dim):
		for j in range(dim):
			#euclidean distance btw (i,j) and (a, b) = sqrt((i-a)^2 + (j-b)^2)
			result[i,j] = math.sqrt((i-a)**2 + (j-b)**2)
	return result 


#Question 3: Using A star to determine shortest path between two points on maze
#heuristic: euclidean distance
#fringe: priority queue
#Mainitaining closed set to keep track of explored states
#if path exits, return value: True, shortest path represented by a list, # of nodes explored
#else, return value: False, None, # of nodes explored

def shortestA (maze, start, end):

	(v,w) = start
	(e1,e2)= end

	#making sure start and end are valid
	if maze[v,w] == 'f' or maze[v,w] == 'x':
		return False, None, 0

	if maze[e1,e2] == 'f' or maze[e1,e2] == 'x':
		return False, None, 0
	
	#determining heuristic values for each state
	#state (i,j) will have heuristic h[i,j]
	h = heuristicGenerator(maze, end)
	
	#initializing fringe, closed sets, prev, dist, explored
	#prev: used to keep track of the path
	#dist: used to keep track of the current smallest utility of the node
	#explored: to keep track of explored nodes

	explored = 0
	prev = {}
	prev[start] = None
	dist = {}
	closed_set = []
	fringe = PriorityQueue()
	(a,b) = start
	startValue = 0 + h[a,b] 
	fringe.put((startValue, start))
	dist[start] = startValue
	
	
	#explore the state space as long as fringe is not empty meaning we cannot explore further
	while not fringe.empty() :
		
		#pulling out the state in the fringe with least value
		current = fringe.get()
		
		(cDis, cState) = current
		(c,d) = cState
	
		#checking if we have reached the end	
		if areTuplesEqual (cState, end) :
			return True, getPath(prev,end), explored+1
			#print(prev)
			#return True, None, explored+1

		#exploring current_state if it hasnt been explored before
		if cState not in closed_set:
			
			explored += 1  
			childStates = validChildren(maze, cState)
			
			#filtering out the child states that have already been explored or that are on fire 
			for x in childStates:
				(e,f) = x
				
				#the value of the child state will be:
				#value of parent state - parent state's heuristic + 1 + childState's heuristic
				#1 indicates the distance between child and parent
				x_value = cDis - h[c,d] + 1+ h[e,f]
				check = True
				if x in dist:
					if x_value >= dist[x]:
						check = False
				if x not in closed_set and check and maze[e,f] != 'x':
					prev[x] = cState	
					fringe.put((x_value ,x))
					dist[x] = x_value
					
					

			#now that state has been explored it can be added to the closed set.
			closed_set.append(cState)

	#if we did not visit the 'end' after exploring as much as possible, we return unsuccessfully
	return False, None, explored

#Question 3: Graph of avg(# of nodes explored by BFS - # of nodes explored by BFS) vs. obstacle density 'p'
#p: 0, 0.05, 0.1, ..., 1
#dim: 80
def part3Graph (dim) :
	#prev stores the (x,y) points in the graph 	
	prev = {}
	runs = 20
	p = 0.0
	while p <= 1:
		count = 0.0
		for i in range(runs):
			maze = create_maze(dim, p)
			a, b, c = shortestBFS(maze, (0,0), (dim-1, dim-1))
			d, e, f = shortestA(maze, (0,0), (dim-1, dim-1))
			diff = c-f
			count += diff
		prev[p] = count/runs
		p += 0.05
		p = round(p,2)

	#used to display the graph
	printGraph(prev)

#helper method to draw the graph from a dictionary with keys as x coordinates and values as y coordinates 
def printGraph(points):

	x_values = []
	y_values = []

	
	p = 0.0
	while p <= 1 :
		
		x_values.append(p)
		y_values.append(points.get(p))
		p += 0.05
		p = round(p,2)
		

	plt.plot(x_values,y_values,color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)

	ymax = max(y_values) 
	plt.ylim(0,ymax)
	plt.xlim(0,1)

	plt.show()


#Question 4: to determine maximum dimension our system can handle in less than a minute with p = 0.3 for the different search algorithms

def maxDims () :

	p = 0.3
	dim = 5
	while True:
		start = time()
		maze = create_maze(dim, p)
		isPath(maze, (0,0), (dim-1, dim-1))
		end = time()
		diff = end - start
		if (diff > 60 ) :
			break
		dim += 1
		
	print('Maximum dimension with DFS = ' + str(dim - 1))


	dim = 5
	while True:
		start = time()
		maze = create_maze(dim, p)
		shortestBFS(maze, (0,0), (dim-1, dim-1))
		end = time()
		diff = end - start
		if (diff > 60 ) :
			break
		dim += 1
		
	print('Maximum dimension with BFS = ' + str(dim - 1))

	dim = 5
	while True:
		start = time()
		maze = create_maze(dim, p)
		shortestA(maze, (0,0), (dim-1, dim-1))
		end = time()
		diff = end - start
		if (diff > 60 ) :
			break
		dim += 1
		
	print('Maximum dimension with A star = ' + str(dim - 1))


#function to create an initial maze for second part of project
#returns the maze and the initial position of the fire
#cells of fire: x

def create_fire_maze (dim, p) :

	maze = create_maze(dim, p)
	counter = 0

	#counting # of empty cells
	for i in range(dim):
		for j in range(dim):
			if maze[i,j] == 'e' :
				counter += 1

	if counter < 1:
		return maze, None
	
	#initiating a fire in a random empty cell
	start = random.randint(1, counter)
	counter2 = 0
	i = 0
	position = (0,0)
	
	while i < dim:
		j = 0
		while j < dim:
			if maze[i,j] == 'e':
				counter2 += 1
				if counter2 == start:
					maze[i,j] = 'x'
					position = (i,j)
					i = dim
					j = dim
					
				
		
			j += 1
		i += 1
	
	return maze, position

#helper_function: to get the neighbors and the number of neighbors of a position (r,c) which are on fire
def neighborsOnFire (maze, position):

	list1 = validChildren(maze, position) #list of all neighbors which are not obstacles
	counter = 0
	result = []
	for x in list1:
		(a,b) = x
		if maze[a,b] == 'x' :
			result.append(x)
			counter += 1
	return result, counter

#helper_function: to expand fire for the second part of project
def advance_fire_one_step (maze, q):

	dim = len(maze)
	maze1 = maze.copy()
	for i in range(dim):
		for j in range(dim):
			
			
			if maze[i,j] == 'e' or maze[i,j] == 'S' or maze[i,j] == 'G' :
				
				l, k = neighborsOnFire(maze, (i,j))
				prob = 1 - ((1-q)**k)
				if random.uniform(0,1) <= prob and prob != 0:
					maze1[i,j] = 'x'
	
	return maze1

#Implementation of Strategy 1 for second part of project
#determines wether or not an agent is successfully able to complete maze

def strategy1 (maze, q, start, end):
	
	#getting the initial path
	b, path, nodes = shortestA(maze, start, end)
	(e1, e2) = end 
	
	if not b:
		return False, "No path"
	if len(path) == 1:
		return True, "Success"
	path.remove(start)
	agent = start
	
	#in the loop: advancing fire and making agent take next step 
	for x in path:
		agent = x
	
		maze = advance_fire_one_step (maze, q)
		(c,d) = x
		if maze[c,d] == 'x':
			return False, "Burned"
		if maze[e1, e2] == 'x':
			return False, "No path"
	return True, "Success"

#Implementation of Strategy 2 for second part of project
#determines wether or not an agent is successfully able to complete maze
def strategy2 (maze, q, start, end):

	agent = start
	#in the loop: calculating new path, advancing agent by one step, and advancing the fire by one step

	while True:
		b, path, nodes = shortestA(maze, agent, end)
	
		if path is None:
			(x,y) = agent
			if nodes == 0 and maze[x,y] == 'x':
				return False, "Burned"
			return False, "No path"
		elif len(path) == 1 :
			return True, "Success"
		elif len(path) == 2:
			agent = path[1]
			return True, "Success"
		else:
			agent = path[1]
			maze = advance_fire_one_step(maze, q)
	
  

#helper to find shortest distance btw point and the fire using BFS
#This will be useful in the next function.

def mod_BFS(maze, point):
	(a,b) = point
	dim = len(maze)
	fringe = []
	fringe.append(point)
	closed_set = []
	prev = {}
	prev[point] = None
	while fringe:
		current_state = fringe.pop(0)
		(c,d) = current_state
		if maze[c,d] == 'x':
			return len(getPath(prev, current_state)) - 2
		if current_state not in closed_set:
			childStates = validChildren(maze, current_state)
			for x in childStates:
				if x not in closed_set:
					prev[x] = current_state
					fringe.append(x)
			closed_set.append(current_state)

	return 0 #This is just a placeholder

#helper_function for strategy 3
#returns a 2D array representing the heuristic of each state of the maze with respect to the fire
#The value of the heuristic would be:
#0: if q=0 
#if adjacent to fire then:
#3*2*dim: if q=1
#m*2*dim: for any other state based on the number of neighbors 'm' on fire
#if not adjacent to the fire then dim/k

def fire_heuristics(maze, q):
	
	dim = len(maze)
	result = np.array([[0.0]*dim]*dim)
	for i in range(dim):
		for j in range(dim):
			if maze[i,j] != 'x' and maze[i,j] != 'f':
				if q==0: #blocks not on fire will never catch fire
					continue

				#calulating # of blocks in shortest path between the point and fire in the maze 
				disToFire = mod_BFS(maze, (i,j))

				if disToFire == 0:  #this point has neighbors which are on fire
			
					list, num = neighborsOnFire(maze, (i,j))
					if q == 1 and num>0: #this position will definitely catch on fire. Should have high heuristic
						result[i,j] = 3*2*dim
					
					else:
						result[i,j] = (num)*2*dim

				else:
					
					result[i,j] = (dim*1.0/disToFire)

	return result
	
 
#This modified AStar would try to find the path from start to end
#It will work similar to the regualr Astar
#however, it uses an extra heuristic which is described in the method above
#returns true and path for such a path. Else it returns False and none
   
def modified_AStar (maze, start, end, q):

	dim = len(maze)
	(s1,s2) = start
	(e1,e2)= end
	if maze[s1,s2] == 'f' or maze[s1,s2] == 'x':
		return False, None

	if maze[e1,e2] == 'f' or maze[e1,e2] == 'x':
		return False, None
	
	h1 = heuristicGenerator(maze, end)
	h2 = fire_heuristics(maze,q)
	
	prev = {}
	prev[start] = None
	dist = {}
	closed_set = []
	fringe = PriorityQueue()
	startValue = 0 + h1[s1,s2] + h2[s1,s2]
	fringe.put((startValue, start))
	dist[start] = startValue

	while not fringe.empty():
		current = fringe.get()
		(cDis,cState) = current
		(c,d) = cState

		if areTuplesEqual (cState, end):
			return True, getPath(prev, end)

		if cState not in closed_set:
			childStates = validChildren(maze,cState)

			for x in childStates:
			
				(e,f) = x
				x_value = cDis - h1[c,d] - h2[c,d] + 1 + h1[e,f] + h2[e,f]
				check = True
				if x in dist:
					if x_value >= dist[x]:
						check = False
				if x not in closed_set and check and maze[e,f]!=x :
					prev[x] = cState
					fringe.put((x_value, x))
					dist[x] = x_value			

			closed_set.append(cState)
		
	return False, None

				
#Implementation of Strategy 3
#similar to strategy 2 but uses modified Astar
#By using modified Astar the intention is that whenever the algorithm recomputes the path at any step,
#it recomputes a safer path that takes into consideration the distance of positions from the fire.
#We try to promote survivability by pushing the agent into a comparitively safe path at every step

def strategy3 (maze, q, start, end):

	agent = start
	#print agent
	while True:
		
		b, path = modified_AStar (maze, agent, end, q)
		
		if path is None:
			(x,y) = agent
			if maze[x,y] == 'x':
				return False, "Burned"
			return False, "No path"
		elif len(path) == 1:
			return True, "Success"
		elif len(path) == 2:
			agent = path[1]
			return True, "Success"
		else:
			agent = path[1]
			maze = advance_fire_one_step(maze, q)
			#helper_print(maze)
		#print agent

#helper function to construct final graph
#using DFS to check if there is a path from point1 to point2 in case point2 is on fire
#this is helpful for discarding the mazes that do not have a path from 'S' to the initial position of the fire.

def isPathFire (maze, point1, point2) :

	(a,b) = point1
	(e1,e2) = point2
	if maze[a,b] == 'f' or maze[a,b] == 'x':
		return False
	if maze[e1,e2] == 'f':
		return False

	fringe = []
	fringe.append(point1)
	closed_set = []
	while fringe :
		
		current_state = fringe.pop()
		
		if areTuplesEqual (current_state, point2) :
			return True

		if current_state not in closed_set :
			#print current_state
			childStates = validChildren (maze, current_state)
			for x in childStates:
				(c,d) = x
				if x not in closed_set:
					if (x == point2 and maze[c,d] == 'x') or maze[c,d] != 'x':
 						fringe.append(x)
			closed_set.append(current_state)
		#print closed_set
		#print fringe
	
	return False

#Question 6: graph for strategies
#1: strategy 1
#2: strategy 2
#3: strategy 3
#p is fixed at 0.3
#return value: dictionary; key: x, value: y

def strategyGraph (dim):

	prev1 = {}
	prev2 = {}
	prev3 = {}
	runs = 20
	q = 0

	while q<=1:
		count1 = count2 = count3 = 0.0
		i = 0
		while i < runs:
			maze, pos = create_fire_maze(dim, 0.3)
			if pos is None:
				continue
			b1 = isPath(maze, (0,0), (dim-1, dim-1))
			b2 = isPathFire(maze, (0,0), pos)
			if (b1 == False) or (b2 == False):
				continue
			a,b = strategy1(maze, q, (0,0), (dim-1, dim-1))
			d,e = strategy2(maze, q, (0,0), (dim-1, dim-1))
			g,h = strategy3(maze, q, (0,0), (dim-1, dim-1))
			if a:
				count1 += 1
			if d:
				count2 += 1
			if g:
				count3 += 1
			i+=1
		prev1[q] = count1/runs
		prev2[q] = count2/runs
		prev3[q] = count3/runs
		q += 0.05
		q = round(q,2)
	
	printSGraph (prev1, prev2, prev3)

#helper function to display the final graph
#blue line: strategy 1
#red line: strategy 2
#green line: strategy 3	
def printSGraph(s1, s2, s3):

	x_values1 = []
	y_values1 = []

	x_values2 = []
	y_values2 = []

	x_values3 = []
	y_values3 = []

	q = 0.0
	while q <= 1 :
		x_values1.append(q)
		y_values1.append(s1.get(q))
		q += 0.05
		q = round(q,2)

	q = 0.0
	while q <= 1 :
		x_values2.append(q)
		y_values2.append(s2.get(q))
		q += 0.05
		q = round(q,2)

	q = 0.0
	while q <= 1 :
		x_values3.append(q)
		y_values3.append(s3.get(q))
		q += 0.05
		q = round(q,2)

	plt.plot(x_values1,y_values1,color='blue', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
	plt.plot(x_values2,y_values2,color='red', linestyle='dashed', linewidth = 3, marker='*', markerfacecolor='red', markersize=12)
	plt.plot(x_values3,y_values3,color='green', linestyle='dashed', linewidth = 3, marker='.', markerfacecolor='green', markersize=12)
	
	ymax1=max(y_values1)
	ymax2=max(y_values2)
	ymax3=max(y_values3)
	
	ymax=max(ymax1, ymax2, ymax3)

	plt.ylim(0, ymax)
	plt.xlim(0,1)

	plt.show()
		

	

#main : used for testing


#testing create_maze

#maze = create_maze(10, 0.1)
#maze = create_maze(5, 0)
#maze = create_maze(10, 0.5)
#helper_print(maze)


#Testing isPath (DFS)
#print isPath (maze, (0,0), (9,9)) 

#Testing shortestBFS

#d, e, f = shortestBFS (maze, (0,0), (4,4))
#print(d)
#print(e)
#print(f)

#Testing shortestA

#a, b, c = shortestA(maze, (0,0), (4,4))
#print(a)
#print(b)
#print(c)

#generating graphs for problems 2 and 3

#part2Graph(200)
#part3Graph(150)


#testing strategy 3

#maze, pos = create_fire_maze(5,0.1)
#helper_print(maze)
#a,b = strategy3(maze, 0.2, (0,0), (4,4))
#print a
#print b

#testing isPathFire
#maze, pos = create_fire_maze(5,0.4)
#helper_print(maze)
#print isPathFire(maze, (0,0), pos)


#maze = np.array([['S','e','e','e','e'],['e','e','x','x','e'],['e','e','f','e','e'],['e','e','e','e','e'],['e','e','e','e','G']])

#strategyGraph(20)

#maze, pos = create_fire_maze(5, 0.2)
#helper_print(maze)
#print('A star')
#a, b, c = shortestA(maze, (0,1), (4,4))
#d, e = modified_AStar(maze, (0,1), (4,4), 0.3)
#print a
#print b

#print('modified')
#print d
#print e 

#helper_print(maze)
#b1, s1 = strategy2 (maze, 0.3, (0,0), (4,4))
#print(b1)
#print(s1)

#print(' ')
#print('strategy 3')
#print(' ')

#helper_print(maze)
#b2, s2 = strategy3 (maze, 0.5, (0,0), (4,4))
#print(b2)
#print(s2)

#maze = np.array([['S','e','e','x','e','e','e'],['e','e','e','x','e','f','e'],['e','e','e','f','e','e','e'],['e','f','e','e','e','e','e'],['e','f','e','f','x','f','e'],['e','e','e','f','e','f','e'],['e','e','e','f','e','e','G']])


#helper_print(maze)

#a,b = modified_AStar(maze, (0,0), (6,6), 0.3)
#print a
#print b

#print mod_BFS(maze, (0,5))
#print mod_BFS(maze, (1,6))

#a, b, c = shortestA(maze, (0,0), (6,6))
#print(a)
#print(b)
#print(c)
