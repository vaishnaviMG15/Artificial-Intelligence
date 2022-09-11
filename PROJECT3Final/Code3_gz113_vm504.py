import numpy as np
import random
import math
from decimal import Decimal

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

#This function creates the environment which is a 2D square array of dimension 50*50
#Each location is a character
#'f': flat
#'h': hilly
#'t': forested
#'m': maze of caves
#The return values are:
#1. The 2D environment
#2. The location of the target (This is a tuple)

def create_env():

	#creating an np array
	arr = np.array([['x']*50]*50)

	#traversing through array and setting each cell as a particular type
	#based on probability

	for i in range(50):
		for j in range(50):
			# get a random number in the range [0.0, 1.0)
			p = random.random()
			if p < 0.25:
				arr[i,j] = 'f'
			elif p < 0.5:
				arr[i,j] = 'h'
			elif p < 0.75:
				arr[i,j] = 't'
			else:
				arr[i,j] = 'm'
	
	return arr

#useful to get a random initial point and target location
def getRandomLocation():
 
	#50*50 array has 2500 total values

	r = random.randint(0,2499)
	
	#The row rth location belongs to
	row = r/50
	
	#The column rth location belongs to
	col = r % 50

	target = (row, col)

	return target

#This function takes in 3 arguments:
#1.tuple/location to search
#2.tuple/location of target
#3.terrain type of location being searched
#returns False if search result is 'not found'
#returns True if search result is found 
def resultOfSearch(location, target, terrain):

	isTarget = areTuplesEqual(location, target)

	#if the location is not target the search result is False
	if (isTarget == False):
		return False
	
	#at this point we know we are searching the target
	#probability of 'found' depends on conditional probabilities of false negatives
	
	r = random.random()

	if (terrain == 'f'):

		if r < 0.1:
			return False
		else:
			return True

	elif (terrain == 'h'):
	
		if r < 0.3:
			return False
		else:
			return True

	elif (terrain == 't'):
	
		if r < 0.7:
			return False
		else:
			return True

	elif(terrain == 'm'):

		if r < 0.9:
			return False
		else:
			return True
 
#helper function to get manhattan distance between two points
def getManhattanDistance(point1, point2):
	
	(a,b) = point1
	(c,d) = point2
	result = abs(a-c) + abs(b-d)

	return result

#This is a helper function to determine the next location to explore
#function takes 2 inputs:
#1.The current location of agent
#2.belief array of agent based on all observations till this point
#3.type of agent (1, 2, 3)
#4.The environment env
#returns 2 values
#1. tuple of next location to explore
#2. manhattan distance to this location from current location

def getNext(point, belief, type, env):
	bMax = 0.0
	dis = 0
	next = (0,0)

	for i in range(50):
		for j in range(50):

			val = 0
			
			if env[i,j] == 'f':
				val = 1-0.1
			elif env[i,j] == 'h':
				val = 1-0.3
			elif env[i,j] == 't':
				val = 1-0.7
			elif env[i,j] == 'm':
				val = 1-0.9

			valueToCompare = 0
			if type == 1:
				valueToCompare = belief[i,j]
			else:
				valueToCompare = belief[i,j] * val

			valueToCompare = round(valueToCompare, 6)
 	
			if valueToCompare > bMax:
				bMax = valueToCompare
				dis = getManhattanDistance(point, (i,j))
				next = (i,j)
			elif valueToCompare == bMax:
				disTemp = getManhattanDistance(point, (i,j))
				if (disTemp < dis):
					dis = disTemp
					next = (i,j)


	#In case this is the advanced agent(type 3) we modify our next location as follows:

	newDis = 0
	if type == 3:
		#getting a list of locations in manhattan distance path to current destination cell	
		list = getList(point, next)
		#choosing the nearest point in this list with a good probability
		for element in list:
			if areTuplesEqual(element, point) == False:
				newDis += 1
			(ex, ey) = element
			if bMax - (belief[ex, ey] * val) < 0.01:
				next = element
				dis = newDis
				break


	return next, dis

#getting the list of points in a manhattan distance path between 'point' and 'next'
def getList(point, next):

	list = []

        (x1, y1) = point
        (x2, y2) = next
        vertical = x1
        if x1 > x2:
                while vertical >= x2:
                        list.append((vertical, y1))
                        vertical = vertical - 1

        elif x1 < x2:
                while vertical <= x2:
                        list.append((vertical, y1))
                        vertical = vertical + 1

        horizontal = y1
        if y1 > y2:
                while horizontal >= y2:
                        list.append((x2, horizontal))
                        horizontal = horizontal - 1

        elif y1 < y2:
                while horizontal <= y2:
                        list.append((x2, horizontal))
                        horizontal = horizontal + 1
	if point in list:	
		list.remove(point)
	if next in list:
		list.remove(next)

	return list


#helper function that returns a new updated belief array
#takes 3 inputs:
#1. Location where result was 'not found'
#2. Current belief array
#3. environment
def updateBelief(location, belief, env):

	(x,y) = location
	dim=len(env)
	#marginalization to get probability that target was not found in location (x,y)
	#This probability P(search at (x,y) is not found) will be the denominator in all the
	#upcoming baysian updates

	#This probability is equal to 
	#P(search at location is not found | target is not at location) * P(target is not at location) +
	#P(search at location is not found | target is at location) * P(target is at location) 

	#Note: P(target is/is not at location) is determined by the current belief which takes
	#all previous observations into consideration 
	
	val = 0

	if env[x,y] == 'f':
		val = 0.1
	elif env[x,y] == 'h':
		val = 0.3
	elif env[x,y] == 't':
		val = 0.7
	elif env[x,y] == 'm':
		val = 0.9

	den = (1.0 * (1-belief[x,y])) + (val * belief[x,y])
	
	#updating the belief at the location using bayesian theorem
	updatedxy = (val*belief[x,y])/den
	updatedxy = round(updatedxy,6)
	belief[x,y] = updatedxy

	#updating the belied at all other locations using bayesian theorem
	for i in range(dim):
		for j in range(dim):
			if areTuplesEqual(location, (i,j)):
				continue;
			
			updatedij = (1.0 * belief[i,j])/den 
			updatedij = round(updatedij, 6)
			belief[i,j] = updatedij


	return belief
			

#This function plays the 1st basic agent
#The return value is the score of the basic agent
#score = total distance travelled + number of searches
#lower scores indicate better performance
#parameter type can be 1 or 2
#1: we are playing basic agent 1
#2: we are playing basic agent 2
#3: we are playing advance agent 3
# The 3 agents only differ based on the next cell to explore
# parameter env is the environment
#env_target is the location of the target in the environment
#currLocation is the initial location of agent
def basic_agent(type, env, env_target, currLocation):
		

	#The agent knows about env but not the location of target 'env_target'

	#generate 50*50 array of values representing current belief state
	initial_belief_perCell = 1.0/2500
	initial_belief_perCell = round(initial_belief_perCell, 6)
	belief = np.array([[initial_belief_perCell]*50]*50)
	
	#to keep track of distance travelled and number of searches so far
	tDistance = 0
	numSearches = 0
	
	started = False

	#Keep playing the game until you find the target
	#each iteration is one one search in the env
	
	while True:

		if started == False:
			#the game has just started, pick a random cell to search
			
			(clx, cly) = currLocation
			found = resultOfSearch(currLocation, env_target, env[clx, cly])
			numSearches += 1
			started = True

		else:
			
			#get the highest probability 
			#pick cell with highest probability that is closest to the currentLocation

			currLocation, distance = getNext(currLocation, belief, type, env)
			(x,y) = currLocation
			found = resultOfSearch(currLocation, env_target, env[x,y])
			numSearches += 1
			tDistance += distance


		#at this point a new cell has been explored
		#Now based on result of exploration particular actions are taken

		if (found == True):
			#if the target has been found then agent is done
			break
		else:
			#update belief array based on this observation
			belief = updateBelief(currLocation, belief, env)				
				

	score = tDistance + numSearches
	return score

#This is a function used to compare the average performances of all the basic agents 	
def avg_agent():
	
	subavg1 = []
	subavg2 = []
	subavg3 = []
	runs = 0
	while runs < 10:
		env = create_env()
		temp1 = 0.0
		temp2 = 0.0	
		temp3 = 0.0
		subruns = 0
		while subruns < 10:
			target = getRandomLocation()
			initialPoint = getRandomLocation()	
			temp1 += basic_agent(1, env, target, initialPoint)
			print temp1
			temp2 += basic_agent(2, env, target, initialPoint)
			print temp2
			temp3 += basic_agent(3, env, target, initialPoint)
			print temp3
			print subruns
			subruns+=1

		score1 = temp1/10.0
		score2 = temp2/10.0
		score3 = temp3/10.0
		score1 = round(score1, 6)
		score2 = round(score2, 6)
		score3 = round(score3, 6)

		subavg1.append(score1)
		subavg2.append(score2)
		subavg3.append(score3)
		print runs
		runs+=1

	print subavg1
	print subavg2
	print subavg3

#main for testing




avg_agent()

#list = getList((4,4), (2,2))
#print list

#env = create_env()
#target = getRandomLocation()
#initialPoint = getRandomLocation()
#print "started"
#score2 = basic_agent(2, env, target, initialPoint)
#print score2
#score3 = basic_agent(3, env, target, initialPoint)
#print score3


