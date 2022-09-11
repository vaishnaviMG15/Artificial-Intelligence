import numpy as np
import random
from Queue import PriorityQueue
import math
from decimal import Decimal
import matplotlib
import matplotlib.pyplot as plt
from time import time

#helper function: prints a two dimensional array in a readable format
def helper_print(arr):

	for i in arr:
		print i

#helper method to generate a list of neighboring positions
def neighbors(point, arr):

	dim = len(arr)
	(i,j) = point
	list = [] 

	list.append((i-1,j-1))
        list.append((i-1,j))
        list.append((i-1,j+1))
	list.append((i+1,j-1))
        list.append((i+1,j))
        list.append((i+1,j+1))
	list.append((i,j+1))
	list.append((i,j-1))
	
	
	toRemove = []

	for k in list:
		(n,m) = k
		if n < 0 or n >= dim or m < 0 or m >= dim :
			toRemove.append(k)

	for y in toRemove:
		list.remove(y)

	return list

#helper function to get: 
#1. total # of neighbors
#2. number of revealed safe neighbors 
#3. number of revealed mine neighbors 
#4. number of hidden neighbors 
#5. list of hidden neighbors

def getValues(point, env):
	list = neighbors(point, env)
	n = len(list)
	rs = 0
	rm = 0
	h = 0
	hidden = []
	for x in list:
		(i,j) = x
		if env[i,j] == 'h':
			h+=1
			hidden.append(x)
		elif env[i,j] == 'M':
			rm+=1
		else:
			rs+=1
	return n, rs, rm, h, hidden


#helper method to get the number of neighboring mines

def mineNum(point, env):
	list = neighbors(point, env)

	#This variable will hold the number of neighboring mines
	count = 0

	for x in list:
		(i,j) = x
		if env[i,j] == 'M':
			 count += 1
	

	return count


#This method takes a dimension dim and number num
#to create a dim*dim array with num random mines

def generateEnv(dim, num):

	#creating an np array
	arr = np.array([['e']*dim]*dim)

	#choose num random locations for mines
	
	#i goes from 0 to num-1
	for i in range(num):
		#in each iteration choose one random free location to set as mine
		if (dim**2 - i) > 1:
			n = random.randint(1, dim**2 - i)
		else:
			n = 1

		#find the nth free location and set as mine

		free = 0

		j = 0
		
		while j < dim:
			k=0
			while k <dim:

				if arr[j,k] == 'e':
                                	free += 1

				if free == n and arr[j,k] != 'M':
					arr[j,k] = 'M'
					j = dim + 10
					k = dim + 10
				
				k+=1
			j+=1 
		

	#in all of the locations not as a mine, fill in the appropriate number

	for i in range (dim):
		for j in range (dim):

			if arr[i,j] != 'M':
				#assign a number
				arr[i,j] = mineNum((i,j), arr)


	return arr

# This function returns the game environment (the one the user has to solve) and reality environment

def generateGame(dim, num):

	#creating an np array
        arr = np.array([['h']*dim]*dim)

	return arr, generateEnv(dim,num)

#This function prints out the environment of the game and solution as a picture that is easier to understand
def imageEnv(env):
	fig, ax=plt.subplots()
	
	dim = len(env)
	arr = np.array([[0]*dim]*dim)
	
	for i in range(dim):
		for j in range(dim):
			if env[i,j]=='M':
				arr[i,j]= 10
			elif env[i,j]=='h':
				arr[i,j]=9
			else:
				arr[i,j]=int(env[i,j])

	cmap=plt.cm.binary
	cmap.set_bad(color='r')
	im=ax.imshow(arr, interpolation='none', cmap=cmap, alpha=0.6)
	
	for i in range(dim):
    		for j in range(dim):
      			text = ax.text(j, i, env[i, j], ha="center", va="center", color="b")

	ax.set_title("MineSweeper")
	fig.tight_layout()
	plt.show()


#helper function for basicAgent
#return True and a list of points to explore next if there are revealed safe cells that needs further analysis
#else it returns false and none
def toAnalyze(env, explored):

	dim = len(env)
	list = []
	for i in range (dim):
		for j in range (dim):
			if env[i,j] != 'h' and env[i,j] != 'M' and (i,j) not in explored:
				list.append((i,j))		

	return list

#helper function for basic Agent
#returns a random hidden point to reveal next
def randomHidden(env):

	list = []

	for i in range (len(env)):
		for j in range (len(env)):

			if env[i,j] == 'h':
				list.append((i,j))
	if len(list) == 0:
		return None
	elif len(list) == 1:
		n = 0
	else:
		n = random.randint(0, len(list) - 1)

	return list[n]

#returns the hidden point in the game using improved cell selection

def selectedHidden(game):

	solDict = {}

	for i in range (len(game)):
		for j in range (len(game)):

			if game[i,j] != 'h' and game[i,j] != 'M':
				
				neighborNum, rsNum, rmNum, hNum, hiddenList = getValues((i,j), game)
				
				if hiddenList != []:
					p = (int(game[i,j]) - rmNum)*1.0/hNum
					for hCell in hiddenList:
						if hCell not in solDict:
							solDict[hCell] = p
						else:
							curr = solDict[hCell]
							solDict[hCell] = max(curr, p)

	min = 2

	if solDict == {}:
		return randomHidden(game)

	for key in solDict:
		if solDict[key] < min:
			min = solDict[key]
			minkey = key

	return minkey						 

#This function plays the basic agent
#It returns the score, which is between 0 and 1
#final score = # of mines safely identified / total number of mine

#Our knowledge base is represented by the 2D game.
#Each point in the 2D game can either be:
#	-> 'M' for a definite mine
#	-> A number between 0 and the number of neighbors at the point which represents the number
#	   of mine neighbors. This is the case where a clear position has been revealed	
#	-> 'C' for a definite clear point which has not been revealed yet
#	-> 'h' if we cannot make any conclusion about this unrevealed point

def basicAgent(dim, m):
	if m != -1:
		num = (int((dim**2) * m))
	else:
		num = random.randint(1, dim**2)
	#print num
	game, solution = generateGame(dim, num)
	
	
	#print "Solution:"
	#helper_print(solution)
	#print " "
	#print "Game:"
	#helper_print(game)
	#print " "

	#The knowledge base (all the info we know at a certain point in the game) could just be 
	#represented by the game array which is gradually unpacked
	#All the information we need to keep track of can be derived from this knowledge base at any point
		
	#currently, we did not identify any mine
	count = 0
	#once this count becomes equal to dim**2  we can stop the process.
	
	#variable score keeps track of the number of mines carefully identified.	
	score = 0.0	
	
	#The below list holds all the safe spots that have been completely explored
	#A safe point is completely explored if we know the identity of all of its neighbors
	#We do not have to do any analysis at these points
 
	explored = []

	while count < dim**2:
		
		#get everything you can currently explore
		list = toAnalyze(game, explored)

		couldDetermine = False

		for x in list:
			#apply rules given in proj description to try to identify neighboring hidden cells
			neighbors, revealedSafe, revealedMine, hidden, hList = getValues(x,game)

			(i,j) = x

			if (int(game[i,j])) - revealedMine == hidden :
				for y in hList:
					(m,n) = y
					game[m,n] = 'M'
					count += 1
					score += 1.0

				explored.append(x)
				couldDetermine = True

			elif neighbors - (int(game[i,j])) - revealedSafe == hidden:
				for y in hList:
                                        (m,n) = y
                                        game[m,n] = solution[m,n]
					count += 1

                                explored.append(x)
				couldDetermine = True

		if couldDetermine == False:
			randomPoint = randomHidden(game)
			(i,j) = randomPoint
			game[i,j] = solution[i,j]
			count += 1

		#helper_print(game)
		#print " "

	finalScore = score/num
	return round(finalScore, 3)
				
#helperFunction that is used to update the KB based on information of a new location
#input parameters:
#KB: knowledge base
#location: (x,y) whose info should be updated in KB
#neighbors = all the hidden neighbors of (x,y)
#clue: the number of mines in the hidden neighbors (derived from clue)
#In case we are revealing (x,y) as a mine, neighbors will be None and clue will be -1
#if we are updating the knowledge base based on an assumption of safe cell, then neighbors is None and  clue is -2

def updateKB(KB, location, neighbors, clue):

	identSafe = []
	identMine = []

	#if the location is safe:
	#add a row in the KB

	if neighbors != None and len(neighbors) >= 1:

		newTuple = (neighbors, clue)
		if newTuple not in KB:
			KB.append(newTuple)
		
		
	#go through all rows in KB and update them
	
	for x in KB:
		(list, c) = x
		if location in list:
			KB.remove(x)
			list.remove(location)
			if clue == -1:
				if c-1 == 0:
					for y in list:
						identSafe.append(y)
				elif len(list) == c-1:
					for y in list:
						identMine.append(y)
				else:
					if len(list) >= 1 and (list, c-1) not in KB:
						KB.append((list, c-1))

			else:
				if len(list) == c:
					for y in list:
						identMine.append(y)
				else:
					if len(list) >= 1 and (list, c) not in KB:
						KB.append((list, c))
	

	for x in KB:
		(list, c) = x 
		if c == 0:
			KB.remove(x)
			for y in list:
				identSafe.append(y)	
		elif list==[]:
			KB.remove(x)

	return KB, identSafe, identMine

#helper function that returns a list of all the hidden cells
def getHiddenCells(game):

	dim = len(game)
	list = []

	for i in range(dim):
		for j in range(dim):
			if game[i,j] == 'h':
				list.append((i,j))
		

	return list

#parameters: A knowledge base, list of assignments for certain positions, and the game
#return Value: True if all assignments conform to rules in KB and state of the game, else False
#assignments: list of 2 tuples. Each tuple: (position in game, 0/1)

def isSatisfiable(KB, assignments, game):

	

	dictAssignments = {}
	for y in assignments:
		(key, value) = y
		dictAssignments[key] = value

	
	for rule in KB:
		(ruleList, ruleNum) = rule
		sum = 0
		count = len(ruleList)
		for location in ruleList:
			(v,w) = location
			if (v,w) in dictAssignments:
				sum = sum + dictAssignments[location]
				count -= 1
			else:
				if game[v,w] == 'M':
					count -= 1
					sum = sum + 1
				elif game[v,w] != 'h':
					count -= 1
			
			if count<ruleNum-sum:
				return False
			
			if sum>ruleNum:
				return False
	
		if count == 0:	
			if sum != ruleNum:
				return False

	
	return True 
 
		

#constraint saticfaction problem
#determining if there is a valid assignment for all the hidden cells in the KB
#constraints: rules of the KB
#domain for each variable: 0 or 1 (it may not have a mine or it may have a mine)
#returns true if constraints can be satisfied else false
#we use the DFS algorithm
#since in this search there is no way to loop back to a state before we can use Tree Search
#We are checking if assuming that the location is a mine (isMine = 1) or is not a mine (isMine = 0) would cause a contradiction

def queryKB(KB, location, isMine, game):
	
	#print "Querying location:"
	#print location

	#print "isMine:"
	#print isMine

	#print "Game:"
	#print game

	#print "KB:"
	#print KB

	(l1,l2) = location
	#create a new KB and a new game which are updated with information we want to verify

	newKB = []
	for rule in KB:
		(ruleList, ruleValue) = rule
		ruleListCopy = ruleList[:]
		newKB.append((ruleListCopy, ruleValue))


	newGame = game.copy()
	if isMine == 1:
		newKB, identSafe, identMine = updateKB(newKB, location, None, -1 )
		newGame[l1,l2] = 'M'
	else:
		newKB, identSafe, identMine = updateKB(newKB, location, None, -2)
		newGame[l1,l2] = 'C'

	for valueS in identSafe:
		(s1, s2) = valueS
		newGame[s1, s2] = 'C'

 
	for valueM in identMine:
		(m1, m2) = valueM
		newGame[m1, m2] = 'M'

	#main idea:
	#check if the newKB is satisfiable 
	#meaning there exists a variable assignment for all hidden cells conforming to rules in newKB
	#if it is satisfiable return true, else false

	#print "newGame:"
	#print newGame

	#print "newKB:"
	#print newKB

	#getting the list of all remaining hidden cells which appear in KB
	hiddenList = locsInKB(newKB)

	illegalLocs = []
	for hListValue in hiddenList:
		(h_x, h_y) = hListValue
		if newGame[h_x,h_y] != 'h':
			illegalLocs.append(hListValue)

	
	for iValue in illegalLocs:
		hiddenList.remove(iValue)
		 
	
	
	#for simplicity level i of variable assignment tree can represents an additional assignment for
	#the location at hiddenList[i]

	#initializing the fringe
	fringe = []
	if hiddenList != []:
		fringe.append([(hiddenList[0], 0)])
		fringe.append([(hiddenList[0], 1)])
	

	while fringe :
		#removing the newest state in the fringe
		currState = fringe.pop()
		#early termination: identifying contradictions at an earlier level of the tree	
		if len(currState)!=len(hiddenList):
			if not isSatisfiable(newKB, currState, newGame):
				continue
		
		if len(currState) == len(hiddenList):
			#check if this complete assignment is satisfiable. If so, return True
			
			if isSatisfiable(newKB, currState, newGame):
				return True
			else:
				continue

		#if not, get the children of this state
			
		childState1 = currState[:]
		childState2 = currState[:]

		childState1.append((hiddenList[len(currState)], 0))
		childState2.append((hiddenList[len(currState)], 1))

		fringe.append(childState1)
		fringe.append(childState2)
	
	#if we came out of the loop without getting a valid assignment for all levels, then KB is not satisfiable
	return False	

#Helper function to update the KB and retrieve information about safe and mine cells identified
def updateHelper(KB, game, isMine, value, listSafe, listMine):

	(x,y) = value
	list1=[]
	list2=[]
	if isMine == False:
		
		neighborsNum, revSafe, revMine, hiddenNum, hiddNeighbors = getValues(value, game)
		if game[x,y] !='M':
			KB, list1, list2 = updateKB(KB, value, hiddNeighbors, int(game[x,y]) - revMine)
		for i in list1:
			if i not in listSafe:
				listSafe.append(i)

		for i in list2:
			if i not in listMine:
				listMine.append(i)

	else:
	
		KB, list1, list2 = updateKB(KB, value, None, -1)
		for i in list1:
			if i not in listSafe:
				listSafe.append(i)

		for i in list2:
			if i not in listMine:
				listMine.append(i)

	return KB, listSafe, listMine

#returns a list of locations that appear in the rules of the KB
#useful to select values to query and to select locations to assign values to in method queryKB

def locsInKB (KB):
	
	list = []
	for x in KB:
		(l,c) = x
		for y in l:
			if y not in list:
				list.append(y)

	return list

#Here 'm' is the mine density between 0.05 and 1
#m would be -1 if the density is not provided in which case a random number of mines is used
#type mentions how we want to run the advanced agent:
#type = 1:Normal advanced agent
#type = 2:Global Information extra credit
#type = 3:Better Decisions extra credit
#The return value is the score which is the number of mines safely identified/total number of mines
def advancedAgent(dim, m, type):
	if m != -1:
		num = (int((dim**2) * m))
	else:
		num = random.randint(1, dim**2)
	# generate the game and knowledge base
	
	
	game, solution = generateGame(dim, num)

	#imageEnv(solution)
	#imageEnv(game)	
	count1 = 0
	count2 = 0

	#keeps track of the number of cells discovered so far
	#The process ends when count becomes equal to dim**2 (total # of cells)
	count = 0

	#keeps track of the number of safely identified cells
	score = 0.0

	#knowledge base
	KB = []

	KBList = []
	#condition for global information type of agent 
	if type == 2:
		for i in range(dim):
			for j in range(dim):
				KBList.append((i,j))

		KB.append((KBList, num))
		if len(KBList) == num:
			#we know everything is a mine
			game = solution
			score = dim**2
			count = dim**2		

	#would have everything identified as safe
	listSafe = []

	#would have everything identified as a mine
	listMine = []

	while count < dim**2 :

		couldDetermine = False
		#updating our knowledge based on the already identified but yet reveal safe cells and mine cells

		for safeValue in listSafe:
			(x,y) = safeValue
			listSafe.remove(safeValue)
			if game[x,y] == 'h':
				game[x,y] = solution[x,y]
				#imageEnv(game)
				count += 1
				KB, listSafe, listMine = updateHelper(KB, game, False, safeValue, listSafe, listMine)
				couldDetermine = True

		#imageEnv(game)
		
		for mineValue in listMine:
			(x,y) = mineValue
			listMine.remove(mineValue)
			if game[x,y] == 'h':
				game[x,y] = solution[x,y]
				#imageEnv(game)
				count += 1
				score += 1.0
				KB, listSafe, listMine = updateHelper(KB, game, True, mineValue, listSafe, listMine)
				couldDetermine = True
		

		#imageEnv(game)
		#Make listSafe and listMine are completely empty before you move on

		#if list safe and list mine are not empty then continue

		if not (len(listSafe)==0 and len(listMine) == 0):
			continue
	
		 

		#query wether or not any of the hidden cells can be indentified as safe or mine
			
		locationsToQuery = locsInKB(KB)

		for location in locationsToQuery:
				
			(x,y) = location

			if game[x,y] == 'h':

				#assume its a mine
				#if the KB is unsatisfiable, the cell is safe
				#reveal it 
				#update KB
				var = queryKB(KB, (x,y), 1, game)
				if var == False:
					
					if game[x,y] == 'h':
						couldDetermine = True
						game[x,y] = solution[x,y]
						#imageEnv(game)
						count += 1
						count1 += 1
						KB, listSafe, listMine = updateHelper(KB, game, False, (x,y), listSafe, listMine)

						#make sure listSafe and listMine are updated before you move on
					
						
						while ((len(listSafe) != 0) or (len(listMine) != 0)):						
							for safeValue in listSafe:
								(x,y) = safeValue
								listSafe.remove(safeValue)
								if game[x,y] == 'h':
									game[x,y] = solution[x,y]
									#imageEnv(game)
									count += 1
									KB, listSafe, listMine = updateHelper(KB, game, False, safeValue, listSafe, listMine)
							#imageEnv(game)

							for mineValue in listMine:
								(x,y) = mineValue
								listMine.remove(mineValue)
								if game[x,y] == 'h':
									game[x,y] = solution[x,y]
									#imageEnv(game)
									count += 1
									score += 1.0
									KB, listSafe, listMine = updateHelper(KB, game, True, mineValue, listSafe, listMine)
							#imageEnv(game)

				else:
	
					#assume its safe
					#if the KB is unsatisfiable, the cell is a mine
					#mark this
					#update KB
							
					var2 = queryKB(KB, (x,y), 0, game)
	
								
					if var2 == False:
	
						if game[x,y] == 'h':
							couldDetermine = True
							game[x,y] = 'M'
							#imageEnv(game)
							count += 1
							count2 += 1
							score += 1.0
							KB, listSafe, listMine = updateHelper(KB, game, True, (x,y), listSafe, listMine)
							#make sure listSafe and listMine are updated before you move on
							
							while ((len(listSafe) != 0) or (len(listMine) != 0)):						
								for safeValue in listSafe:
									(x,y) = safeValue
									listSafe.remove(safeValue)
									if game[x,y] == 'h':
										game[x,y] = solution[x,y]
										#imageEnv(game)
										count += 1
										KB, listSafe, listMine = updateHelper(KB, game, False, safeValue, listSafe, listMine)
								#imageEnv(game)

								for mineValue in listMine:
									(x,y) = mineValue
									listMine.remove(mineValue)
									if game[x,y] == 'h':
										game[x,y] = solution[x,y]
										#imageEnv(game)
										count += 1
										score += 1.0
										KB, listSafe, listMine = updateHelper(KB, game, True, mineValue, listSafe, listMine)
								#imageEnv(game)


	
		 
		
		#randomly select one of the hidden nodes, reveal it and update KB

		if couldDetermine == False and listSafe == [] and listMine == []:
			#condition for better decisions agent
			if type == 3:
				randomPoint = selectedHidden(game)
			else:	
				randomPoint = randomHidden(game)
			
			if randomPoint == None:
				break

			(i,j) = randomPoint

			
			if game[i,j] == 'h':
				game[i,j] = solution[i,j]
				#imageEnv(game)
				count += 1

				if game[i,j] == 'M':
					 KB, listSafe, listMine = updateHelper(KB, game, True, (i,j), listSafe, listMine)
				else:
					 KB, listSafe, listMine = updateHelper(KB, game, False, (i,j), listSafe, listMine)

	
	#print "# of time we identify something as safe from Query KB"
	#print count1
	#print "# of times we identify something as mine from Query KB"
	#print count2
	#print "score"
	#print score
	#print "Final game"
	#helper_print (game)
	final_score = score/num
	return round(final_score, 3)



#helper function to display the performance graphs
def printGraph(s1, s2):
	if s2=={}:
		x_values1 = []
		y_values1 = []
		num = 0.05
		while num <= 1 :
			x_values1.append(num)
			y_values1.append(s1.get(num))
			num += 0.05
			num = round(num,2)
		plt.plot(x_values1,y_values1,color='blue', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
		ymax=max(y_values1)

		plt.ylim(0, ymax)
		plt.xlim(0,1)

		plt.show()

	else:	
		x_values1 = []
		y_values1 = []

		x_values2 = []
		y_values2 = []

		num = 0.05
		while num <= 1 :
			x_values1.append(num)
			y_values1.append(s1.get(num))
			num += 0.05
			num = round(num,2)

		num = 0.05
		while num <= 1 :
			x_values2.append(num)
			y_values2.append(s2.get(num))
			num += 0.05
			num = round(num,2)

		plt.plot(x_values1,y_values1,color='blue', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
		plt.plot(x_values2,y_values2,color='red', linestyle='dashed', linewidth = 3, marker='*', markerfacecolor='red', markersize=12)
	
		ymax1=max(y_values1)
		ymax2=max(y_values2)
	
		ymax=max(ymax1, ymax2)

		plt.ylim(0, ymax)
		plt.xlim(0,1)

		plt.show()

#generate the point for the plots in the graph comparing the basic agent with the advanced agent
def performanceGraph():
	prev1 = {}
	prev2 = {}
	runs = 25
	num = 0.05

	while num<=1:
		count1 = count2 = 0.0
		i = 0
		while i < runs:
			count1+=basicAgent(30, num)
			count2+=advancedAgent(30, num, 1)

			i+=1
		prev1[num] = count1/runs
		prev2[num] = count2/runs
		num += 0.05
		num = round(num,2)
	
	printGraph (prev1, prev2)

#generate the point for the plots in the graph using global information of knowing 
#the number of mines in the maze comparing the basic agent with the advanced agent
def globalInfoGraph():
	prev2 = {}
	runs = 20
	num = 0.05

	while num<=1:
		count2 = 0.0
		i = 0
		while i < runs:
			count2+=advancedAgent(5, num, 2)
			i+=1
		prev2[num] = count2/runs
		num += 0.05
		num = round(num,2)

	printGraph (prev2, {})

#generate the point for the plots in the graph using better decisions 
#comparing the basic agent with the advanced agent
def betterDecisionsGraph():
	prev2 = {}
	runs = 20
	num = 0.05

	while num<=1:
		count2 = 0.0
		i = 0
		while i < runs:
			count2+=advancedAgent(5, num, 3)

			i+=1
		prev2[num] = count2/runs
		num += 0.05
		num = round(num,2)
	
	printGraph (prev2, {})

#main frunction for testing

#arr1, arr2 = generateGame(5, 6)
	
#helper_print(arr1)

#print "   "

#helper_print(arr2)
	

#score = advancedAgent(5, 0.5, 1)

#print score

#score = advancedAgent(5, 0.5, 1)

#print score


#score = advancedAgent(5, 1, 2)

#print score

#arr1, arr2 = generateGame(10, 25)
#imageEnv(arr2)

#performanceGraph()

#globalInfoGraph()

#betterDecisionsGraph()


